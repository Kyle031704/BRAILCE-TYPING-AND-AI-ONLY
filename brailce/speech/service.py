"""
Brailce Centralized Speech Service
===================================

Provides a unified, high-level speech service for all application modes.
Coordinates between:
    - Low-latency character-by-character feedback (eSpeak NG)
    - High-fidelity natural reading for words, sentences, and AI (Piper TTS)
    - Graceful fallback when Piper is unavailable
    - Non-overlapping queue and rapid-typing interruption

Configurable speech behaviors:
    - "character": announce each newly entered character
    - "character_and_word": announce each character and completed word at word boundary
    - "completed_text_only": read completed words and sentences as ordinary text
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple, Union

from brailce.speech.base import SpeechEngine
from brailce.speech.espeak_engine import EspeakEngine, clean_speech_text
from brailce.speech.piper_engine import PiperEngine

logger = logging.getLogger(__name__)

# Standard Grade 1 Letter Names (with period for falling, definitive pitch)
LETTER_NAME_MAP: Dict[str, str] = {
    "A": "A.",
    "B": "Bee.",
    "C": "See.",
    "D": "Dee.",
    "E": "Ee.",
    "F": "Eff.",
    "G": "Jee.",
    "H": "Aitch.",
    "I": "Eye.",
    "J": "Jay.",
    "K": "Kay.",
    "L": "Ell.",
    "M": "Em.",
    "N": "En.",
    "O": "Oh.",
    "P": "Pee.",
    "Q": "Cue.",
    "R": "Ar.",
    "S": "Ess.",
    "T": "Tee.",
    "U": "You.",
    "V": "Vee.",
    "W": "Double you.",
    "X": "Ex.",
    "Y": "Why.",
    "Z": "Zee.",
}

DIGIT_NAME_MAP: Dict[str, str] = {
    "0": "zero.",
    "1": "one.",
    "2": "two.",
    "3": "three.",
    "4": "four.",
    "5": "five.",
    "6": "six.",
    "7": "seven.",
    "8": "eight.",
    "9": "nine.",
}

MATH_SYMBOL_MAP: Dict[str, str] = {
    "+": "plus.",
    "-": "minus.",
    "*": "times.",
    "/": "divided by.",
    "=": "equals.",
}

PUNCTUATION_NAME_MAP: Dict[str, str] = {
    ".": "period.",
    ",": "comma.",
    "?": "question mark.",
    "!": "exclamation mark.",
    "'": "apostrophe.",
    '"': "quote.",
    ":": "colon.",
    ";": "semicolon.",
    "#": "number sign.",
}

VALID_BEHAVIORS = {"character", "character_and_word", "completed_text_only"}


class SpeechService(SpeechEngine):
    """
    Centralized speech coordinator for the Brailce device.
    Implements SpeechEngine interface for drop-in compatibility.
    """

    def __init__(
        self,
        character_engine: Optional[SpeechEngine] = None,
        text_engine: Optional[SpeechEngine] = None,
        behavior: str = "character_and_word",
        voice: str = "en-us+f3",
        rate: int = 150,
        volume: int = 100,
        pitch: int = 52,
    ) -> None:
        self._character_engine = character_engine or EspeakEngine(
            voice=voice, rate=rate, volume=volume, pitch=pitch
        )
        self._text_engine = text_engine or self._character_engine
        self._behavior = behavior if behavior in VALID_BEHAVIORS else "character_and_word"
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._pitch = pitch
        self._last_utterance: Optional[Tuple[str, str, Dict[str, Any]]] = None
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        c_name = self._character_engine.engine_name
        t_name = self._text_engine.engine_name
        if c_name == t_name:
            return f"SpeechService ({c_name})"
        return f"SpeechService (Char: {c_name}, Text: {t_name})"

    @property
    def character_engine(self) -> SpeechEngine:
        return self._character_engine

    @property
    def text_engine(self) -> SpeechEngine:
        return self._text_engine

    @property
    def behavior(self) -> str:
        return self._behavior

    @property
    def last_utterance(self) -> Optional[str]:
        if self._last_utterance:
            return self._last_utterance[1]
        return None

    def set_behavior(self, behavior: str) -> None:
        """Set the active speech feedback behavior."""
        if behavior in VALID_BEHAVIORS:
            self._behavior = behavior
            logger.info("SpeechService behavior set to: %s", behavior)
        else:
            logger.warning("Invalid speech behavior: %r (valid: %s)", behavior, VALID_BEHAVIORS)

    def is_available(self) -> bool:
        """Check if at least one speech engine is available."""
        return self._character_engine.is_available() or self._text_engine.is_available()

    def set_voice(self, voice: str) -> None:
        self._voice = voice
        self._character_engine.set_voice(voice)
        if self._text_engine is not self._character_engine:
            self._text_engine.set_voice(voice)

    def set_rate(self, rate: int) -> None:
        self._rate = rate
        self._character_engine.set_rate(rate)
        if self._text_engine is not self._character_engine:
            self._text_engine.set_rate(rate)

    def set_volume(self, volume: int) -> None:
        self._volume = volume
        self._character_engine.set_volume(volume)
        if self._text_engine is not self._character_engine:
            self._text_engine.set_volume(volume)

    def set_pitch(self, pitch: int) -> None:
        self._pitch = pitch
        if hasattr(self._character_engine, "set_pitch"):
            self._character_engine.set_pitch(pitch)
        if self._text_engine is not self._character_engine and hasattr(self._text_engine, "set_pitch"):
            self._text_engine.set_pitch(pitch)

    def stop_speech(self) -> None:
        """Stop all speech playback across all active engines."""
        self.stop()

    def stop(self) -> None:
        """Immediately stop currently playing audio and drain speech queues.
        Does NOT kill the worker thread — it remains ready for new speech."""
        try:
            if hasattr(self._character_engine, 'stop_current'):
                self._character_engine.stop_current()
            else:
                self._character_engine.stop()
        except Exception as e:
            logger.debug("Error stopping character engine: %s", e)

        if self._text_engine is not self._character_engine:
            try:
                if hasattr(self._text_engine, 'stop_current'):
                    self._text_engine.stop_current()
                else:
                    self._text_engine.stop()
            except Exception as e:
                logger.debug("Error stopping text engine: %s", e)

    def shutdown(self) -> None:
        """Fully shut down the speech service and worker threads.
        Called only during application shutdown."""
        try:
            self._character_engine.stop()
        except Exception as e:
            logger.debug("Error shutting down character engine: %s", e)

        if self._text_engine is not self._character_engine:
            try:
                self._text_engine.stop()
            except Exception as e:
                logger.debug("Error shutting down text engine: %s", e)

    # -----------------------------------------------------------------------
    # High-Level Speech Methods
    # -----------------------------------------------------------------------

    def speak_character(self, character: str) -> None:
        """
        Speak immediate, unambiguous letter name or character feedback.
        Respects speech behavior: suppressed if behavior is 'completed_text_only'.
        """
        if not character:
            return

        # In completed_text_only mode, individual character feedback is suppressed
        if self._behavior == "completed_text_only":
            return

        char_clean = character.strip()
        if not char_clean:
            return

        char_upper = char_clean.upper()

        # Resolve explicit pronunciation mapping
        if char_upper in LETTER_NAME_MAP:
            spoken_text = LETTER_NAME_MAP[char_upper]
        elif char_clean in DIGIT_NAME_MAP:
            spoken_text = DIGIT_NAME_MAP[char_clean]
        elif char_clean in MATH_SYMBOL_MAP:
            spoken_text = MATH_SYMBOL_MAP[char_clean]
        elif char_clean in PUNCTUATION_NAME_MAP:
            spoken_text = PUNCTUATION_NAME_MAP[char_clean]
        else:
            spoken_text = f"{char_clean}."

        self._record_utterance("character", spoken_text, {"character": character})

        # Use centralized eSpeak NG engine for sequential, non-overlapping speech
        with self._lock:
            self._character_engine.speak(spoken_text, interrupt=False)

    def speak_symbol(self, symbol: str, context: str = "text") -> None:
        """
        Speak a mathematical operator or punctuation mark in context.
        """
        if not symbol:
            return

        sym = symbol.strip()
        if context == "math" and sym == "*":
            spoken = "multiplied by."
        elif context == "math" and sym in MATH_SYMBOL_MAP:
            spoken = MATH_SYMBOL_MAP[sym]
        elif sym in PUNCTUATION_NAME_MAP:
            spoken = PUNCTUATION_NAME_MAP[sym]
        elif sym in MATH_SYMBOL_MAP:
            spoken = MATH_SYMBOL_MAP[sym]
        else:
            spoken = sym

        self._record_utterance("symbol", spoken, {"symbol": symbol, "context": context})

        with self._lock:
            self._character_engine.speak(spoken, interrupt=False)

    def speak_text(
        self,
        text: str,
        interrupt: bool = False,
        engine_hint: Optional[str] = None,
    ) -> None:
        """
        Speak complete words, sentences, math explanations, scanner output,
        or AI tutor responses as natural, fluent speech.

        Punctuation is preserved for cadence, natural pauses, and intonation,
        rather than announced as words.
        """
        if not text:
            return

        cleaned = clean_speech_text(text)
        if not cleaned:
            return

        self._record_utterance("text", cleaned, {"interrupt": interrupt, "engine_hint": engine_hint})

        # Select engine: Piper if available/configured, otherwise fallback to eSpeak
        target_engine = self._text_engine
        if engine_hint == "character" or not target_engine.is_available():
            target_engine = self._character_engine

        with self._lock:
            # If interrupt requested, stop both engines first
            if interrupt:
                self.stop()
            target_engine.speak(cleaned, interrupt=interrupt)

    def speak_math_expression(self, expression: str) -> None:
        """
        Evaluate and speak a mathematical expression with pedagogical clarity.
        """
        if not expression:
            return

        from brailce.braille.arithmetic import SafeArithmetic
        solution = SafeArithmetic.solve(expression)
        if solution.is_valid:
            self.speak_text(solution.explanation, interrupt=False)
        else:
            # Format expression into natural speech words if solve failed
            s = expression.replace("+", " plus ").replace("-", " minus ")
            s = s.replace("*", " times ").replace("/", " divided by ").replace("=", " equals ")
            s = " ".join(s.split())
            self.speak_text(s, interrupt=False)

    def replay_last_utterance(self) -> None:
        """Replay the most recently spoken utterance."""
        if not self._last_utterance:
            logger.debug("No previous utterance to replay.")
            return

        kind, text, kwargs = self._last_utterance
        logger.info("Replaying last utterance (%s): %r", kind, text)

        if kind == "character":
            char = kwargs.get("character", text)
            self.speak_character(char)
        elif kind == "symbol":
            sym = kwargs.get("symbol", text)
            ctx = kwargs.get("context", "text")
            self.speak_symbol(sym, context=ctx)
        else:
            self.speak_text(text, interrupt=True)

    def speak(self, text: str, interrupt: Optional[bool] = None) -> None:
        """
        Backward-compatible speak interface.
        Single characters are routed to speak_character;
        words and sentences are routed to speak_text.
        """
        if not text:
            return

        trimmed = text.strip()
        # Single character or character name lookup
        if len(trimmed) == 1:
            self.speak_character(trimmed)
            return

        upper_check = trimmed.upper()
        if upper_check in LETTER_NAME_MAP and len(upper_check) == 1:
            self.speak_character(upper_check)
            return

        # Whole words and sentences
        self.speak_text(text, interrupt=bool(interrupt))

    def _record_utterance(self, kind: str, text: str, kwargs: Dict[str, Any]) -> None:
        self._last_utterance = (kind, text, kwargs)

    # -----------------------------------------------------------------------
    # Factory / Loader
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "SpeechService":
        """
        Build SpeechService according to application configuration and environment.
        All modes share the exact same centralized eSpeak NG engine and speech queue.
        """
        speech_cfg = config.section("speech") if hasattr(config, "section") else {}

        behavior = os.getenv("SPEECH_BEHAVIOR", speech_cfg.get("behavior", "character_and_word")).lower()
        voice = speech_cfg.get("voice", "en-us")
        rate = int(speech_cfg.get("rate", 155))
        volume = int(speech_cfg.get("volume", 100))
        pitch = int(speech_cfg.get("pitch", 50))
        word_gap = int(speech_cfg.get("word_gap_ms", 10))
        max_queue = int(speech_cfg.get("max_queue_size", 50))

        # Instantiate unified, single eSpeak NG engine for all modes
        espeak = EspeakEngine(
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            word_gap=word_gap,
            max_queue_size=max_queue,
        )

        logger.info("SpeechService initialized with centralized eSpeak NG engine (voice: %s, rate: %d).", voice, rate)

        return cls(
            character_engine=espeak,
            text_engine=espeak,
            behavior=behavior,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
