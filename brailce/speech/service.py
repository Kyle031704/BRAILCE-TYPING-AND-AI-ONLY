"""
Brailce Centralized Speech Service
===================================

Unified speech coordinator:
- ALL voice output (characters, words, sentences, math, AI tutor, and mode announcements)
  passes through normalizer.py directly into Piper Neural TTS (Ryan).
- No fallback to eSpeak NG when Piper is available.
- No disk lookup of old broken char_*.wav files.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from brailce.speech.audio_device import AudioDeviceCoordinator
from brailce.speech.base import SpeechEngine
from brailce.speech.normalizer import (
    DIGIT_MAP,
    LETTER_MAP,
    MATH_OPERATORS,
    PUNCTUATION_MAP,
    normalize_for_piper,
)
from brailce.speech.piper_engine import PiperEngine

logger = logging.getLogger(__name__)

# Exported maps for backward-compatibility with tests & UI
LETTER_NAME_MAP: Dict[str, str] = {chr(c): f"{chr(c).lower()}." for c in range(ord("A"), ord("Z") + 1)}
DIGIT_NAME_MAP: Dict[str, str] = dict(DIGIT_MAP)
MATH_SYMBOL_MAP: Dict[str, str] = dict(MATH_OPERATORS)
PUNCTUATION_NAME_MAP: Dict[str, str] = dict(PUNCTUATION_MAP)

VALID_BEHAVIORS = {"character", "character_and_word", "completed_text_only"}


class SpeechService(SpeechEngine):
    """
    Centralized speech coordinator for the Brailce device.
    Directs all speech through Piper Neural TTS.
    """

    def __init__(
        self,
        character_engine: Optional[SpeechEngine] = None,
        text_engine: Optional[SpeechEngine] = None,
        piper_engine: Optional[SpeechEngine] = None,
        behavior: str = "character_and_word",
        voice: str = "en_US-ryan-medium.onnx",
        rate: int = 160,
        volume: int = 100,
        pitch: int = 50,
    ) -> None:
        engine = piper_engine or text_engine or character_engine or PiperEngine(rate=rate, volume=volume)
        self._engine = engine
        self._character_engine = character_engine or engine
        self._text_engine = text_engine or engine
        self._behavior = behavior if behavior in VALID_BEHAVIORS else "character_and_word"
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._pitch = pitch
        self._last_utterance: Optional[Tuple[str, str, Dict[str, Any]]] = None
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        return f"SpeechService ({self._engine.engine_name})"

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
        if behavior in VALID_BEHAVIORS:
            self._behavior = behavior
            logger.info("SpeechService behavior set to: %s", behavior)

    def is_available(self) -> bool:
        return self._engine.is_available()

    def set_voice(self, voice: str) -> None:
        self._voice = voice
        self._engine.set_voice(voice)

    def set_rate(self, rate: int) -> None:
        self._rate = rate
        self._engine.set_rate(rate)

    def set_volume(self, volume: int) -> None:
        self._volume = volume
        self._engine.set_volume(volume)

    def set_pitch(self, pitch: int) -> None:
        self._pitch = pitch
        if hasattr(self._engine, "set_pitch"):
            self._engine.set_pitch(pitch)

    def stop_speech(self) -> None:
        self.stop()

    def stop_current(self) -> None:
        self.stop()

    def stop(self) -> None:
        """Immediately stop active speech and cancel queued items."""
        try:
            self._engine.stop()
        except Exception as e:
            logger.debug("Error stopping engine: %s", e)
        if hasattr(self, "_character_engine") and self._character_engine is not self._engine:
            try:
                self._character_engine.stop()
            except Exception:
                pass
        if hasattr(self, "_text_engine") and self._text_engine is not self._engine and self._text_engine is not self._character_engine:
            try:
                self._text_engine.stop()
            except Exception:
                pass
        AudioDeviceCoordinator.interrupt_active()

    def shutdown(self) -> None:
        self.stop()

    # -----------------------------------------------------------------------
    # High-Level Speech Methods (All Routed Directly into Piper via normalizer)
    # -----------------------------------------------------------------------

    def speak_character(self, character: str) -> None:
        """
        Speak individual keystroke feedback (A-Z -> a., digits, operators, punctuation).
        Passes directly through normalizer.py into Piper.
        Uses instant pre-cached carrier WAV if available.
        """
        if not character:
            return

        if self._behavior == "completed_text_only":
            return

        normalized = normalize_for_piper(character, is_character=True)
        if not normalized:
            return

        self._record_utterance("character", normalized, {"character": character})

        # Try instant pre-cached carrier audio playback first
        if hasattr(self._character_engine, "play_cached_character"):
            if self._character_engine.play_cached_character(character):
                return

        with self._lock:
            self._character_engine.speak(normalized, interrupt=False)

    def speak_word(self, word: str) -> None:
        """
        Speak a completed word (on SPACE) through normalizer.py into Piper.
        If word is a single letter or character, route directly to speak_character()
        to play the clean cached character WAV.
        """
        if not word:
            return

        clean = word.strip().rstrip(".")
        if len(clean) == 1:
            self.speak_character(clean)
            return

        normalized = normalize_for_piper(word, is_word=True)
        if not normalized:
            return

        self._record_utterance("word", normalized, {"word": word})

        with self._lock:
            self._text_engine.speak(normalized, interrupt=False)

    def speak_symbol(self, symbol: str, context: str = "text") -> None:
        """Speak mathematical operator or punctuation in context."""
        if not symbol:
            return

        normalized = normalize_for_piper(symbol, context=context, is_character=True)
        if not normalized:
            return

        self._record_utterance("symbol", normalized, {"symbol": symbol, "context": context})

        with self._lock:
            self._character_engine.speak(normalized, interrupt=False)

    def speak_text(
        self,
        text: str,
        interrupt: bool = False,
        engine_hint: Optional[str] = None,
    ) -> None:
        """
        Speak sentences, math explanations, AI responses, or mode announcements.
        Passes directly through normalizer.py into Piper.
        If single character, routes to speak_character() to play clean cached audio.
        """
        if not text:
            return

        clean = text.strip().rstrip(".")
        if len(clean) == 1:
            self.speak_character(clean)
            return

        normalized = normalize_for_piper(text)
        if not normalized:
            return

        self._record_utterance("text", normalized, {"interrupt": interrupt, "engine_hint": engine_hint})

        target_engine = self._text_engine
        if engine_hint == "character" or not target_engine.is_available():
            target_engine = self._character_engine

        with self._lock:
            if interrupt:
                self.stop()
            target_engine.speak(normalized, interrupt=interrupt)

    def speak_math_expression(self, expression: str) -> None:
        """Evaluate and speak mathematical expression with teacher clarity."""
        if not expression:
            return

        from brailce.braille.arithmetic import SafeArithmetic
        solution = SafeArithmetic.solve(expression)
        if solution.is_valid:
            self.speak_text(solution.explanation, interrupt=False)
        else:
            self.speak_text(expression, interrupt=False)

    def replay_last_utterance(self) -> None:
        if not self._last_utterance:
            return

        kind, text, kwargs = self._last_utterance
        if kind == "character":
            self.speak_character(kwargs.get("character", text))
        elif kind == "word":
            self.speak_word(kwargs.get("word", text))
        else:
            self.speak_text(text, interrupt=True)

    def speak(self, text: str, interrupt: Optional[bool] = None) -> None:
        """
        Main entry point: passes through normalizer.py into Piper.
        """
        if not text:
            return

        trimmed = text.strip()
        if not trimmed:
            return

        clean = trimmed.rstrip(".")
        if len(clean) == 1:
            self.speak_character(clean)
            return

        self.speak_text(text, interrupt=bool(interrupt))

    def _record_utterance(self, kind: str, text: str, kwargs: Dict[str, Any]) -> None:
        self._last_utterance = (kind, text, kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "SpeechService":
        """
        Build SpeechService according to application configuration.
        Initializes Piper Neural TTS as the unified speech engine.
        """
        speech_cfg = config.section("speech") if hasattr(config, "section") else {}

        behavior = os.getenv("SPEECH_BEHAVIOR", speech_cfg.get("behavior", "character_and_word")).lower()
        voice = speech_cfg.get("model", "en_US-ryan-medium.onnx")
        rate = int(speech_cfg.get("rate", 160))
        volume = int(speech_cfg.get("volume", 100))
        pitch = int(speech_cfg.get("pitch", 50))
        max_queue = int(speech_cfg.get("max_queue_size", 50))

        piper = PiperEngine(
            rate=rate,
            volume=volume,
            max_queue_size=max_queue,
        )

        return cls(
            character_engine=piper,
            text_engine=piper,
            piper_engine=piper,
            behavior=behavior,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
