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
    prepare_spoken_word,
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
        filipino_letter_voice: str = "filipino",
        filipino_letter_style: str = "english_names",
        filipino_voice: str = "friendly_prerendered_plus_offline",
        filipino_pitch_semitones: float = 3.5,
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
        self._language = "en"
        self._filipino_letter_voice = (filipino_letter_voice or "filipino").lower()
        self._filipino_letter_style = (filipino_letter_style or "english_names").lower()
        self._filipino_voice = (filipino_voice or "friendly_prerendered_plus_offline").lower()
        self._filipino_pitch_semitones = filipino_pitch_semitones
        self._filipino_engine = None
        self._last_utterance: Optional[Tuple[str, str, Dict[str, Any]]] = None
        self._lock = threading.RLock()

    @property
    def filipino_letter_style(self) -> str:
        return self._filipino_letter_style

    @filipino_letter_style.setter
    def filipino_letter_style(self, val: str) -> None:
        self._filipino_letter_style = (val or "english_names").lower()
        if self._filipino_engine is not None:
            self._filipino_engine.letter_style = self._filipino_letter_style

    @property
    def filipino_letter_voice(self) -> str:
        return self._filipino_letter_voice

    @filipino_letter_voice.setter
    def filipino_letter_voice(self, val: str) -> None:
        self._filipino_letter_voice = (val or "filipino").lower()

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, val: str) -> None:
        self._language = val.lower() if val else "en"

    @property
    def filipino_engine(self):
        if self._filipino_engine is None:
            from brailce.speech.filipino_engine import FilipinoEngine
            self._filipino_engine = FilipinoEngine(fallback_speech=self, pitch_semitones=getattr(self, "_filipino_pitch_semitones", 3.5))
        return self._filipino_engine

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
        """Immediately stop active speech and cancel queued items in both EN and FIL queues."""
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
        if self._filipino_engine is not None:
            try:
                self._filipino_engine.stop()
            except Exception:
                pass
        AudioDeviceCoordinator.interrupt_active()

    def shutdown(self) -> None:
        self.stop()

    # -----------------------------------------------------------------------
    # High-Level Speech Methods (Single Unified Queue per Language)
    # -----------------------------------------------------------------------

    def speak_character(self, character: str) -> None:
        """
        Speak individual keystroke feedback (A-Z, digits, operators, punctuation).
        In Filipino mode (when filipino_letter_voice == 'filipino'), uses the same
        Filipino voice queue. Otherwise uses Piper Ryan via the character engine queue.
        """
        if not character:
            return

        if self._behavior == "completed_text_only":
            return

        if self._language == "fil" and self._filipino_letter_voice == "filipino":
            self._record_utterance("character", character, {"character": character, "lang": "fil"})
            style = getattr(self, "filipino_letter_style", "english_names")
            self.filipino_engine.enqueue_character(character, use_filipino_voice=True, style=style)
            return

        normalized = normalize_for_piper(character, is_character=True)
        if not normalized:
            return

        self._record_utterance("character", normalized, {"character": character})

        with self._lock:
            self._character_engine.speak(normalized, interrupt=False)

    def speak_word_clean(self, word: str, interrupt: bool = False) -> None:
        """
        Single unified word-speaking entry point used by Space, Enter-on-single-word,
        Free Input, AI mode, Math mode, and Scanner mode.
        Completed words NEVER pass through single-letter phonetic mapping.
        Preserves trailing punctuation (. , ? ! :) for natural intonation (prosody).
        """
        if not word:
            return

        prepared = prepare_spoken_word(word, lang=self._language)
        if not prepared:
            return

        bare = prepared.rstrip(".!?,;:")
        if len(bare) == 1 and prepared.endswith("."):
            self.speak_character(bare)
            return

        if self._language == "fil":
            if interrupt:
                self.stop()
            self._record_utterance("word", prepared, {"word": prepared, "lang": "fil"})
            self.filipino_engine.enqueue_text(prepared, interrupt=interrupt)
            return

        if interrupt:
            self.stop()

        self._record_utterance("word", prepared, {"word": prepared})
        with self._lock:
            # Check cached word if standard period ending
            if prepared.endswith(".") and hasattr(self._engine, "play_cached_word") and self._engine.play_cached_word(bare):
                return
            self.speak_text(prepared, interrupt=interrupt)

    def speak_word(self, word: str) -> None:
        """Backward-compatible alias for speak_word_clean."""
        self.speak_word_clean(word)

    def speak_symbol(self, symbol: str, context: str = "text") -> None:
        """Speak mathematical operator or punctuation in context."""
        if not symbol:
            return

        if self._language == "fil" and self._filipino_letter_voice == "filipino":
            self._record_utterance("symbol", symbol, {"symbol": symbol, "context": context, "lang": "fil"})
            self.filipino_engine.enqueue_character(symbol, use_filipino_voice=True)
            return

        normalized = normalize_for_piper(symbol, context=context, is_character=True)
        if not normalized:
            return

        self._record_utterance("symbol", normalized, {"symbol": symbol, "context": context})

        with self._lock:
            self._character_engine.speak(normalized, interrupt=False)

    def speak_sentence(self, text: str, interrupt: bool = False) -> None:
        """Speak a complete sentence or completed word (on ENTER)."""
        if not text:
            return

        words = text.strip().split()
        if len(words) == 1:
            self.speak_word_clean(words[0], interrupt=interrupt)
            return

        self.speak_text(text, interrupt=interrupt)

    def speak_text(
        self,
        text: str,
        interrupt: bool = False,
        engine_hint: Optional[str] = None,
    ) -> None:
        """
        Speak sentences, math explanations, AI responses, or mode announcements.
        """
        if not text:
            return

        clean = text.strip().rstrip(".")
        if len(clean) == 1:
            self.speak_character(clean)
            return

        if self._language == "fil":
            if interrupt:
                self.stop()
            self._record_utterance("text", text, {"interrupt": interrupt, "lang": "fil"})
            self.filipino_engine.enqueue_text(text, interrupt=interrupt)
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

    def speak_mode_intro(self, mode: Any, interrupt: bool = True) -> str:
        """
        Return the localized mode intro text from i18n and enqueue it on the
        single unified speech queue for the active language (zero overlapping threads).
        """
        from brailce.i18n.strings import get_mode_intro
        intro = get_mode_intro(mode, language=self._language)

        if self._language == "fil":
            from brailce.speech.filipino_engine import get_filipino_mode_path
            mode_wav = get_filipino_mode_path(mode)
            if interrupt:
                self.stop()
            self._record_utterance("text", intro, {"mode": str(mode), "lang": "fil"})
            if mode_wav and os.path.isfile(mode_wav):
                self.filipino_engine.enqueue_wav(mode_wav, interrupt=False)
            else:
                self.filipino_engine.enqueue_text(intro, interrupt=False)
        else:
            self.speak_text(intro, interrupt=interrupt)
        return intro

    def speak_language_switch(self, new_lang: str, interrupt: bool = True) -> str:
        """
        Return localized language-switch announcement and enqueue it on the single queue.
        """
        from brailce.i18n.strings import get_string
        lang_key = "LANG_SWITCHED_FIL" if (new_lang or "en").lower() == "fil" else "LANG_SWITCHED_EN"
        phrase = get_string(lang_key, language=new_lang)

        if (new_lang or "en").lower() == "fil":
            from brailce.speech.filipino_engine import get_language_announcement_path
            ann_wav = get_language_announcement_path("fil")
            if interrupt:
                self.stop()
            self._record_utterance("text", phrase, {"lang_switch": "fil"})
            if ann_wav and os.path.isfile(ann_wav):
                self.filipino_engine.enqueue_wav(ann_wav, interrupt=False)
            else:
                self.filipino_engine.enqueue_text(phrase, interrupt=False)
        else:
            self.speak_text(phrase, interrupt=interrupt)
        return phrase

    def speak_math_expression(self, expression: str) -> None:
        """Evaluate and speak mathematical expression with teacher clarity."""
        if not expression:
            return

        from brailce.braille.arithmetic import SafeArithmetic
        solution = SafeArithmetic.solve(expression, language=self._language)
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
        Main entry point: routes EN through Piper queue and FIL through FilipinoEngine queue.
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

        if self._language == "fil":
            if interrupt:
                self.stop()
            self._record_utterance("text", trimmed, {"interrupt": bool(interrupt), "lang": "fil"})
            self.filipino_engine.enqueue_text(trimmed, interrupt=bool(interrupt))
            return

        self.speak_text(text, interrupt=bool(interrupt))

    def _record_utterance(self, kind: str, text: str, kwargs: Dict[str, Any]) -> None:
        self._last_utterance = (kind, text, kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "SpeechService":
        """
        Build SpeechService according to application configuration.
        Initializes Piper Neural TTS as the unified English engine and
        reads filipino_letter_voice toggle.
        """
        speech_cfg = config.section("speech") if hasattr(config, "section") else {}

        behavior = os.getenv("SPEECH_BEHAVIOR", speech_cfg.get("behavior", "character_and_word")).lower()
        voice = speech_cfg.get("model", "en_US-ryan-medium.onnx")
        rate = int(speech_cfg.get("rate", 160))
        volume = int(speech_cfg.get("volume", 100))
        pitch = int(speech_cfg.get("pitch", 50))
        max_queue = int(speech_cfg.get("max_queue_size", 50))
        fil_letter_voice = str(speech_cfg.get("filipino_letter_voice", "filipino")).lower()
        fil_voice = str(speech_cfg.get("filipino_voice", "sherpa_mms_tgl")).lower()
        fil_pitch = float(speech_cfg.get("filipino_pitch_semitones", 3.5))

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
            filipino_letter_voice=fil_letter_voice,
            filipino_voice=fil_voice,
            filipino_pitch_semitones=fil_pitch,
        )
