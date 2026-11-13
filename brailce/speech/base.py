"""
Brailce Speech Engine – Abstract Base
=======================================

Defines the interface for all speech engines.

The application calls speak(), stop(), etc. without knowledge
of the underlying TTS technology.

Concrete implementations:
    - EspeakEngine    (eSpeak NG — offline)
    - MockSpeechEngine (unit testing)
"""

from __future__ import annotations

import abc
from typing import Optional


class SpeechEngine(abc.ABC):
    """Abstract interface for text-to-speech engines."""

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """
        Enqueue text for immediate speech output.

        For Free Input Mode, each character press calls this method
        with the single character or its speech representation.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop any currently playing speech."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check whether the TTS engine is available on this system."""

    @abc.abstractmethod
    def set_voice(self, voice: str) -> None:
        """Set the TTS voice/language."""

    @abc.abstractmethod
    def set_rate(self, rate: int) -> None:
        """Set the speech rate (words per minute)."""

    @abc.abstractmethod
    def set_volume(self, volume: int) -> None:
        """Set the speech volume (0–100)."""

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """Human-readable name for this speech engine."""

    # ---- Optional High-Level Speech Facade Defaults ----

    def speak_character(self, character: str) -> None:
        """Speak an individual character/keystroke."""
        self.speak(character)

    def speak_word_clean(self, word: str) -> None:
        """Speak a single completed word cleanly."""
        self.speak(word)

    def speak_word(self, word: str) -> None:
        """Alias for speak_word_clean."""
        self.speak_word_clean(word)

    def speak_sentence(self, text: str, interrupt: bool = False) -> None:
        """Speak a sentence or phrase."""
        self.speak(text)

    def speak_text(self, text: str, interrupt: bool = False) -> None:
        """Speak general text."""
        self.speak(text)

    def speak_symbol(self, symbol: str, context: str = "text") -> None:
        """Speak a math/punctuation symbol."""
        self.speak(symbol)
