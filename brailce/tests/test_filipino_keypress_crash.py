"""
Regression test for Crash #2: SpeechService.speak_character calling enqueue_character with style=.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from brailce.speech.service import SpeechService
from brailce.speech.filipino_engine import FilipinoEngine


def test_filipino_keypress_style_parameter():
    """Verify FilipinoEngine.enqueue_character accepts style keyword argument without TypeError."""
    engine = FilipinoEngine()
    engine._enqueue = MagicMock()

    # Calling with style must not raise TypeError
    engine.enqueue_character("A", use_filipino_voice=True, style="english_names")
    engine._enqueue.assert_called_once_with("char", ("A", True, "english_names"), interrupt=False)


def test_speech_service_speak_character_filipino_no_crash():
    """Verify SpeechService.speak_character in Filipino mode calls enqueue_character with style safely."""
    mock_char = MagicMock()
    mock_text = MagicMock()

    fil_engine = FilipinoEngine()
    fil_engine.enqueue_character = MagicMock()

    service = SpeechService(
        character_engine=mock_char,
        text_engine=mock_text,
    )
    service._filipino_engine = fil_engine
    service.language = "fil"
    service.filipino_letter_voice = "filipino"
    service.filipino_letter_style = "english_names"

    # Must execute smoothly without TypeError
    service.speak_character("B")
    fil_engine.enqueue_character.assert_called_once_with("B", use_filipino_voice=True, style="english_names")
