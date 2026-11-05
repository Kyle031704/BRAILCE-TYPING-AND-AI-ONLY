"""
Unit tests for Filipino Engine & Mode Toggle
"""

import os
from unittest.mock import MagicMock
import pytest

from brailce.speech.filipino_engine import (
    FilipinoEngine,
    filipino_to_phonetic,
    get_filipino_word_path,
    get_language_announcement_path,
    get_thinking_audio_path,
)
from brailce.modes.free_input import FreeInputMode
from brailce.modes.ai_assistant import AIAssistantMode
from brailce.ai.ollama_client import OllamaClient, FILIPINO_SYSTEM_PROMPT
from brailce.core.events import EventType, InputEvent


def test_filipino_to_phonetic():
    assert filipino_to_phonetic("mga bata") == "mang ah bah tah."
    assert filipino_to_phonetic("ng aso") == "nang ah soh."
    assert filipino_to_phonetic("bahay") == "bah heye."
    assert filipino_to_phonetic("ikaw") == "ee kow."


def test_filipino_word_path():
    path_aso = get_filipino_word_path("aso")
    assert path_aso is not None
    assert os.path.isfile(path_aso)

    path_nonexistent = get_filipino_word_path("supercalifragilistic")
    assert path_nonexistent is None


def test_language_announcement_path():
    path_fil = get_language_announcement_path("fil")
    assert path_fil is not None
    assert os.path.isfile(path_fil)

    path_en = get_language_announcement_path("en")
    assert path_en is not None
    assert os.path.isfile(path_en)


def test_free_input_language_toggle():
    app = MagicMock()
    app.state.text_buffer = ""
    app.state.ai_processing = False
    app.state.language = "en"
    app.config.get.return_value = True

    mode = FreeInputMode(app)
    assert mode.language == "en"

    # Type "FIL" and press ENTER
    app.state.text_buffer = "FIL"
    event_enter = InputEvent(event_type=EventType.ENTER_PRESSED)
    mode.handle_event(event_enter)

    assert mode.language == "fil"
    app.state.clear_buffer.assert_called()

    # Type "EN" and press ENTER
    app.state.text_buffer = "EN"
    mode.handle_event(event_enter)

    assert mode.language == "en"


def test_thinking_audio_path():
    p = get_thinking_audio_path()
    assert p is not None
    assert os.path.isfile(p)


def test_dedicated_language_toggle_event():
    app = MagicMock()
    app.state.language = "en"
    app.state.ai_processing = False
    mode = FreeInputMode(app)

    event_toggle = InputEvent(event_type=EventType.LANGUAGE_TOGGLE)
    mode.handle_event(event_toggle)
    app.toggle_language.assert_called_once()


def test_ai_assistant_filipino_offline_response():
    app = MagicMock()
    app.state.language = "fil"
    app.state.ai_query_buffer = "kumusta"
    app.state.get_ai_query.return_value = "kumusta"
    app.config.get.return_value = True

    mode = AIAssistantMode(app)
    mode._finish_processing = MagicMock()

    mode._process_query("kumusta")
    mode._finish_processing.assert_called_once()
    called_response = mode._finish_processing.call_args[0][0]
    assert "Mabuti naman ako" in called_response
