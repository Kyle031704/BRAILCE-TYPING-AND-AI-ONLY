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
)
from brailce.modes.free_input import FreeInputMode
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
