"""
Brailce Tests — Scanner Mode
============================

Tests live viewfinder lifecycle, missing camera reporting, presence hysteresis
+ cooldown, actionable scan errors, letter-by-letter + word-by-word readout,
key cancellation, and Clear reset behavior.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from brailce.core.app import BrailceApp
from brailce.core.events import AppMode, EventType, InputEvent
from brailce.core.state import ApplicationState
from brailce.i18n.strings import get_string
from brailce.modes.scanner_mode import ScannerMode


@pytest.fixture
def mock_app() -> BrailceApp:
    """Create a mock BrailceApp with real state and mocked subsystems."""
    app = MagicMock(spec=BrailceApp)
    app.state = ApplicationState()
    app.display = MagicMock()
    app.speech = MagicMock()
    app.camera = MagicMock()
    app.camera.is_available.return_value = True
    app.camera.capture_image.return_value = np.full((240, 320, 3), 200, dtype=np.uint8)
    app.obr = MagicMock()
    app.config = MagicMock()
    app.config.get.return_value = False
    return app


@pytest.fixture
def scanner_mode(mock_app: BrailceApp) -> ScannerMode:
    mode = ScannerMode(mock_app)
    mode.CHAR_PACE_S = 0.01
    mode.WORD_PAUSE_S = 0.01
    mode.LINE_PAUSE_S = 0.02
    return mode


class TestScannerMode:
    """Tests for ScannerMode."""

    def test_mode_properties(self, scanner_mode: ScannerMode) -> None:
        assert scanner_mode.mode_type == AppMode.SCANNER
        assert scanner_mode.display_name == "Scanner Mode"

    def test_enter_and_exit_lifecycle(self, scanner_mode: ScannerMode, mock_app: BrailceApp) -> None:
        scanner_mode.on_enter()
        assert scanner_mode._preview_running is True
        mock_app.display.show_mode.assert_called_with(AppMode.SCANNER)

        scanner_mode.on_exit()
        assert scanner_mode._preview_running is False
        mock_app.display.close_camera_preview.assert_called_once()

    def test_clear_pressed_resets_state(self, scanner_mode: ScannerMode, mock_app: BrailceApp) -> None:
        scanner_mode.scanned_text = "NEED"
        scanner_mode._last_result = {"readable": True, "text": "NEED", "cells": []}
        scanner_mode._braille_detected = True

        scanner_mode.handle_event(InputEvent(event_type=EventType.CLEAR_PRESSED))

        assert scanner_mode.scanned_text is None
        assert scanner_mode._last_result is None
        assert scanner_mode._braille_detected is None
        mock_app.display.show_scanner_viewfinder.assert_called_with(
            status=get_string("SCANNER_READY", "en")
        )

    def test_camera_missing_reports_once_without_fake_braille(
        self, scanner_mode: ScannerMode, mock_app: BrailceApp
    ) -> None:
        mock_app.camera.is_available.return_value = False
        scanner_mode._report_camera_missing()
        scanner_mode._report_camera_missing()

        expected = get_string("SCANNER_CAMERA_NOT_CONNECTED", "en")
        mock_app.display.show_error.assert_called_once_with(expected)
        mock_app.speech.speak_text.assert_called_once_with(expected, interrupt=True)

    def test_scan_error_reason_maps_to_bilingual_message(
        self, scanner_mode: ScannerMode, mock_app: BrailceApp
    ) -> None:
        mock_app.state.language = "fil"
        frame = np.full((240, 320, 3), 200, dtype=np.uint8)
        mock_app.camera.capture_sharpest.return_value = frame
        mock_app.obr.process_image.return_value = {
            "readable": False,
            "reason": "blurry",
            "text": "",
            "cells": [],
        }

        scanner_mode._scan_worker()

        expected = get_string("SCANNER_BLURRY", "fil")
        mock_app.display.show_status.assert_called_with(expected)
        mock_app.speech.speak_text.assert_called_with(expected, interrupt=True)

    def test_readout_speaks_letter_by_letter_then_whole_word(
        self, scanner_mode: ScannerMode, mock_app: BrailceApp
    ) -> None:
        result = {
            "readable": True,
            "text": "NEED",
            "cells": [
                {"char": "N", "kind": "char", "dots_str": "Dots 1-3-4-5", "word_index": 0, "line_index": 0},
                {"char": "E", "kind": "char", "dots_str": "Dots 1-5", "word_index": 0, "line_index": 0},
                {"char": "E", "kind": "char", "dots_str": "Dots 1-5", "word_index": 0, "line_index": 0},
                {"char": "D", "kind": "char", "dots_str": "Dots 1-4-5", "word_index": 0, "line_index": 0},
            ],
        }
        scanner_mode.scanned_text = "NEED"
        scanner_mode._readout_worker(result)

        char_calls = [call.args[0] for call in mock_app.speech.speak_character.call_args_list]
        assert char_calls == ["N", "E", "E", "D"]
        mock_app.speech.speak_word_clean.assert_called_once_with("NEED", interrupt=False)
        assert mock_app.display.show_braille.call_count == 4

    def test_readout_unknown_cell_and_any_key_cancel(
        self, scanner_mode: ScannerMode, mock_app: BrailceApp
    ) -> None:
        scanner_mode.CHAR_PACE_S = 0.25
        result = {
            "readable": True,
            "text": "A?",
            "cells": [
                {"char": "?", "kind": "unknown", "dots_str": "Dots 1-2-3-4-5-6", "word_index": 0, "line_index": 0},
                {"char": "A", "kind": "char", "dots_str": "Dots 1", "word_index": 0, "line_index": 0},
            ],
        }
        scanner_mode._start_readout_thread(result)
        time.sleep(0.05)

        # Any key during readout stops cleanly and clears audio queue
        scanner_mode.handle_event(InputEvent(event_type=EventType.SPACE_PRESSED))
        time.sleep(0.05)

        assert scanner_mode._is_reading is False
        mock_app.speech.stop.assert_called()
        mock_app.speech.speak_text.assert_any_call(get_string("SCANNER_UNKNOWN_CELL", "en"), interrupt=False)
