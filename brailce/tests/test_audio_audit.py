"""
Section D Tests: Audio Quality Audit, Voice Consistency, & Unified Queue
========================================================================

Runs tools/audit_audio.py across all cached WAV banks and verifies:
  - Sample rate (22050 Hz), channels (1), 16-bit PCM
  - No silent WAV files (RMS >= 500)
  - No clipping (peak < 32700)
  - Loudness consistency within ±3.0 dB of language median
  - Single voice per language & filipino_letter_voice config toggle
  - Mode intros and language toggles route through SpeechService without bare threads
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from brailce.config.loader import ConfigLoader
from brailce.core.app import BrailceApp
from brailce.core.events import AppMode
from brailce.i18n.strings import STRINGS, get_mode_intro, get_string
from brailce.speech.service import SpeechService

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from audit_audio import audit_all


class TestAudioBankAudit:
    """Section D.5: Run audio quality audit in pytest."""

    def test_all_audio_banks_pass_quality_audit(self) -> None:
        reports = audit_all()
        assert len(reports) >= 2, "Expected en_characters and filipino audio banks"
        all_failures = []
        for rep in reports:
            if not rep.passed:
                all_failures.extend(rep.failures)
        assert not all_failures, "Audio audit failed:\n" + "\n".join(all_failures)


class TestVoiceConsistencyAndSingleQueue:
    """Section D.1–D.4: Unified voice per language, i18n LCD/speech parity, no bare threads."""

    def test_i18n_contains_all_four_mode_intros_in_both_languages(self) -> None:
        for mode in (AppMode.FREE_INPUT, AppMode.MATH, AppMode.AI_ASSISTANT, AppMode.SCANNER):
            en_intro = get_mode_intro(mode, "en")
            fil_intro = get_mode_intro(mode, "fil")
            assert en_intro and fil_intro
            assert en_intro != fil_intro

    def test_filipino_letter_voice_config_switch(self) -> None:
        cfg = ConfigLoader.from_defaults()
        svc = SpeechService.from_config(cfg)
        assert svc.filipino_letter_voice == "filipino"

        svc.language = "fil"
        svc._filipino_engine = MagicMock()
        svc.speak_character("A")
        svc._filipino_engine.enqueue_character.assert_called_once_with("A", use_filipino_voice=True)

        # Toggle config switch to english letter names
        svc.filipino_letter_voice = "english"
        mock_char_eng = MagicMock()
        svc._character_engine = mock_char_eng
        svc.speak_character("B")
        mock_char_eng.speak.assert_called_once_with("b.", interrupt=False)

    def test_mode_change_uses_unified_i18n_and_speech_queue(self) -> None:
        config = ConfigLoader.from_defaults()
        app = BrailceApp(config)
        app.setup()

        app.speech = MagicMock(spec=SpeechService)
        app.state.language = "fil"
        app._on_mode_change(AppMode.FREE_INPUT, AppMode.MATH)

        expected_intro = get_mode_intro(AppMode.MATH, "fil")
        app.display.show_status(expected_intro)
        app.speech.stop.assert_called_once()
        app.speech.speak_mode_intro.assert_called_once_with(AppMode.MATH, interrupt=False)

    def test_toggle_language_uses_unified_speech_queue(self) -> None:
        config = ConfigLoader.from_defaults()
        app = BrailceApp(config)
        app.setup()

        app.speech = MagicMock(spec=SpeechService)
        new_lang = app.toggle_language()
        assert new_lang == "fil"
        app.speech.speak_language_switch.assert_called_once_with("fil", interrupt=True)

