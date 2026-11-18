"""
Tests for Phase B: Filipino digits, math operators, and punctuation.
Verifies file existence, duration, RMS, and zero MMS fallback on playback.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from brailce.i18n.fil_keys import (
    FILIPINO_KEY_LETTERS_ENGLISH,
    FILIPINO_KEY_DIGITS,
    FILIPINO_KEY_OPERATORS,
    FILIPINO_KEY_PUNCTUATION,
    get_filipino_keystroke_filename,
)
from brailce.speech.filipino_engine import FilipinoEngine

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "filipino"))

@pytest.mark.parametrize("char", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789") + ["+", "-", "*", "/", "=", ".", ",", "?", "!"])
def test_filipino_keystroke_file_exists_and_bounds(char: str):
    fname = get_filipino_keystroke_filename(char)
    assert fname, f"Filename not found for char '{char}'"
    fpath = os.path.join(AUDIO_DIR, fname)
    assert os.path.isfile(fpath), f"Audio file does not exist: {fpath}"

    sig, sr = sf.read(fpath)
    dur = len(sig) / sr
    assert 0.20 <= dur <= 1.50, f"Duration out of bounds for '{char}': {dur:.3f}s"

    rms = np.sqrt(np.mean(sig ** 2)) * 32768.0
    assert 2000 <= rms <= 6500, f"RMS out of bounds for '{char}': {rms:.1f}"

def test_no_mms_fallback_on_keystrokes():
    engine = FilipinoEngine()
    engine.synthesize_speech_offline = MagicMock(return_value=None)
    engine._play_wav_direct = MagicMock()

    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=.,?!")
    for c in chars:
        engine.speak_character(c, use_filipino_voice=True, style="english_names")

    # MMS synthesize must never be called for any cached keystroke
    engine.synthesize_speech_offline.assert_not_called()
