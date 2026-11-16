"""
Test Filipino TTS v2 Audio Assets & Integration
=================================================
Verifies:
(a) Every key in Filipino keystroke map has a valid WAV that passes duration/loudness checks
(b) word_kumusta.wav speech duration >= 0.45s (no dropped syllables)
(c) Typing K, U, M, U, S, T, A + Space in Free Input enqueues the 7 character clips and then word_kumusta.wav
(d) Mode intro WAVs exist for all four modes
(e) English character audio assets remain 100% bit-exact and unchanged
"""

import hashlib
import os
import wave
from unittest.mock import MagicMock, call
import numpy as np
import pytest

from brailce.core.events import EventType, InputEvent, character_event
from brailce.i18n.fil_keys import (
    FILIPINO_KEY_LETTERS_ENGLISH,
    FILIPINO_KEY_DIGITS,
    FILIPINO_KEY_OPERATORS,
    FILIPINO_KEY_PUNCTUATION,
    get_filipino_keystroke_filename,
)
from brailce.modes.free_input import FreeInputMode
from brailce.speech.filipino_engine import (
    FILIPINO_AUDIO_DIR as FIL_DIR,
    get_filipino_character_path,
    get_filipino_mode_path,
    get_filipino_word_path,
)

FROZEN_ENGLISH_CHARACTERS_SHA256 = "a3c783ddf53f47c20a025480064080bfd51a24ff6d23b46b273e3da32901d12e"


def test_english_audio_assets_unchanged():
    """Requirement 1: English audio assets must remain 100% bit-exact."""
    en_dir = os.path.join(os.path.dirname(__file__), "..", "audio", "characters")
    hashes = {}
    for root, _, files in os.walk(en_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, en_dir)
            with open(p, "rb") as fp:
                hashes[rel] = hashlib.sha256(fp.read()).hexdigest()
    
    assert len(hashes) == 46
    manifest_hash = hashlib.sha256(str(sorted(hashes.items())).encode("utf-8")).hexdigest()
    assert manifest_hash == FROZEN_ENGLISH_CHARACTERS_SHA256


def test_filipino_keystroke_wavs_exist_and_pass_duration_loudness():
    """(a) Every keystroke in Filipino map has a valid WAV within duration/loudness range."""
    all_keys = {
        **FILIPINO_KEY_LETTERS_ENGLISH,
        **FILIPINO_KEY_DIGITS,
        **FILIPINO_KEY_OPERATORS,
        **FILIPINO_KEY_PUNCTUATION,
    }
    for ch in all_keys.keys():
        fname = get_filipino_keystroke_filename(ch)
        path = os.path.join(FIL_DIR, fname)
        assert os.path.isfile(path), f"Missing keystroke WAV for {ch!r}: {fname}"
        
        with wave.open(path, "rb") as wf:
            nframes = wf.getnframes()
            sr = wf.getframerate()
            frames = wf.readframes(nframes)
            
        dur = nframes / float(sr)
        # Keystroke duration must be between 0.25s and 1.65s (including standard silence padding)
        assert 0.25 <= dur <= 1.65, f"Keystroke {ch!r} duration out of bounds: {dur:.2f}s"
        
        samples = np.frombuffer(frames, dtype=np.int16)
        max_amp = np.max(np.abs(samples))
        assert max_amp < 32765, f"Keystroke {ch!r} is clipped: max={max_amp}"
        
        voiced = np.where(np.abs(samples) > 200)[0]
        assert voiced.size > 0, f"Keystroke {ch!r} contains no speech audio"
        voiced_samples = samples[voiced[0] : voiced[-1] + 1]
        rms = float(np.sqrt(np.mean(voiced_samples.astype(np.float64) ** 2)))
        assert 1500 <= rms <= 6000, f"Keystroke {ch!r} RMS out of bounds: {rms:.1f}"


def test_word_kumusta_duration_and_speech():
    """(b) word_kumusta.wav duration and active speech duration >= 0.45s."""
    path = os.path.join(FIL_DIR, "word_kumusta.wav")
    assert os.path.isfile(path), "word_kumusta.wav is missing"
    
    with wave.open(path, "rb") as wf:
        nframes = wf.getnframes()
        sr = wf.getframerate()
        frames = wf.readframes(nframes)
        
    total_dur = nframes / float(sr)
    samples = np.frombuffer(frames, dtype=np.int16)
    voiced = np.where(np.abs(samples) > 200)[0]
    active_dur = (voiced[-1] - voiced[0]) / float(sr)
    
    assert total_dur >= 0.80, f"word_kumusta.wav total duration too short: {total_dur:.2f}s"
    assert active_dur >= 0.45, f"word_kumusta.wav active speech too short ({active_dur:.2f}s < 0.45s)"


def test_mode_intro_wavs_exist_for_all_four_modes():
    """(d) Mode intro WAVs exist for all four modes."""
    for mode_name in ["free_input", "math", "ai", "scanner"]:
        fname = f"mode_{mode_name}.wav"
        path = os.path.join(FIL_DIR, fname)
        assert os.path.isfile(path), f"Mode intro {fname} is missing"
        with wave.open(path, "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
        assert dur >= 2.0, f"Mode intro {fname} too short: {dur:.2f}s"


def test_typing_kumusta_in_free_input_enqueues_characters_then_word():
    """(c) Typing K,U,M,U,S,T,A + Space in Free Input enqueues 7 characters and then word_kumusta.wav."""
    from brailce.braille.mapping import BrailleMappingService
    from brailce.braille.encoder import BrailleEncoder
    from brailce.core.state import ApplicationState
    
    app = MagicMock()
    app.state = ApplicationState()
    app.state.language = "fil"
    app.config.get.return_value = True
    
    from brailce.config.loader import ConfigLoader
    mapping_svc = BrailleMappingService()
    mapping_svc.load_from_config(ConfigLoader.from_defaults())
    app.encoder = BrailleEncoder(mapping_svc)
    
    # Mock speech service
    app.speech = MagicMock()
    app.speech.language = "fil"
    
    mode = FreeInputMode(app)
    
    # Type K, U, M, U, S, T, A
    for ch in "KUMUSTA":
        event = character_event(ch, "keyboard")
        mode.handle_event(event)
        
    assert app.state.text_buffer == "KUMUSTA"
    assert app.speech.speak_character.call_count == 7
    expected_char_calls = [call("K"), call("U"), call("M"), call("U"), call("S"), call("T"), call("A")]
    assert app.speech.speak_character.call_args_list == expected_char_calls
    
    # Press Space
    event_space = InputEvent(event_type=EventType.SPACE_PRESSED)
    mode.handle_event(event_space)
    
    # Should speak completed word "KUMUSTA"
    app.speech.speak_word_clean.assert_called_once_with("KUMUSTA")
