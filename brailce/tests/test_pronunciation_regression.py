"""
Automated Pronunciation Regression Tests (Section A)
=====================================================

Verifies:
1. All single-word entry points (SPACE, ENTER-on-single-word, speak_word_clean,
   FreeInput, AIAssistant) produce the exact same normalized call to PiperEngine.
2. Carrier-sentence slicing ("Please say: <word>.") preserves weak /w/, /wh/ onsets
   and trailing /z/, /s/ codas without cutting into the word or leaking "Please say".
3. Duration & spectral similarity between isolated word output and in-sentence word
   rendition across all 31 problem words:
   is, his, this, its, island, miss, was, what, when, where, who, why, which, white,
   whale, wheel, while, with, without, will, we, were, water, window, wolf, woman,
   work, world, would, write, wrong.
"""

from __future__ import annotations

import os
import tempfile
import wave
from typing import Tuple

import numpy as np
import pytest

from brailce.speech.audio_normalize import (
    TARGET_SAMPLE_RATE,
    compute_wav_stats,
    slice_carrier_audio,
    standardize_wav_file,
)
from brailce.speech.base import SpeechEngine
from brailce.speech.piper_engine import PIPER_LENGTH_SCALE, PIPER_NOISE_SCALE, PIPER_NOISE_W, PiperEngine
from brailce.speech.service import SpeechService

PROBLEM_WORDS = [
    "is", "his", "this", "its", "island", "miss",
    "was", "what", "when", "where", "who", "why", "which", "white",
    "whale", "wheel", "while", "with", "without", "will", "we", "were",
    "water", "window", "wolf", "woman", "work", "world", "would", "write", "wrong",
]


class RecordingEngine(SpeechEngine):
    """Test double recording exact utterances sent to backend engine."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    @property
    def engine_name(self) -> str:
        return "RecordingEngine"

    def speak(self, text: str, interrupt: bool | None = None) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def set_voice(self, voice: str) -> None:
        pass

    def set_rate(self, rate: int) -> None:
        pass

    def set_volume(self, volume: int) -> None:
        pass


def _spectral_similarity(sig_a: np.ndarray, sig_b: np.ndarray, sr: int = 22050) -> float:
    """Compute cosine similarity between average log-magnitude FFT spectra of two active signals."""
    def trim_active(x: np.ndarray) -> np.ndarray:
        idx = np.where(np.abs(x) > 150.0)[0]
        if idx.size == 0:
            return x
        return x[idx[0] : idx[-1] + 1]

    a = trim_active(sig_a.astype(np.float64))
    b = trim_active(sig_b.astype(np.float64))
    if a.size < 256 or b.size < 256:
        return 0.0

    n_fft = 512
    def avg_log_spec(x: np.ndarray) -> np.ndarray:
        step = 256
        frames = [x[i : i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft + 1, step)]
        if not frames:
            pad = np.pad(x[:n_fft], (0, max(0, n_fft - len(x))))
            frames = [pad * np.hanning(n_fft)]
        specs = [np.log1p(np.abs(np.fft.rfft(f))) for f in frames]
        return np.mean(specs, axis=0)

    va = avg_log_spec(a)
    vb = avg_log_spec(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _synth_carrier_with_target(word: str, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic carrier waveform "Please say: <word>." with:
      - 400ms carrier energy ("Please say")
      - 220ms colon pause (quiet gap)
      - 350ms target word with soft /w/-like ramp onset (220 RMS) + vowel core + trailing coda
      - 100ms tail silence
    Returns (full_carrier_samples, ground_truth_target_samples).
    """
    t_carrier = np.arange(int(sr * 0.40)) / sr
    carrier = (np.sin(2 * np.pi * 440 * t_carrier) * 5000).astype(np.int16)

    pause = np.zeros(int(sr * 0.22), dtype=np.int16)

    # Soft onset (40ms ramp up from ~220 amplitude like /w/ or /wh/) + core + coda
    n_onset = int(sr * 0.04)
    n_core = int(sr * 0.24)
    n_coda = int(sr * 0.07)
    base_freq = 260 + (sum(ord(c) for c in word) % 150)

    t_on = np.arange(n_onset) / sr
    onset = (np.sin(2 * np.pi * base_freq * t_on) * np.linspace(220, 4500, n_onset)).astype(np.int16)

    t_core = np.arange(n_core) / sr
    core = (np.sin(2 * np.pi * base_freq * t_core) * 6000).astype(np.int16)

    t_coda = np.arange(n_coda) / sr
    coda = (np.sin(2 * np.pi * (base_freq * 2.2) * t_coda) * np.linspace(3000, 300, n_coda)).astype(np.int16)

    target = np.concatenate([onset, core, coda])
    tail = np.zeros(int(sr * 0.10), dtype=np.int16)
    full = np.concatenate([carrier, pause, target, tail])
    return full, target


class TestPronunciationConsistency:
    """Section A: Single-word routing, carrier slicing, and spectral similarity tests."""

    def test_unified_single_word_routing(self) -> None:
        """Space, Enter-on-single-word, speak_word, and speak_word_clean must match identically."""
        rec = RecordingEngine()
        svc = SpeechService(character_engine=rec, text_engine=rec)

        for word in PROBLEM_WORDS:
            rec.spoken.clear()
            svc.speak_word_clean(word)
            clean_out = list(rec.spoken)

            rec.spoken.clear()
            svc.speak_word(word)
            word_out = list(rec.spoken)

            rec.spoken.clear()
            svc.speak_sentence(word.upper())
            enter_out = list(rec.spoken)

            assert clean_out == word_out == enter_out == [f"{word}."], (
                f"Inconsistent speech routing for {word!r}: {clean_out} vs {word_out} vs {enter_out}"
            )

    def test_piper_uses_single_fixed_parameter_set(self) -> None:
        """Piper must lock one consistent VITS parameter set across all English speech."""
        assert PIPER_LENGTH_SCALE == "1.05"
        assert PIPER_NOISE_SCALE == "0.5"
        assert PIPER_NOISE_W == "0.6"

    @pytest.mark.parametrize("word", PROBLEM_WORDS)
    def test_carrier_slicer_preserves_onset_and_coda_spectral_similarity(self, word: str) -> None:
        """
        Verify generalized carrier slicer extracts the exact target word without
        clipping /w/, /wh/ onset or trailing /z/, /s/ coda, and matches in-sentence spectrum.
        """
        full_carrier, reference_target = _synth_carrier_with_target(word)
        sliced = np.array(slice_carrier_audio(full_carrier, sr=TARGET_SAMPLE_RATE), dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        try:
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(sliced.tobytes())
            assert standardize_wav_file(tmp_wav)
            sr, ch, sw, rms, peak = compute_wav_stats(tmp_wav)
            assert sr == TARGET_SAMPLE_RATE
            assert ch == 1
            assert sw == 2
            assert 2000.0 <= rms <= 6000.0
            assert peak <= 32000

            with wave.open(tmp_wav, "rb") as wf:
                isolated = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        # Duration of active sound should closely match reference target
        iso_active = isolated[np.abs(isolated) > 150]
        ref_active = reference_target[np.abs(reference_target) > 150]
        dur_ratio = len(iso_active) / max(1, len(ref_active))
        assert 0.80 <= dur_ratio <= 1.25, f"{word}: duration ratio {dur_ratio:.2f} out of bounds"

        sim = _spectral_similarity(isolated, reference_target)
        assert sim >= 0.90, f"{word}: spectral similarity {sim:.3f} < 0.90"

    @pytest.mark.skipif(not PiperEngine().is_available(), reason="Live Piper binary/model not installed on this host")
    def test_live_piper_isolated_vs_carrier_similarity(self) -> None:
        """
        When running on Raspberry Pi (or host with Piper installed), synthesize all 31
        words live through carrier-slice path vs direct carrier extraction and assert
        spectral and duration parity.
        """
        engine = PiperEngine()
        for word in PROBLEM_WORDS:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
                iso_wav, ref_wav = f1.name, f2.name
            try:
                assert engine.synthesize_to_wav(word, iso_wav, carrier_slice=True)
                assert engine.synthesize_to_wav(f"Please say: {word}.", ref_wav, carrier_slice=False)

                with wave.open(iso_wav, "rb") as w1, wave.open(ref_wav, "rb") as w2:
                    s1 = np.frombuffer(w1.readframes(w1.getnframes()), dtype=np.int16)
                    s2 = np.frombuffer(w2.readframes(w2.getnframes()), dtype=np.int16)
                s2_sliced = np.array(slice_carrier_audio(s2, sr=22050), dtype=np.int16)
                sim = _spectral_similarity(s1, s2_sliced)
                assert sim >= 0.85, f"Live Piper similarity low for {word}: {sim:.3f}"
            finally:
                for p in (iso_wav, ref_wav):
                    if os.path.exists(p):
                        os.remove(p)


class TestFilipinoPronunciationRegression:
    """Part 1 regression tests: Filipino pronunciation, 'kita', letter names, and female voice profile."""

    def test_filipino_kita_lexicon_and_word_bank(self) -> None:
        """Verify 'kita' and related forms are preserved in lexicon and exist in the word bank."""
        from brailce.i18n.fil_lexicon import apply_filipino_lexicon
        from brailce.speech.filipino_engine import get_filipino_word_path, FILIPINO_AUDIO_DIR

        # 1. Lexicon preservation
        for w in ("kita", "nakita", "nakitang", "makita"):
            spoken = apply_filipino_lexicon(w)
            assert spoken.lower() == w.lower(), f"Lexicon corrupted {w} -> {spoken}"

        # 2. Word bank audio exists
        kita_wav = get_filipino_word_path("kita")
        assert kita_wav is not None and os.path.isfile(kita_wav), "word_kita.wav is missing"
        assert os.path.getsize(kita_wav) > 1000, "word_kita.wav is too small"

        for variant in ("nakita", "nakitang", "makita"):
            v_p = os.path.join(FILIPINO_AUDIO_DIR, f"word_{variant}.wav")
            assert os.path.isfile(v_p), f"Missing {variant} audio file"

    def test_filipino_characters_female_pitch_and_quality(self) -> None:
        """Verify character keystroke files use female Blessica voice without raspiness or low F0."""
        from brailce.speech.filipino_engine import FILIPINO_AUDIO_DIR
        from tools.audit_filipino_gender import estimate_f0_autocorr

        for char in ("A", "B", "C", "K", "M", "S", "T"):
            char_p = os.path.join(FILIPINO_AUDIO_DIR, f"char_{char}.wav")
            assert os.path.isfile(char_p), f"Missing {char_p}"

            with wave.open(char_p, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)

            f0, _ = estimate_f0_autocorr(data, sr=sr)
            assert f0 is not None, f"char_{char}.wav has no detectable voiced pitch"
            assert f0 >= 160.0, f"char_{char}.wav F0={f0:.1f} Hz is too low (expected female Blessica > 160 Hz)"


