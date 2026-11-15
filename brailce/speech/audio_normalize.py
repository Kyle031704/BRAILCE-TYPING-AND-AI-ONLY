"""
Brailce Unified Audio Post-Processing & Carrier-Slicer Pipeline
================================================================

Single source of truth for WAV standardization across English (Piper) and
Filipino (sherpa-onnx VITS) audio synthesis and offline cache building:
- Converts PCM WAVs to 22050 Hz, mono (1 channel), 16-bit signed PCM (no ffmpeg needed).
- Normalizes active speech RMS loudness to a consistent target (-18 dBFS / RMS ~4200)
  with hard-clip protection (peak <= 30000).
- Applies uniform pre/post digital zero silence padding (120 ms pre, 180 ms post).
- Provides generalized carrier-sentence audio slicing ("Please say: <target>.")
  for isolated English words and characters.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import wave
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1
TARGET_SAMPWIDTH = 2
TARGET_ACTIVE_RMS = 4200.0
MAX_PEAK_INT16 = 30000.0
DEFAULT_PRE_SILENCE_MS = 120
DEFAULT_POST_SILENCE_MS = 180


def trim_silence_pcm(
    samples: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    threshold: float = 80.0,
    keep_margin_ms: int = 40,
) -> np.ndarray:
    """Trim leading and trailing near-silence while safely preserving soft onsets and codas."""
    if samples.size == 0:
        return samples
    abs_s = np.abs(samples)
    above = np.where(abs_s > threshold)[0]
    if above.size == 0:
        return samples
    margin = int(sr * (keep_margin_ms / 1000.0))
    start = max(0, int(above[0]) - margin)
    end = min(samples.size, int(above[-1]) + margin + 1)
    trimmed = samples[start:end].copy()

    # Apply gentle 5ms cosine envelope to prevent pops without cutting onsets
    fade_len = min(int(sr * 0.005), trimmed.size // 4)
    if fade_len > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    return trimmed


def resample_mono_pcm(samples: np.ndarray, src_sr: int, dst_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample 1-D float PCM array cleanly using linear interpolation (zero external dependencies)."""
    if src_sr == dst_sr or samples.size <= 1:
        return samples
    dst_len = max(1, int(round(samples.size * (dst_sr / float(src_sr)))))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
    return np.interp(x_new, x_old, samples)


def normalize_loudness_pcm(
    samples: np.ndarray,
    target_rms: float = TARGET_ACTIVE_RMS,
    max_peak: float = MAX_PEAK_INT16,
) -> np.ndarray:
    """
    Normalize active speech frames (above noise floor) to target_rms while
    preventing int16 clipping via peak ceiling max_peak.
    """
    if samples.size == 0:
        return samples
    abs_s = np.abs(samples)
    peak = float(np.max(abs_s))
    if peak < 10.0:
        return samples

    # Compute RMS on active frames (> 3% of peak) so silence gaps don't distort gain
    active = samples[abs_s > max(40.0, peak * 0.03)]
    if active.size == 0:
        active = samples
    active_rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
    if active_rms < 1.0:
        return samples

    gain = target_rms / active_rms
    # Clamp gain so peak never exceeds max_peak
    if peak * gain > max_peak:
        gain = max_peak / peak
    # Prevent insane amplification on near-silent tracks
    gain = min(gain, 12.0)

    scaled = samples * gain
    return np.clip(scaled, -32767.0, 32767.0)


def slice_carrier_audio(samples: Sequence[int] | np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> List[int]:
    """
    Extract the target word/character from a carrier sentence such as
    "Please say: <word>." by locating the quiet pause after "Please say:"
    and preserving the entire target onset (including soft /w/, /h/, /th/, /s/)
    and coda through the end of the word.
    """
    arr = np.asarray(samples, dtype=np.float64)
    win = max(1, int(sr * 0.010))  # 10 ms windows
    num_windows = arr.size // win
    if num_windows < 35:
        return [int(x) for x in arr]

    reshaped = arr[: num_windows * win].reshape(num_windows, win)
    rms_list = np.sqrt(np.mean(reshaped ** 2, axis=1))

    first_voiced_candidates = np.where(rms_list > 250.0)[0]
    first_voiced = int(first_voiced_candidates[0]) if first_voiced_candidates.size > 0 else 0

    last_voiced_candidates = np.where(rms_list > 200.0)[0]
    last_voiced = int(last_voiced_candidates[-1]) if last_voiced_candidates.size > 0 else num_windows - 1

    # Skip "Please" onset (at least 250ms), search before final trailing word decay
    search_start = min(num_windows - 12, first_voiced + 25)
    search_end = max(search_start + 1, last_voiced - 6)

    # Locate the lowest-RMS quiet window in the colon pause between "Please say:" and "<word>"
    sub_rms = rms_list[search_start:search_end]
    if sub_rms.size == 0:
        return [int(x) for x in arr]
    min_win = search_start + int(np.argmin(sub_rms))

    # Find target onset just after min_win (low threshold 160 preserves soft /w/, /h/, /f/, /s/)
    target_onset = min_win
    for j in range(min_win, min(num_windows, min_win + 35)):
        if rms_list[j] > 160.0:
            target_onset = j
            break

    # Include 60ms before target onset to keep full /w/ and /wh/ aspiration/glide
    cut_window = max(first_voiced + 20, target_onset - 6)
    end_window = min(num_windows, last_voiced + 6)
    if end_window <= cut_window:
        return [int(x) for x in arr]

    sliced = arr[cut_window * win : end_window * win]
    if sliced.size < int(sr * 0.030) or float(np.max(np.abs(sliced))) < 300.0:
        return [int(x) for x in arr]

    return [int(round(x)) for x in sliced]


def standardize_wav_file(
    wav_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_rms: float = TARGET_ACTIVE_RMS,
    pre_ms: int = DEFAULT_PRE_SILENCE_MS,
    post_ms: int = DEFAULT_POST_SILENCE_MS,
    trim_silence: bool = True,
) -> bool:
    """
    Read a WAV file in-place, convert to 22050 Hz mono s16, trim excess silence,
    normalize loudness to target_rms, and apply uniform pre/post silence padding.
    """
    if not os.path.isfile(wav_path):
        return False
    try:
        with wave.open(wav_path, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if not raw or sampwidth != 2:
            return False

        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if nchannels > 1:
            data = data.reshape(-1, nchannels).mean(axis=1)

        if src_sr != target_sr:
            data = resample_mono_pcm(data, src_sr, target_sr)

        if trim_silence:
            data = trim_silence_pcm(data, sr=target_sr)

        data = normalize_loudness_pcm(data, target_rms=target_rms)

        pre_samples = int(target_sr * (pre_ms / 1000.0))
        post_samples = int(target_sr * (post_ms / 1000.0))
        pcm_int16 = np.clip(np.round(data), -32767, 32767).astype(np.int16)
        padded = np.concatenate([
            np.zeros(pre_samples, dtype=np.int16),
            pcm_int16,
            np.zeros(post_samples, dtype=np.int16),
        ])

        # Ensure full-file RMS stays within +-2.5 dB of target bank level without clipping
        total_rms = float(np.sqrt(np.mean(padded.astype(np.float64) ** 2))) if padded.size else 0.0
        min_total_rms = target_rms * 0.58  # ~2436 when target_rms=4200
        if 10.0 < total_rms < min_total_rms:
            boost = min_total_rms / total_rms
            peak = float(np.max(np.abs(padded)))
            if peak * boost > MAX_PEAK_INT16:
                boost = MAX_PEAK_INT16 / peak
            padded = np.clip(np.round(padded.astype(np.float64) * boost), -32767, 32767).astype(np.int16)

        with wave.open(wav_path, "wb") as wf_out:
            wf_out.setnchannels(TARGET_CHANNELS)
            wf_out.setsampwidth(TARGET_SAMPWIDTH)
            wf_out.setframerate(target_sr)
            wf_out.writeframes(padded.tobytes())

        return True
    except Exception as exc:
        logger.warning("standardize_wav_file failed for %s: %s", wav_path, exc)
        return False


def compute_wav_stats(wav_path: str) -> Tuple[int, int, int, float, int]:
    """
    Return (sample_rate, channels, sampwidth, rms, peak) for a WAV file.
    """
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if not raw or sw != 2:
        return sr, ch, sw, 0.0, 0
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
    peak = int(np.max(np.abs(arr))) if arr.size else 0
    return sr, ch, sw, rms, peak
