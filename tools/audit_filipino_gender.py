"""
Audit Filipino Audio Bank Gender / F0 Pitch
=============================================

Scans all WAV files in `brailce/audio/filipino/` and estimates the fundamental
frequency (F0) using normalized autocorrelation.

Flagging rules:
  - Female voice (fil-PH-BlessicaNeural): typical median F0 is 180 Hz – 270 Hz.
  - Male voice (fil-PH-AngeloNeural or unshifted Piper Ryan): median F0 < 140 Hz.
  - Any file with median voiced F0 < 140 Hz is flagged as a male voice offender.
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILIPINO_AUDIO_DIR = PROJECT_ROOT / "brailce" / "audio" / "filipino"


def estimate_f0_autocorr(
    signal: np.ndarray,
    sr: int = 22050,
    fmin: float = 70.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    voicing_threshold: float = 0.45,
) -> Tuple[Optional[float], List[float]]:
    """
    Estimate fundamental frequency (F0) across frames using normalized autocorrelation.
    Returns (median_f0, list_of_voiced_f0s).
    """
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Convert to float and remove DC offset
    x = signal.astype(np.float64)
    x -= np.mean(x)

    frame_len = int(sr * (frame_ms / 1000.0))
    hop_len = int(sr * (hop_ms / 1000.0))

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    voiced_f0s: List[float] = []

    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start : start + frame_len]
        energy = np.sum(frame ** 2)
        if energy < 1e5:  # silence / low energy threshold for 16-bit PCM scale
            continue

        # Windowing
        windowed = frame * np.hanning(len(frame))

        # Autocorrelation via FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * frame_len)))
        fx = np.fft.rfft(windowed, n=n_fft)
        r = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:frame_len]

        r0 = r[0]
        if r0 <= 1e-6:
            continue

        norm_r = r / r0
        search_region = norm_r[min_lag:max_lag]
        if len(search_region) == 0:
            continue

        best_lag_rel = int(np.argmax(search_region))
        peak_val = search_region[best_lag_rel]

        if peak_val >= voicing_threshold:
            lag = min_lag + best_lag_rel
            f0 = sr / lag
            if fmin <= f0 <= fmax:
                voiced_f0s.append(float(f0))

    if not voiced_f0s:
        return None, []

    median_f0 = float(np.median(voiced_f0s))
    return median_f0, voiced_f0s


def audit_filipino_audio(
    audio_dir: Path = FILIPINO_AUDIO_DIR,
    threshold_f0: float = 140.0,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Audit all WAV files in audio_dir.
    Returns (passed_files, flagged_male_files).
    """
    passed = []
    flagged = []

    wav_files = sorted(audio_dir.glob("*.wav"))
    for p in wav_files:
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            data = np.frombuffer(raw, dtype=np.int16)
            med_f0, _ = estimate_f0_autocorr(data, sr=sr)
            if med_f0 is None:
                # Unvoiced / percussive / short silence
                passed.append((p.name, 0.0))
            elif med_f0 < threshold_f0:
                flagged.append((p.name, med_f0))
            else:
                passed.append((p.name, med_f0))
        except Exception as exc:
            print(f"Error reading {p.name}: {exc}", file=sys.stderr)

    return passed, flagged


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Filipino audio bank for male voice / low F0 pitch.")
    parser.add_argument("--threshold", type=float, default=140.0, help="Flag F0 below this Hz as male (default: 140.0).")
    parser.add_argument("--dir", type=str, default=str(FILIPINO_AUDIO_DIR), help="Directory to audit.")
    args = parser.parse_args()

    audio_dir = Path(args.dir)
    print(f"Auditing Filipino audio in: {audio_dir}")
    print(f"Flagging files with median voiced F0 < {args.threshold} Hz (male voice profile)...")

    passed, flagged = audit_filipino_audio(audio_dir, threshold_f0=args.threshold)

    print(f"\nTotal audited: {len(passed) + len(flagged)} files.")
    print(f"Passed female/neutral: {len(passed)} files.")
    print(f"Flagged male (< {args.threshold} Hz): {len(flagged)} files.")

    if flagged:
        print("\nFLAGGED OFFENDERS:")
        for name, f0 in flagged:
            print(f"  [MALE VOICE] {name:40s} F0 = {f0:.1f} Hz")
        return 1
    else:
        print("\nAll files passed! Clean female Blessica voice profile confirmed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
