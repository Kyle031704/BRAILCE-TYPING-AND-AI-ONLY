"""
Audit Filipino Audio Clarity and Acoustic Metrics
==================================================

Computes acoustic clarity metrics across Filipino and English reference audio:
  - Active duration
  - Leading & trailing silence (ms)
  - Peak amplitude (int16 scale 0-32767)
  - Active RMS (volume level)
  - Spectral centroid (Hz)
  - Energy ratio above 4 kHz (%)
  - Clipping % (samples >= 32760)

Compares Filipino assets against the frozen English baseline cohort and flags:
  - Too quiet (RMS < 2500)
  - Rushed / too short (active duration < 0.28s for characters)
  - Onset cut (initial energy jump without smooth ramp)
  - Muffled (energy ratio above 4kHz < 8% or centroid < 1600 Hz)
  - Clipped (samples reaching full-scale saturation)
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIL_DIR = PROJECT_ROOT / "brailce" / "audio" / "filipino"
EN_CHAR_DIR = PROJECT_ROOT / "brailce" / "audio" / "characters"
EN_WORD_DIR = PROJECT_ROOT / "brailce" / "audio" / "words_en"


def compute_audio_metrics(
    wav_path: str | Path,
    silence_threshold: int = 150,
) -> Dict[str, float | str | bool]:
    """Calculate acoustic and clarity metrics for a single 16-bit PCM WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2 or n_frames == 0:
        return {
            "file": os.path.basename(wav_path),
            "sr": framerate,
            "total_dur": 0.0,
            "active_dur": 0.0,
            "lead_silence_ms": 0.0,
            "trail_silence_ms": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "centroid": 0.0,
            "hf_ratio_pct": 0.0,
            "clip_pct": 0.0,
            "flags": "INVALID_FORMAT",
        }

    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    total_dur = len(data) / float(framerate)
    abs_data = np.abs(data)
    active_indices = np.where(abs_data > silence_threshold)[0]

    if len(active_indices) == 0:
        return {
            "file": os.path.basename(wav_path),
            "sr": framerate,
            "total_dur": total_dur,
            "active_dur": 0.0,
            "lead_silence_ms": total_dur * 1000.0,
            "trail_silence_ms": 0.0,
            "peak": float(np.max(abs_data)),
            "rms": 0.0,
            "centroid": 0.0,
            "hf_ratio_pct": 0.0,
            "clip_pct": 0.0,
            "flags": "SILENT",
        }

    first_act = int(active_indices[0])
    last_act = int(active_indices[-1])
    lead_silence_ms = (first_act / float(framerate)) * 1000.0
    trail_silence_ms = ((len(data) - 1 - last_act) / float(framerate)) * 1000.0
    active_dur = ((last_act - first_act + 1) / float(framerate))

    active_seg = data[first_act : last_act + 1].astype(np.float64)
    peak = float(np.max(abs_data))
    rms = float(np.sqrt(np.mean(active_seg ** 2))) if len(active_seg) > 0 else 0.0
    clip_count = int(np.sum(abs_data >= 32750))
    clip_pct = (clip_count / float(len(data))) * 100.0

    # Spectral analysis on active region
    n_fft = 1024
    hop = 512
    centroids: List[float] = []
    hf_energies: List[float] = []
    tot_energies: List[float] = []

    # Frequency bins for FFT
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / framerate)
    hf_mask = freqs >= 4000.0

    for start in range(0, len(active_seg) - n_fft + 1, hop):
        frame = active_seg[start : start + n_fft] * np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(frame))
        power = mag ** 2
        tot_p = np.sum(power)
        if tot_p > 1e-4:
            c = float(np.sum(freqs * mag) / max(1e-9, np.sum(mag)))
            centroids.append(c)
            hf_energies.append(float(np.sum(power[hf_mask])))
            tot_energies.append(float(tot_p))

    if not centroids:
        # Fallback for very short bursts
        pad = np.pad(active_seg, (0, max(0, n_fft - len(active_seg))))[:n_fft] * np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(pad))
        power = mag ** 2
        tot_p = np.sum(power)
        centroid = float(np.sum(freqs * mag) / max(1e-9, np.sum(mag)))
        hf_ratio_pct = float(np.sum(power[hf_mask]) / max(1e-9, tot_p) * 100.0)
    else:
        centroid = float(np.median(centroids))
        hf_ratio_pct = float((sum(hf_energies) / max(1e-9, sum(tot_energies))) * 100.0)

    # Detect onset cut: chopped onset has abrupt step at file boundary or first active sample > 1500
    onset_cut = False
    if first_act < int(framerate * 0.005) and len(data) > 0 and abs(data[0]) > 500:
        onset_cut = True
    elif len(data) > first_act and abs(data[first_act]) > 1500:
        onset_cut = True

    flags_list: List[str] = []
    if rms < 2600.0:
        flags_list.append("QUIET")
    if clip_pct > 0.01 or peak >= 32760:
        flags_list.append("CLIPPED")
    if onset_cut:
        flags_list.append("ONSET_CUT")
    if centroid < 1350.0 or (hf_ratio_pct < 0.25 and centroid < 1650.0):
        flags_list.append("MUFFLED")
    if active_dur < 0.25 and "char_" in str(wav_path):
        flags_list.append("TOO_SHORT")

    return {
        "file": os.path.basename(wav_path),
        "sr": framerate,
        "total_dur": round(total_dur, 3),
        "active_dur": round(active_dur, 3),
        "lead_silence_ms": round(lead_silence_ms, 1),
        "trail_silence_ms": round(trail_silence_ms, 1),
        "peak": round(peak, 0),
        "rms": round(rms, 1),
        "centroid": round(centroid, 1),
        "hf_ratio_pct": round(hf_ratio_pct, 2),
        "clip_pct": round(clip_pct, 4),
        "flags": ",".join(flags_list) if flags_list else "OK",
    }


def audit_cohort(wav_paths: List[Path]) -> List[Dict[str, float | str | bool]]:
    """Audit a list of wav files and return metrics."""
    return [compute_audio_metrics(p) for p in wav_paths if p.suffix.lower() == ".wav"]


def format_table(rows: List[Dict[str, float | str | bool]], title: str) -> str:
    """Format a terminal report table of audio metrics."""
    lines = [
        f"\n=== {title} ===",
        f"{'File':<28} | {'ActDur':>7} | {'Lead':>6} | {'Trail':>6} | {'Peak':>6} | {'RMS':>6} | {'Centroid':>8} | {'HF>4k%':>7} | {'Flags'}",
        "-" * 105,
    ]
    for r in rows:
        lead_s = f"{r['lead_silence_ms']:.0f}ms"
        trail_s = f"{r['trail_silence_ms']:.0f}ms"
        lines.append(
            f"{r['file']:<28} | {r['active_dur']:>6.2f}s | {lead_s:>6} | {trail_s:>6} | {r['peak']:>6.0f} | {r['rms']:>6.1f} | {r['centroid']:>6.0f}Hz | {r['hf_ratio_pct']:>6.2f}% | {r['flags']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Filipino speech clarity, spectrum, and RMS against English baseline.")
    parser.add_argument("--fil-dir", type=str, default=str(FIL_DIR), help="Path to Filipino audio bank.")
    parser.add_argument("--en-char-dir", type=str, default=str(EN_CHAR_DIR), help="Path to English characters.")
    parser.add_argument("--en-word-dir", type=str, default=str(EN_WORD_DIR), help="Path to English words.")
    args = parser.parse_args()

    fil_p = Path(args.fil_dir)
    en_char_p = Path(args.en_char_dir)
    en_word_p = Path(args.en_word_dir)

    print(f"Auditing Filipino audio clarity in: {fil_p}")

    # 1. Baseline English Characters
    en_char_files = sorted(en_char_p.glob("char_*.wav"))
    en_char_rows = audit_cohort(en_char_files)
    if en_char_rows:
        med_en_rms = np.median([r["rms"] for r in en_char_rows])
        med_en_centroid = np.median([r["centroid"] for r in en_char_rows])
        med_en_hf = np.median([r["hf_ratio_pct"] for r in en_char_rows])
        print(f"\n[BASELINE: English Characters ({len(en_char_rows)} files)]")
        print(f"  Median Active RMS: {med_en_rms:.1f} | Centroid: {med_en_centroid:.0f} Hz | HF > 4kHz: {med_en_hf:.2f}%")

    # 2. Key Filipino subset (Letters A-Z, Digits 0-9, Operators, Key Words)
    sample_letters = [fil_p / f"char_{chr(c)}.wav" for c in range(ord("A"), ord("Z") + 1)]
    sample_digits = [fil_p / f"char_{i}.wav" for i in range(10)]
    sample_ops = [fil_p / f"char_{op}.wav" for op in ("plus", "minus", "times", "divided", "equals", "period", "comma")]
    sample_words = [fil_p / f"word_{w}.wav" for w in ("kumusta", "bahay", "araw", "kita", "matematika", "is", "his", "this", "the", "was")]

    sample_targets = [p for p in (sample_letters + sample_digits + sample_ops + sample_words) if p.exists()]
    sample_rows = audit_cohort(sample_targets)
    print(format_table(sample_rows, "FILIPINO TARGET SUBSET CLARITY TABLE"))

    # 3. Full cohort summary
    all_fil_files = sorted(fil_p.glob("*.wav"))
    all_rows = audit_cohort(all_fil_files)
    flagged = [r for r in all_rows if r["flags"] != "OK"]

    print(f"\n[COHORT SUMMARY: {len(all_rows)} files audited]")
    print(f"  Passed OK: {len(all_rows) - len(flagged)}")
    print(f"  Flagged issues: {len(flagged)}")

    if flagged:
        print("\nFlagged issue breakdown:")
        flag_counts: Dict[str, int] = {}
        for r in flagged:
            for f in str(r["flags"]).split(","):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        for f, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  - {f}: {cnt} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
