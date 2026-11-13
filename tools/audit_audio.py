"""
Brailce Audio Bank Quality Auditor
===================================

Scans every cached WAV file in both language banks (English and Filipino) and
fails if any file violates quality invariants:
  1. Format mismatch: must be 22050 Hz, mono (1 channel), 16-bit signed PCM.
  2. Silence / broken synthesis: total RMS < 500.0 (e.g. silent word_oo.wav).
  3. Clipping: peak amplitude >= 32700 (int16 saturation).
  4. Loudness deviation: total RMS deviates more than ±3.0 dB from its
     language bank median RMS.

Runnable as CLI (`python tools/audit_audio.py`) or imported in pytest.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brailce.speech.audio_normalize import compute_wav_stats

TARGET_SR = 22050
TARGET_CH = 1
TARGET_SW = 2
MIN_RMS = 500.0
MAX_PEAK = 32699
MAX_DB_DEVIATION = 3.0


@dataclass
class AuditReport:
    bank_name: str
    file_count: int = 0
    median_rms: float = 0.0
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


def audit_directory(bank_name: str, directory: str) -> AuditReport:
    """Audit all WAV files inside a single directory."""
    report = AuditReport(bank_name=bank_name)
    if not os.path.isdir(directory):
        return report

    wav_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".wav"))
    report.file_count = len(wav_files)
    if not wav_files:
        return report

    stats: List[Tuple[str, int, int, int, float, int]] = []
    for fname in wav_files:
        full_path = os.path.join(directory, fname)
        try:
            sr, ch, sw, rms, peak = compute_wav_stats(full_path)
            stats.append((fname, sr, ch, sw, rms, peak))
        except Exception as exc:
            report.failures.append(f"[{bank_name}] {fname}: unreadable WAV ({exc})")

    valid_rms = [r for (_, sr, ch, sw, r, _) in stats if r >= MIN_RMS]
    median_rms = float(np.median(valid_rms)) if valid_rms else 0.0
    report.median_rms = median_rms

    for fname, sr, ch, sw, rms, peak in stats:
        if sr != TARGET_SR or ch != TARGET_CH or sw != TARGET_SW:
            report.failures.append(
                f"[{bank_name}] {fname}: wrong format {sr}Hz/{ch}ch/{sw*8}bit (expected 22050Hz/1ch/16bit)"
            )
        if rms < MIN_RMS:
            report.failures.append(
                f"[{bank_name}] {fname}: silent or near-silent (RMS={rms:.1f} < {MIN_RMS})"
            )
        if peak > MAX_PEAK:
            report.failures.append(
                f"[{bank_name}] {fname}: clipping detected (peak={peak} > {MAX_PEAK})"
            )
        if median_rms > 0.0 and rms >= MIN_RMS:
            db_diff = 20.0 * math.log10(rms / median_rms)
            if abs(db_diff) > MAX_DB_DEVIATION:
                report.failures.append(
                    f"[{bank_name}] {fname}: loudness {db_diff:+.2f} dB from median {median_rms:.1f} (limit ±{MAX_DB_DEVIATION} dB)"
                )

    return report


def audit_all(base_dir: str = PROJECT_ROOT) -> List[AuditReport]:
    """Audit all configured language audio banks."""
    banks = [
        ("en_characters", os.path.join(base_dir, "brailce", "audio", "characters")),
        ("en_words", os.path.join(base_dir, "brailce", "audio", "words_en")),
        ("filipino", os.path.join(base_dir, "brailce", "audio", "filipino")),
    ]
    return [audit_directory(name, path) for name, path in banks if os.path.isdir(path)]


def main() -> int:
    reports = audit_all()
    all_ok = True
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        print(f"{status}: {r.bank_name} ({r.file_count} files, median RMS={r.median_rms:.1f})")
        for fail in r.failures:
            print(f"  - {fail}")
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

