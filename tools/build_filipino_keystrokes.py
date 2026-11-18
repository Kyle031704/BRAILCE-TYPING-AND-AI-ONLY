"""
Brailce Filipino Keystroke Builder (Digits, Operators, Punctuation)
===================================================================
Renders Filipino punctuation with fil-PH-BlessicaNeural (tuldok, kuwit,
tandang pananong, tandang padamdam), re-renders outlier digits/operators with
en-PH-RosaNeural, and normalizes all clips to match Section A letters:
lead: 15ms, tail: 80ms, active RMS: ~4000.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

import edge_tts
import numpy as np
import soundfile as sf
import scipy.signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brailce.speech.audio_normalize import slice_carrier_audio
from tools.letter_probe import trim_and_pad, estimate_pitch_track

FIL_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "filipino")
SR = 22050

PUNCTUATION_TARGETS = {
    "char_period.wav": "Tuldok.",
    "char_comma.wav": "Kuwit.",
    "char_question.wav": "Tandang pananong.",
    "char_exclamation.wav": "Tandang padamdam.",
}

DIGIT_OPERATOR_TARGETS = {
    "char_0.wav": "Please say: zero.",
    "char_1.wav": "Please say: one.",
    "char_2.wav": "Please say: two.",
    "char_3.wav": "Please say: three.",
    "char_4.wav": "Please say: four.",
    "char_5.wav": "Please say: five.",
    "char_6.wav": "Please say: six.",
    "char_7.wav": "Please say: seven.",
    "char_8.wav": "Please say: eight.",
    "char_9.wav": "Please say: nine.",
    "char_plus.wav": "Please say: plus.",
    "char_minus.wav": "Please say: minus.",
    "char_times.wav": "Please say: times.",
    "char_divide.wav": "Please say: divided by.",
    "char_equals.wav": "Please say: equals.",
}

async def synthesize(text: str, voice: str) -> np.ndarray:
    comm = edge_tts.Communicate(text, voice)
    data = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    sig, sr = sf.read(io.BytesIO(data))
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != SR:
        sig = scipy.signal.resample(sig, int(round(len(sig) * SR / sr)))
    return sig

async def main():
    print("1. Rendering Filipino punctuation with Blessica...")
    for filename, text in PUNCTUATION_TARGETS.items():
        sig = await synthesize(text, "fil-PH-BlessicaNeural")
        clean = trim_and_pad(sig, SR, target_rms=4000.0, lead_ms=15, tail_ms=80, fade_ms=8)
        out_p = os.path.join(FIL_DIR, filename)
        sf.write(out_p, clean, SR, subtype="PCM_16")
        f0, slope, _ = estimate_pitch_track(clean, SR)
        print(f"  {filename:<22} -> '{text}' | dur={len(clean)/SR:.2f}s | F0={f0:.0f}Hz")

    print("\n2. Normalizing digits & math operators with Rosa...")
    for filename, carrier in DIGIT_OPERATOR_TARGETS.items():
        out_p = os.path.join(FIL_DIR, filename)
        # Re-render to ensure same carrier & uniform intonation
        raw = await synthesize(carrier, "en-PH-RosaNeural")
        raw16 = (raw * 32767.0).astype(np.int16)
        sliced_int = slice_carrier_audio(raw16, SR)
        sig = np.array(sliced_int, dtype=np.float32) / 32767.0
        clean = trim_and_pad(sig, SR, target_rms=4000.0, lead_ms=15, tail_ms=80, fade_ms=8)
        sf.write(out_p, clean, SR, subtype="PCM_16")
        f0, slope, _ = estimate_pitch_track(clean, SR)
        print(f"  {filename:<22} -> dur={len(clean)/SR:.2f}s | F0={f0:.0f}Hz | slope={slope:.0f}")

    print("\nAll Filipino digits, operators, and punctuation regenerated & normalized.")

if __name__ == "__main__":
    asyncio.run(main())
