"""
Filipino Clarity Post-Processor and Missing Audio Generator
============================================================

1. Generates missing English-in-Filipino and function word clips (is, his, this, the, was, etc.)
   using `fil-PH-BlessicaNeural` with `rate="-5%"`.
2. Re-synthesizes single letter clips (A-Z) with clean onset/tail padding and `rate="-5%"`.
3. Applies high-pass 80 Hz, +3dB high-shelf presence boost (2.5-6 kHz), and active RMS ~4400
   normalization with soft knee limiter (peak <= 30000, no clipping).
4. Post-processes all existing Filipino WAVs to eliminate muffling and match English baseline.
5. Produces `tools/voice_audition/fil_clarity_after/` with letters, digits, key words,
   and 10 sentences rendered at default (-5%), 0%, and -10% rate settings for human audition.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import edge_tts
import miniaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIL_DIR = PROJECT_ROOT / "brailce" / "audio" / "filipino"
AUDITION_DIR = PROJECT_ROOT / "tools" / "voice_audition" / "fil_clarity_after"

VOICE = "fil-PH-BlessicaNeural"

# Clean spoken letter names
BLESSICA_LETTERS: Dict[str, str] = {
    "A": "ey.", "B": "bi.", "C": "si.", "D": "di.", "E": "i.",
    "F": "ef.", "G": "dyi.", "H": "eych.", "I": "ay.", "J": "dyey.",
    "K": "key.", "L": "el.", "M": "em.", "N": "en.", "O": "o.",
    "P": "pi.", "Q": "kyu.", "R": "ar.", "S": "es.", "T": "ti.",
    "U": "yu.", "V": "vi.", "W": "dobolyu.", "X": "eks.", "Y": "way.",
    "Z": "zi.",
}

# Missing English function words and common terms
MISSING_WORDS: Dict[str, str] = {
    "is": "is.", "his": "his.", "this": "this.", "the": "the.", "was": "was.",
    "what": "what.", "when": "when.", "where": "where.", "which": "which.",
    "who": "who.", "why": "why.", "with": "with.", "you": "you.", "and": "and.",
    "are": "are.", "in": "in.", "that": "that.", "can": "can.", "cat": "cat.",
    "dog": "dog.", "book": "book.", "page": "page.", "read": "read.", "help": "help.",
    "it": "it.", "its": "its.", "they": "they.", "we": "we.", "do": "do.",
    "have": "have.", "for": "for.", "not": "not.", "on": "on.", "at": "at.",
    "by": "by.", "from": "from.", "as": "as.", "be": "be.", "to": "to.",
    "of": "of.", "so": "so.", "up": "up.", "out": "out.", "if": "if.",
    "or": "or.", "an": "an.", "my": "my.", "one": "one.", "all": "all.",
    "there": "there.", "their": "their.", "get": "get.", "go": "go.", "me": "me.",
    "0": "sero.", "1": "isa.", "2": "dalawa.", "3": "tatlo.", "4": "apat.",
    "5": "lima.", "6": "anim.", "7": "pito.", "8": "walo.", "9": "siyam.",
    "space": "ispeys.", "backspace": "bak-ispeys.", "tab": "tab.",
    "mode": "mowd.", "camera": "kamera.", "type": "tayp.", "ok": "okay.",
    "lcd": "el-si-di.", "led": "el-i-di.", "usb": "yu-es-bi.", "wifi": "way-fay.",
}

# 10 representative sentences for rate audition
AUDITION_SENTENCES: List[Tuple[str, str]] = [
    ("s01_kumusta", "Kumusta ka! Masaya akong mag-aral ng Braille ngayon."),
    ("s02_araw", "Ano ang araw ngayon? Magandang umaga sa iyo."),
    ("s03_math_add", "Limang mansanas dagdag tatlong mansanas ay walong mansanas."),
    ("s04_math_sub", "Sampu bawas apat ay anim."),
    ("s05_free_input", "I-type ang iyong pangalan gamit ang mga pindutan ng Braille."),
    ("s06_scanner_ready", "Ilagay ang pahina ng Braille sa tapat ng kamera, pagkatapos ay pindutin ang Enter."),
    ("s07_scanner_scan", "Sinusuri ang pahina ng Braille. Mangyaring maghintay."),
    ("s08_english_words", "This is how we learn to read and write with Braille."),
    ("s09_ai_thinking", "Nag-iisip, mangyaring maghintay nang sandali."),
    ("s10_kita_sample", "Nakitang malinaw ang mga tuldok sa papel na ito."),
]


def apply_clarity_dsp(
    samples: np.ndarray,
    sr: int = 22050,
    lead_silence_ms: int = 20,
    trail_silence_ms: int = 90,
    target_rms: float = 4400.0,
    presence_boost_db: float = 3.0,
) -> bytes:
    """
    Apply professional clarity DSP:
    - Zero DC offset
    - Trim dead silence with 80 amplitude floor
    - 80 Hz high-pass (zero-phase Butterworth)
    - Gentle +3 dB high-shelf boost from 3 kHz upwards
    - Active RMS normalization to 4400
    - Soft knee tanh limiter (cap 30000, <= -1 dBFS, no clipping)
    - Controlled lead and trail silence padding with 8ms cosine fade
    """
    x = samples.astype(np.float64)
    x -= np.mean(x)

    abs_x = np.abs(x)
    nz = np.where(abs_x > 80)[0]
    if len(nz) == 0:
        return b"\x00\x00" * int(sr * 0.2)

    # Keep a gentle margin around speech
    start = max(0, nz[0] - int(sr * 0.015))
    end = min(len(x), nz[-1] + int(sr * 0.040))
    seg = x[start:end]

    # Smooth 8ms cosine fade in/out
    fade_len = min(int(sr * 0.008), len(seg) // 4)
    if fade_len > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        seg[:fade_len] *= fade_in
        seg[-fade_len:] *= fade_out

    # FFT-domain high-pass + high-shelf EQ
    n = len(seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    hp = 1.0 / np.sqrt(1.0 + (80.0 / np.maximum(freqs, 1.0)) ** 4)
    gain = 10.0 ** (presence_boost_db / 20.0)
    shelf = 1.0 + (gain - 1.0) / (1.0 + np.exp(-(freqs - 3000.0) / 700.0))
    H = hp * shelf

    Y = np.fft.irfft(np.fft.rfft(seg) * H, n=n)

    # Active RMS normalization
    act = Y[np.abs(Y) > 150]
    rms = np.sqrt(np.mean(act ** 2)) if len(act) > 0 else 1.0
    if rms > 1e-3:
        Y = Y * (target_rms / rms)

    # Soft limiter (tanh above 26000 up to 30000)
    thresh = 26000.0
    over = np.abs(Y) > thresh
    if np.any(over):
        Y[over] = np.sign(Y[over]) * (
            thresh + (30000.0 - thresh) * np.tanh((np.abs(Y[over]) - thresh) / (30000.0 - thresh))
        )

    pcm_core = np.clip(np.round(Y), -30000, 30000).astype(np.int16).tobytes()
    lead_silence = b"\x00\x00" * int(sr * (lead_silence_ms / 1000.0))
    trail_silence = b"\x00\x00" * int(sr * (trail_silence_ms / 1000.0))

    return lead_silence + pcm_core + trail_silence


async def synth_text_to_pcm(text: str, voice: str = VOICE, rate: str = "-5%") -> Optional[np.ndarray]:
    """Synthesize text via edge-tts, decode MP3 to 22050 Hz s16 mono samples."""
    for attempt in range(3):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate)
            mp3_bytes = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])

            if not mp3_bytes:
                await asyncio.sleep(0.5)
                continue

            dec = miniaudio.mp3_read_s16(bytes(mp3_bytes))
            raw = np.frombuffer(dec.samples, dtype=np.int16)
            if dec.nchannels == 2:
                raw = raw.reshape(-1, 2).mean(axis=1).astype(np.int16)

            # Resample to 22050
            target_len = int(len(raw) * 22050 / dec.sample_rate)
            resampled = np.interp(
                np.linspace(0, len(raw), target_len, endpoint=False),
                np.arange(len(raw)),
                raw,
            ).astype(np.int16)
            return resampled
        except Exception as exc:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def run_pipeline():
    FIL_DIR.mkdir(parents=True, exist_ok=True)
    AUDITION_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(10)

    print("--- 1. Generating Missing Words & Clear Letters ---")
    gen_tasks = []

    # 1. Letters A-Z
    for letter, spoken in BLESSICA_LETTERS.items():
        out_p = FIL_DIR / f"char_{letter}.wav"
        out_lower = FIL_DIR / f"char_{letter.lower()}.wav"
        gen_tasks.append((spoken, out_p, "-5%"))
        gen_tasks.append((spoken, out_lower, "-5%"))

    # 2. Missing English and function words
    for w, spoken in MISSING_WORDS.items():
        out_p = FIL_DIR / f"word_{w}.wav"
        gen_tasks.append((spoken, out_p, "-5%"))

    async def _synth_and_save(text: str, path: Path, rate: str):
        async with sem:
            samples = await synth_text_to_pcm(text, VOICE, rate=rate)
            if samples is not None:
                pcm_data = apply_clarity_dsp(samples, sr=22050, lead_silence_ms=20, trail_silence_ms=90)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(pcm_data)
                return True
            return False

    coros = [_synth_and_save(t, p, r) for t, p, r in gen_tasks]
    results = await asyncio.gather(*coros)
    print(f"Generated {sum(1 for r in results if r)}/{len(gen_tasks)} target files.")

    print("\n--- 2. Post-Processing Existing Filipino WAVs with Clarity DSP ---")
    all_fil_wavs = sorted(FIL_DIR.glob("*.wav"))
    processed_count = 0
    for p in all_fil_wavs:
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                sw = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            if sw == 2 and sr == 22050:
                data = np.frombuffer(raw, dtype=np.int16)
                # Check current active RMS; if below 4000 or muffled, re-process
                abs_data = np.abs(data)
                act = data[abs_data > 150]
                rms = np.sqrt(np.mean(act ** 2)) if len(act) > 0 else 0.0
                if rms < 4000.0 or "char_" in p.name:
                    pcm_out = apply_clarity_dsp(data, sr=22050)
                    with wave.open(str(p), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(22050)
                        wf.writeframes(pcm_out)
                    processed_count += 1
        except Exception as exc:
            print(f"Error processing {p.name}: {exc}")

    print(f"Post-processed {processed_count} files with high-pass, presence boost, and RMS normalization.")

    print("\n--- 3. Generating Audition Set (tools/voice_audition/fil_clarity_after/) ---")
    # Copy representative characters
    for c in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["0", "1", "2", "plus", "minus", "period", "comma"]:
        src = FIL_DIR / f"char_{c}.wav"
        if src.exists():
            shutil.copy2(src, AUDITION_DIR / f"char_{c}.wav")

    # Copy key words
    for w in ("is", "his", "this", "the", "was", "kumusta", "bahay", "araw", "kita", "matematika"):
        src = FIL_DIR / f"word_{w}.wav"
        if src.exists():
            shutil.copy2(src, AUDITION_DIR / f"word_{w}.wav")

    # Generate 10 sentences at 3 rates (-5% default, 0% normal, -10% slow)
    for slug, sent_text in AUDITION_SENTENCES:
        for rate, label in (("-5%", "default_rate"), ("0%", "normal_rate"), ("-10%", "slow_rate")):
            out_p = AUDITION_DIR / f"{slug}_{label}.wav"
            samples = await synth_text_to_pcm(sent_text, VOICE, rate=rate)
            if samples is not None:
                pcm_data = apply_clarity_dsp(samples, sr=22050, lead_silence_ms=30, trail_silence_ms=120)
                with wave.open(str(out_p), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(pcm_data)

    print("Audition set complete!")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
