"""
Rebuild Filipino Audio Bank using ONE consistent female voice: fil-PH-BlessicaNeural
=====================================================================================

1. Re-renders all character keystroke audio (`char_*.wav`) using Blessica with natural
   letter names (A="ey.", B="bi.", etc.) without carrier slicing or raspiness.
2. Re-renders all 38 male voice offenders identified by `audit_filipino_gender.py`.
3. Synthesizes `word_kita.wav`.
4. Pre-renders all multi-word phrases in `STRINGS` and mode intros as both named files
   and `sentence_<md5>.wav` hashes to guarantee zero choppy concatenation at runtime.
5. Standardizes all WAV files to 22050 Hz, 16-bit mono PCM.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import edge_tts
import miniaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIL_DIR = PROJECT_ROOT / "brailce" / "audio" / "filipino"

VOICE = "fil-PH-BlessicaNeural"

# Clean Blessica spoken names
BLESSICA_LETTERS: Dict[str, str] = {
    "A": "ey.",
    "B": "bi.",
    "C": "si.",
    "D": "di.",
    "E": "i.",
    "F": "ef.",
    "G": "dyi.",
    "H": "eych.",
    "I": "ay.",
    "J": "dyey.",
    "K": "key.",
    "L": "el.",
    "M": "em.",
    "N": "en.",
    "O": "o.",
    "P": "pi.",
    "Q": "kyu.",
    "R": "ar.",
    "S": "es.",
    "T": "ti.",
    "U": "yu.",
    "V": "vi.",
    "W": "dobolyu.",
    "X": "eks.",
    "Y": "way.",
    "Z": "zi.",
}

BLESSICA_SPECIAL: Dict[str, Tuple[str, str]] = {
    # char: (filename_base, spoken_text)
    "0": ("char_0.wav", "zero."),
    "1": ("char_1.wav", "one."),
    "2": ("char_2.wav", "two."),
    "3": ("char_3.wav", "three."),
    "4": ("char_4.wav", "four."),
    "5": ("char_5.wav", "five."),
    "6": ("char_6.wav", "six."),
    "7": ("char_7.wav", "seven."),
    "8": ("char_8.wav", "eight."),
    "9": ("char_9.wav", "nine."),
    "+": ("char_plus.wav", "plus."),
    "-": ("char_minus.wav", "minus."),
    "*": ("char_times.wav", "times."),
    "/": ("char_divided.wav", "divided by."),
    "=": ("char_equals.wav", "equals."),
    ".": ("char_period.wav", "period."),
    ",": ("char_comma.wav", "comma."),
    "?": ("char_question.wav", "question mark."),
    "!": ("char_exclamation.wav", "exclamation mark."),
    "'": ("char_apostrophe.wav", "apostrophe."),
    ":": ("char_colon.wav", "colon."),
    ";": ("char_semicolon.wav", "semicolon."),
    "#": ("char_number_sign.wav", "number sign."),
    " ": ("char_space.wav", "space."),
    "-hyphen": ("char_hyphen.wav", "hyphen."),
}

# The 38 male voice offenders identified by audit_filipino_gender.py
MALE_OFFENDERS: Dict[str, str] = {
    "ai_pangalan.wav": "Ang pangalan ko ay Brailce.",
    "ai_umaga.wav": "Magandang umaga sa iyo.",
    "char_minus.wav": "minus.",
    "word_1824.wav": "isang libo walong daan dalawampu't apat.",
    "word_binura.wav": "binura.",
    "word_bolyum.wav": "bolyum.",
    "word_device.wav": "device.",
    "word_dr.wav": "doktor.",
    "word_eagle.wav": "agila.",
    "word_earth.wav": "mundo.",
    "word_france.wav": "pransya.",
    "word_hydrogen.wav": "hydroheno.",
    "word_itype.wav": "i-type.",
    "word_jupiter.wav": "jupiter.",
    "word_laba.wav": "laba.",
    "word_learning.wav": "learning.",
    "word_luto.wav": "luto.",
    "word_mabagal.wav": "mabagal.",
    "word_mabait.wav": "mabait.",
    "word_mabilis.wav": "mabilis.",
    "word_mag-scan.wav": "mag-scan.",
    "word_mag-type.wav": "mag-type.",
    "word_mahina.wav": "mahina.",
    "word_malakas.wav": "malakas.",
    "word_mali.wav": "mali.",
    "word_mundo.wav": "mundo.",
    "word_oxygen.wav": "oksiheno.",
    "word_paalam.wav": "paalam.",
    "word_pag-type.wav": "pag-type.",
    "word_pagpindot.wav": "pagpindot.",
    "word_panuri.wav": "panuri.",
    "word_philippine.wav": "pilipinas.",
    "word_sara.wav": "sara.",
    "word_scanner.wav": "scanner.",
    "word_system.wav": "sistema.",
    "word_tama.wav": "tama.",
    "word_tanghali.wav": "tanghali.",
    "word_voice.wav": "boses.",
}


def standardize_wav_data(samples: np.ndarray, sr: int = 22050, pre_ms: int = 40, post_ms: int = 80) -> bytes:
    """Trim dead silence, apply gentle fade envelope, standardize to 16-bit PCM."""
    x = samples.astype(np.float64)
    # Trim leading and trailing quiet silence
    nz = np.where(np.abs(x) > 150)[0]
    if nz.size > 0:
        pad = int(sr * 0.02)
        x = x[max(0, nz[0] - pad) : min(len(x), nz[-1] + pad)]

    # RMS normalization to ~3200 (clear, unclipped)
    rms = np.sqrt(np.mean(x ** 2)) if len(x) > 0 else 0.0
    if rms > 1e-3:
        target_rms = 3200.0
        x = x * (target_rms / rms)

    # Hard clip safety
    x = np.clip(x, -32000, 32000)

    # Edge fade to prevent pops
    fade_len = min(int(sr * 0.005), len(x) // 8)
    if fade_len > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        x[:fade_len] *= fade_in
        x[-fade_len:] *= fade_out

    pcm = np.round(x).astype(np.int16).tobytes()
    pre_silence = b"\x00\x00" * int(sr * (pre_ms / 1000.0))
    post_silence = b"\x00\x00" * int(sr * (post_ms / 1000.0))
    return pre_silence + pcm + post_silence


async def synthesize_one(text: str, out_path: Path, sem: asyncio.Semaphore) -> bool:
    async with sem:
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text, VOICE)
                mp3_bytes = bytearray()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        mp3_bytes.extend(chunk["data"])

                if not mp3_bytes:
                    await asyncio.sleep(0.5)
                    continue

                decoded = miniaudio.mp3_read_s16(bytes(mp3_bytes))
                raw_samples = np.frombuffer(decoded.samples, dtype=np.int16)
                if decoded.nchannels == 2:
                    raw_samples = raw_samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Resample to 22050 Hz
                target_len = int(len(raw_samples) * 22050 / decoded.sample_rate)
                resampled = np.interp(
                    np.linspace(0, len(raw_samples), target_len, endpoint=False),
                    np.arange(len(raw_samples)),
                    raw_samples,
                ).astype(np.int16)

                final_pcm = standardize_wav_data(resampled, sr=22050)

                with wave.open(str(out_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(final_pcm)

                return True
            except Exception as exc:
                await asyncio.sleep(0.5 * (attempt + 1))
        return False


async def run_rebuild():
    FIL_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(10)
    tasks = []

    # 1. Letters A-Z
    print("Scheduling letters A-Z...")
    for letter, spoken in BLESSICA_LETTERS.items():
        out_p = FIL_DIR / f"char_{letter}.wav"
        tasks.append((spoken, out_p))
        # Lowercase copy target
        out_lower = FIL_DIR / f"char_{letter.lower()}.wav"
        tasks.append((spoken, out_lower))

    # 2. Digits, Operators, Punctuation
    print("Scheduling digits, operators, punctuation...")
    for ch, (fname, spoken) in BLESSICA_SPECIAL.items():
        out_p = FIL_DIR / fname
        tasks.append((spoken, out_p))

    # 3. Male voice offenders
    print("Scheduling male offenders re-render...")
    for fname, spoken in MALE_OFFENDERS.items():
        out_p = FIL_DIR / fname
        tasks.append((spoken, out_p))

    # 4. Word "kita" and related
    print("Scheduling 'kita'...")
    tasks.append(("kita.", FIL_DIR / "word_kita.wav"))
    tasks.append(("nakita.", FIL_DIR / "word_nakita.wav"))
    tasks.append(("nakitang.", FIL_DIR / "word_nakitang.wav"))
    tasks.append(("makita.", FIL_DIR / "word_makita.wav"))

    # 5. Fixed multi-word phrases from STRINGS (pre-rendered whole sentence files & hashes)
    sys.path.insert(0, str(PROJECT_ROOT))
    from brailce.i18n.strings import STRINGS
    from brailce.speech.filipino_engine import normalize_for_sherpa_filipino

    fixed_aliases = {
        "mode_free_input.wav": STRINGS["MODE_INTRO_FREE_INPUT"]["fil"],
        "mode_math.wav": STRINGS["MODE_INTRO_MATH"]["fil"],
        "mode_ai.wav": STRINGS["MODE_INTRO_AI"]["fil"],
        "mode_scanner.wav": STRINGS["MODE_INTRO_SCANNER"]["fil"],
        "switched_fil.wav": STRINGS["LANG_SWITCHED_FIL"]["fil"],
        "thinking_fil.wav": STRINGS["AI_THINKING"]["fil"],
        "scanning_fil.wav": STRINGS["SCANNER_SCANNING"]["fil"],
        "no_braille_fil.wav": STRINGS["SCANNER_NO_BRAILLE"]["fil"],
    }

    print("Scheduling UI strings and sentence hash caches...")
    for fname, phrase in fixed_aliases.items():
        out_p = FIL_DIR / fname
        tasks.append((phrase, out_p))

    for k, entry in STRINGS.items():
        if "fil" in entry:
            phrase = entry["fil"]
            words = phrase.split()
            if len(words) >= 2:
                # Store under md5 of normalized string
                norm_key = normalize_for_sherpa_filipino(phrase) or phrase.lower()
                h1 = hashlib.md5(norm_key.encode("utf-8")).hexdigest()
                tasks.append((phrase, FIL_DIR / f"sentence_{h1}.wav"))
                # Also store under raw phrase md5
                h2 = hashlib.md5(phrase.strip().lower().encode("utf-8")).hexdigest()
                if h2 != h1:
                    tasks.append((phrase, FIL_DIR / f"sentence_{h2}.wav"))

    print(f"Total files to synthesize: {len(tasks)}")
    coros = [synthesize_one(text, path, sem) for text, path in tasks]
    results = await asyncio.gather(*coros)
    succeeded = sum(1 for r in results if r)
    print(f"Successfully generated {succeeded}/{len(tasks)} files.")


if __name__ == "__main__":
    asyncio.run(run_rebuild())
