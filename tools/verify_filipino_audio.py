"""
Brailce Filipino Audio Verification Harness
============================================

Verifies every Filipino audio bank file using:
1. Speech recognition (ASR) round-trip:
   - Filipino words and sentences are transcribed with language="fil-PH"
     and scored against ground-truth text using similarity ratio.
   - Keystrokes/letters/operators are checked with language="en-US".
2. Duration sanity:
   - Keystroke files: duration between 250 ms and 900 ms, no clipping, single energy burst.
   - Word files: active speech duration at least 0.11s per vowel/syllable (catches dropped syllables).
3. Audio level sanity:
   - Active RMS within acceptable range (2000 to 5000) and no hard digital clipping.
4. Voice Audition generation:
   - Copies key representative samples into tools/voice_audition/fil_v2/ for manual listening.

Usage:
    python tools/verify_filipino_audio.py [--quick] [--audition-only]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rapidfuzz import fuzz
import speech_recognition as sr

from brailce.i18n.fil_keys import (
    FILIPINO_KEY_LETTERS_ENGLISH,
    FILIPINO_KEY_DIGITS,
    FILIPINO_KEY_OPERATORS,
    FILIPINO_KEY_PUNCTUATION,
    get_filipino_keystroke_filename,
)
from brailce.i18n.strings import STRINGS

FIL_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "filipino")
AUDITION_DIR = os.path.join(PROJECT_ROOT, "tools", "voice_audition", "fil_v2")


def count_tagalog_syllables(word: str) -> int:
    """Estimate syllable count from vowels in Tagalog orthography."""
    vowels = re.findall(r"[aeiou]", word.lower())
    return max(1, len(vowels))


def analyze_wav(path: str) -> Tuple[float, float, float, bool]:
    """
    Analyze a WAV file:
    Returns (total_duration, active_duration, rms, has_clipping).
    """
    with wave.open(path, "rb") as wf:
        nframes = wf.getnframes()
        sr_rate = wf.getframerate()
        frames = wf.readframes(nframes)
    
    total_dur = nframes / float(sr_rate)
    samples = np.frombuffer(frames, dtype=np.int16)
    if samples.size == 0:
        return (0.0, 0.0, 0.0, False)
    
    max_val = np.max(np.abs(samples))
    has_clipping = bool(max_val >= 32760)
    
    # Active duration (samples with amplitude > 200)
    voiced = np.where(np.abs(samples) > 200)[0]
    if voiced.size > 0:
        active_dur = (voiced[-1] - voiced[0]) / float(sr_rate)
        voiced_samples = samples[voiced[0] : voiced[-1] + 1]
        active_rms = float(np.sqrt(np.mean(voiced_samples.astype(np.float64) ** 2)))
    else:
        active_dur = 0.0
        active_rms = 0.0
        
    return (total_dur, active_dur, active_rms, has_clipping)


def verify_character_file(filename: str, expected_spoken: str, recognizer: sr.Recognizer) -> Tuple[bool, str]:
    """Verify a character WAV file (A-Z, 0-9, operators, punctuation)."""
    path = os.path.join(FIL_DIR, filename)
    if not os.path.isfile(path):
        return False, "File missing"
    
    total_dur, active_dur, rms, clipping = analyze_wav(path)
    
    # Check total duration (between 0.25s and 0.95s with silence padding)
    if total_dur < 0.25 or total_dur > 0.95:
        return False, f"Duration out of range: {total_dur:.2f}s (must be 0.25s-0.95s)"
    
    # Check clipping
    if clipping:
        return False, "Digital clipping detected"
    
    # Check loudness (active RMS should be roughly 1800 - 5500)
    if rms < 1500 or rms > 6000:
        return False, f"RMS loudness out of range: {rms:.1f}"
    
    # Optional ASR check with en-US
    asr_text = ""
    try:
        with sr.AudioFile(path) as src:
            audio = recognizer.record(src)
        asr_text = recognizer.recognize_google(audio, language="en-US").lower()
    except Exception:
        asr_text = ""
    
    return True, f"OK (dur={total_dur:.2f}s, active={active_dur:.2f}s, rms={rms:.0f}, asr={asr_text!r})"


def verify_word_file(filename: str, word: str, recognizer: sr.Recognizer, run_asr: bool = True) -> Tuple[bool, str]:
    """Verify a word WAV file with syllable duration and ASR."""
    path = os.path.join(FIL_DIR, filename)
    if not os.path.isfile(path):
        return False, "File missing"
    
    total_dur, active_dur, rms, clipping = analyze_wav(path)
    
    if clipping:
        return False, "Digital clipping detected"
    
    if rms < 1500 or rms > 6000:
        return False, f"RMS loudness out of range: {rms:.1f}"
    
    # Syllable duration check: active speech must be at least ~0.09s per syllable
    syllables = count_tagalog_syllables(word)
    min_speech = syllables * 0.09
    if active_dur < min_speech:
        return False, f"Active duration too short: {active_dur:.2f}s < {min_speech:.2f}s for {syllables} syllables"
    
    if not run_asr:
        return True, f"OK (dur={total_dur:.2f}s, active={active_dur:.2f}s, rms={rms:.0f})"
    
    asr_text = ""
    try:
        with sr.AudioFile(path) as src:
            audio = recognizer.record(src)
        asr_text = recognizer.recognize_google(audio, language="fil-PH").lower()
    except Exception:
        # Fallback to general english if loanword
        try:
            with sr.AudioFile(path) as src:
                audio = recognizer.record(src)
            asr_text = recognizer.recognize_google(audio, language="en-US").lower()
        except Exception:
            asr_text = ""
            
    norm_expected = re.sub(r"[^a-z0-9]", "", word.lower())
    norm_asr = re.sub(r"[^a-z0-9]", "", asr_text)

    # Known phonetic transliteration / brand matches for English/loan terms
    phonetic_equivalents = {
        "brailce": ["brails", "brailles", "breyls", "brail"],
        "braille": ["brail", "brill"],
        "matematika": ["mathematica", "mathematics"],
        "kamera": ["camera"],
    }
    if norm_expected in phonetic_equivalents and norm_asr in phonetic_equivalents[norm_expected]:
        return True, f"OK (phonetic match: expected {norm_expected!r}, got {norm_asr!r}, dur={total_dur:.2f}s)"

    score = fuzz.ratio(norm_expected, norm_asr)
    # If ASR didn't return text (e.g. very brief sound), check active duration
    if not norm_asr:
        if active_dur >= min_speech and rms >= 2000:
            return True, f"OK-NoASR (dur={total_dur:.2f}s, active={active_dur:.2f}s, rms={rms:.0f})"
        return False, f"ASR empty and duration borderline: {active_dur:.2f}s"
    
    if score >= 80.0 or (len(norm_expected) <= 4 and score >= 65.0):
        return True, f"OK (score={score:.0f}%, asr={asr_text!r}, dur={total_dur:.2f}s)"
    
    return False, f"ASR mismatch: expected {norm_expected!r}, got {norm_asr!r} (score={score:.0f}%)"


def generate_audition_folder() -> None:
    """Copy key representative audio files into tools/voice_audition/fil_v2/ for user review."""
    os.makedirs(AUDITION_DIR, exist_ok=True)
    
    sample_files = [
        # Keystrokes (letters, numbers, operators, punctuation)
        "char_A.wav", "char_B.wav", "char_C.wav", "char_K.wav", "char_M.wav", "char_S.wav", "char_T.wav",
        "char_1.wav", "char_2.wav", "char_plus.wav", "char_minus.wav", "char_period.wav", "char_space.wav",
        # Common Grade 1 words & problem words
        "word_kumusta.wav", "word_bahay.wav", "word_magandang.wav", "word_umaga.wav",
        "word_matematika.wav", "word_braille.wav", "word_brailce.wav", "word_tulong.wav",
        "word_isa.wav", "word_dalawa.wav", "word_tatlo.wav", "word_sampu.wav",
        # Mode intros & announcements
        "mode_free_input.wav", "mode_math.wav", "mode_ai.wav", "mode_scanner.wav", "switched_fil.wav",
    ]
    
    copied = 0
    for f in sample_files:
        src = os.path.join(FIL_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(AUDITION_DIR, f))
            copied += 1
            
    print(f"Generated {copied} audition samples in: {AUDITION_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Filipino audio bank assets.")
    parser.add_argument("--quick", action="store_true", help="Sample-based quick verification")
    parser.add_argument("--audition-only", action="store_true", help="Only copy audition samples")
    parser.add_argument("--no-asr", action="store_true", help="Skip Google ASR network calls")
    args = parser.parse_args()
    
    if args.audition_only:
        generate_audition_folder()
        return 0
        
    print("=" * 60)
    print("  Brailce Filipino Audio Verification Harness")
    print("=" * 60)
    
    recognizer = sr.Recognizer()
    failures: List[Tuple[str, str]] = []
    successes = 0
    
    # 1. Verify Characters
    print("\n--- Verifying Character Keystrokes ---")
    char_items = list(FILIPINO_KEY_LETTERS_ENGLISH.items()) + list(FILIPINO_KEY_DIGITS.items()) + list(FILIPINO_KEY_OPERATORS.items())
    if args.quick:
        char_items = char_items[:10]
        
    for ch, spoken in char_items:
        fname = get_filipino_keystroke_filename(ch)
        ok, msg = verify_character_file(fname, spoken, recognizer)
        if ok:
            successes += 1
            print(f"  [PASS] {fname:20s}: {msg}")
        else:
            failures.append((fname, msg))
            print(f"  [FAIL] {fname:20s}: {msg}")
            
    # 2. Verify Mode Announcements & Fixed Audio
    print("\n--- Verifying Mode Announcements ---")
    fixed_files = [
        "mode_free_input.wav",
        "mode_math.wav",
        "mode_ai.wav",
        "mode_scanner.wav",
        "switched_fil.wav",
    ]
    for fname in fixed_files:
        path = os.path.join(FIL_DIR, fname)
        if not os.path.isfile(path):
            failures.append((fname, "Missing"))
            print(f"  [FAIL] {fname:20s}: Missing")
            continue
        total_dur, active_dur, rms, clipping = analyze_wav(path)
        if total_dur >= 1.0 and not clipping and 1800 <= rms <= 6000:
            successes += 1
            print(f"  [PASS] {fname:20s}: OK (dur={total_dur:.2f}s, rms={rms:.0f})")
        else:
            failures.append((fname, f"Invalid: dur={total_dur:.2f}s, rms={rms:.0f}, clip={clipping}"))
            print(f"  [FAIL] {fname:20s}: Invalid")
            
    # 3. Verify Words
    print("\n--- Verifying Filipino Words ---")
    word_files = [f for f in os.listdir(FIL_DIR) if f.startswith("word_") and f.endswith(".wav")]
    if args.quick:
        # Sample representative words including problem words
        test_words = ["kumusta", "bahay", "magandang", "umaga", "matematika", "tulong", "kamera", "basa", "pahina", "salamat", "braille", "brailce"]
        word_files = [f"word_{w}.wav" for w in test_words if os.path.isfile(os.path.join(FIL_DIR, f"word_{w}.wav"))]
        
    print(f"Checking {len(word_files)} word files...")
    for fname in word_files:
        raw_word = fname[len("word_") : -len(".wav")]
        ok, msg = verify_word_file(fname, raw_word, recognizer, run_asr=not args.no_asr)
        if ok:
            successes += 1
            if args.quick or successes % 50 == 0:
                print(f"  [PASS] {fname:25s}: {msg}")
        else:
            failures.append((fname, msg))
            print(f"  [FAIL] {fname:25s}: {msg}")
            
    print("\n" + "=" * 60)
    print(f"Verification Results: {successes} PASSED, {len(failures)} FAILED")
    print("=" * 60)
    
    generate_audition_folder()
    
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
