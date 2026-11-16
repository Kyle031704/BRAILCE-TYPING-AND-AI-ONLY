"""
Brailce Unified Audio Bank Builder
====================================

Regenerates and standardizes the English and Filipino audio banks from a single
manifest derived directly from:
  - brailce/i18n/strings.py
  - brailce/braille/arithmetic.py (math explainer vocabulary + Tagalog 0-999 tokens)
  - brailce/modes/ai_assistant.py & brailce/ai/ollama_client.py (Q&A vocabulary)
  - Common Grade 1 English (~1,000 words) & Filipino vocabulary lists

Voices:
  - English : Piper en_US-ryan-medium (length_scale=1.05, noise_scale=0.5, noise_w=0.6)
  - Filipino: fil-PH-BlessicaNeural (words, sentences, modes, AI) &
              en-PH-RosaNeural with carrier slicing (letters, digits, operators, punctuation)
              [fallback to sherpa-onnx MMS-TTS tgl if offline]

All generated/updated WAV files are standardized via standardize_wav_file():
  22050 Hz, mono (1 ch), 16-bit PCM, active RMS ~4200, 120ms pre / 180ms post silence.

Usage:
  python tools/build_audio_bank.py --filipino-only
  python tools/build_audio_bank.py --all
  python tools/build_audio_bank.py --normalize-en-only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
import wave
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brailce.ai.ollama_client import FACTUAL_QA_RESPONSES
from brailce.braille.arithmetic import SafeArithmetic
from brailce.i18n.strings import STRINGS
from brailce.modes.ai_assistant import FILIPINO_AI_OFFLINE_RESPONSES
from brailce.speech.audio_normalize import (
    DEFAULT_POST_SILENCE_MS,
    DEFAULT_PRE_SILENCE_MS,
    slice_carrier_audio,
    standardize_wav_file,
)
from brailce.speech.filipino_engine import (
    FilipinoEngine,
    TAGALOG_HUNDREDS,
    TAGALOG_ONES,
    TAGALOG_TEENS,
    TAGALOG_TENS,
    normalize_for_sherpa_filipino,
)
from brailce.i18n.fil_keys import (
    FILIPINO_KEY_LETTERS_ENGLISH,
    FILIPINO_KEY_DIGITS,
    FILIPINO_KEY_OPERATORS,
    FILIPINO_KEY_PUNCTUATION,
    get_filipino_keystroke_filename,
)
from brailce.speech.normalizer import CHARACTER_FILENAME_MAP
from brailce.speech.piper_engine import PiperEngine

EN_CHAR_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "characters")
EN_WORD_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "words_en")
FIL_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "filipino")

FILIPINO_BANK_VOICE = "fil-PH-BlessicaNeural"
FILIPINO_KEYSTROKE_VOICE = "en-PH-RosaNeural"

# Core Grade 1 English sight words + problem words + Dolch/Fry high-frequency vocabulary
ENGLISH_CORE_WORDS: List[str] = sorted({
    # Section A regression list
    "is", "his", "this", "its", "island", "miss",
    "was", "what", "when", "where", "who", "why", "which", "white",
    "whale", "wheel", "while", "with", "without", "will", "we", "were",
    "water", "window", "wolf", "woman", "work", "world", "would", "write", "wrong",
    # High-frequency Dolch / Fry / classroom vocabulary
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for", "not", "on",
    "he", "as", "you", "do", "at", "but", "by", "from", "they", "say", "her", "she",
    "or", "an", "my", "one", "all", "there", "their", "so", "up", "out", "if", "about",
    "get", "go", "me", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "braille", "need",
    "hello", "cat", "dog", "book", "page", "read", "letter", "word", "number", "plus",
    "minus", "times", "divided", "equals", "zero", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "yes", "please", "thank", "help", "school",
    "teacher", "student", "learn", "home", "hand", "finger", "dot", "dots", "cell",
    "sun", "moon", "star", "earth", "sky", "rain", "tree", "apple", "ball", "house",
    "mom", "dad", "friend", "happy", "big", "small", "hot", "cold", "fast", "slow",
    "red", "blue", "green", "yellow", "black", "brown", "orange", "pink", "purple",
    "up", "down", "left", "right", "open", "close", "start", "stop", "run", "walk",
    "sit", "stand", "eat", "drink", "sleep", "play", "sing", "count", "add", "subtract",
})

# Core Grade 1 Filipino sight words (in addition to all strings extracted from i18n/math/AI)
FILIPINO_SIGHT_WORDS: List[str] = [
    "ano", "sino", "saan", "kailan", "bakit", "paano",
    "ito", "iyan", "iyon", "ako", "ikaw", "siya", "kami", "tayo", "sila",
    "ko", "mo", "niya", "namin", "natin", "nila",
    "ang", "mga", "sa", "ng", "na", "at", "ay", "ba", "pa", "po", "opo", "oo", "hindi",
    "para", "kung", "dahil", "pero", "kaya", "wala", "may", "mayroon",
    "bata", "aso", "pusa", "ibon", "isda", "baka", "baboy", "manok",
    "nanay", "tatay", "ate", "kuya", "bunso", "lolo", "lola", "guro", "kaibigan",
    "bahay", "paaralan", "silid", "mesa", "silya", "libro", "aklat", "lapis", "papel", "pisara", "bag",
    "damit", "sapatos", "pagkain", "tubig", "kanin", "ulam", "gatas", "tinapay", "prutas", "gulay",
    "saging", "mangga", "mansanas", "araw", "buwan", "bituin", "ulan", "hangin", "ulap", "apoy",
    "lupa", "puno", "bulaklak", "dahon", "mata", "ilong", "bibig", "tainga", "kamay", "paa",
    "ulo", "buhok", "ngipin", "puso", "daliri", "bulag", "braille", "brailce",
    "mabuti", "masaya", "malungkot", "galit", "takot", "gutom", "uhaw", "pagod", "tulog", "gising",
    "malaki", "maliit", "mahaba", "maikli", "mataas", "mababa", "maganda", "malinis", "marumi",
    "bago", "luma", "mainit", "malamig", "matamis", "maalat", "maasim", "mapait",
    "pula", "asul", "dilaw", "berde", "puti", "itim", "kayumanggi",
    "kain", "inom", "tulog", "lakad", "takbo", "upo", "tayo", "basa", "sulat", "bilang",
    "pindot", "pakinig", "tingin", "hawak", "kuha", "bigay", "tulong", "aral", "laro",
    "umaga", "hapon", "gabi", "ngayon", "bukas", "kahapon", "oras", "taon", "buhay", "edad",
    "pangalan", "pamilya", "wika", "filipino", "pilipinas", "salamat", "kumusta", "magandang",
    "mahal", "pambansang", "bayani", "kabisera", "maynila",
]

# Plain English pronunciation targets for character carrier-slicing
PLAIN_KEYSTROKE_TARGETS: Dict[str, str] = {
    # Letters A-Z
    **{chr(c): chr(c) for c in range(ord("A"), ord("Z") + 1)},
    # Digits 0-9
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    # Math Operators
    "+": "plus",
    "-": "minus",
    "*": "times",
    "/": "divided by",
    "=": "equals",
    # Punctuation
    ".": "period",
    ",": "comma",
    "?": "question mark",
    "!": "exclamation mark",
    "-": "hyphen",
    "'": "apostrophe",
    ":": "colon",
    ";": "semicolon",
    "#": "number sign",
    " ": "space",
}


def collect_filipino_words() -> Set[str]:
    """Collect every Filipino word token needed by i18n, SafeArithmetic, AI offline dict, and sight words."""
    tokens: Set[str] = set(FILIPINO_SIGHT_WORDS)

    # 1. Tagalog numbers 0-999 vocabulary
    for v in TAGALOG_ONES.values():
        tokens.add(v)
    for v in TAGALOG_TEENS.values():
        tokens.add(v)
    for v in TAGALOG_TENS.values():
        tokens.add(v)
        for o_idx, o_val in TAGALOG_ONES.items():
            if o_idx > 0:
                tokens.add(f"{v}'t")
                tokens.add(o_val)
    for v in TAGALOG_HUNDREDS.values():
        for w in v.split():
            tokens.add(w)

    # 2. All i18n strings (FIL)
    phrases: List[str] = []
    for entry in STRINGS.values():
        if "fil" in entry:
            phrases.append(entry["fil"])

    # 3. Sample SafeArithmetic Filipino explanations across all operations & edge cases
    sample_exprs = [
        "0 + 0", "5 + 0", "0 + 5", "5 + 1", "2 + 3", "12 + 23", "18 + 15", "120 + 25",
        "5 - 0", "5 - 5", "6 - 1", "9 - 4", "45 - 12", "42 - 17", "150 - 20", "3 - 5",
        "3 * 0", "1 * 6", "3 * 4", "8 * 9",
        "8 / 2", "7 / 2", "8 / 0",
        "2 + 3 + 4", "2 + 3 * 4", "", "5 +", "+ 5", "abc",
    ]
    for ex in sample_exprs:
        sol = SafeArithmetic.solve(ex, language="fil")
        phrases.append(sol.explanation)
        if sol.short_step:
            phrases.append(sol.short_step)

    # 4. Offline Filipino AI responses + Factual Q&A Filipino entries
    phrases.extend(FILIPINO_AI_OFFLINE_RESPONSES.values())
    for k, v in FACTUAL_QA_RESPONSES.items():
        if any(marker in k for marker in ("ano", "sino", "kailan", "ilang", "paano", "kumusta", "saang", "magandang", "salamat")):
            phrases.append(v)

    # Normalize & tokenize every phrase into individual Filipino words
    for p in phrases:
        expanded = normalize_for_sherpa_filipino(p)
        for tok in expanded.split():
            clean = re.sub(r"[^a-z0-9\-']", "", tok.lower())
            if clean:
                tokens.add(clean)

    # Additional math/scanner glue tokens
    tokens.update({
        "zero", "sero", "dagdag", "bawas", "beses", "hatiin", "katumbas", "sagot",
        "magsimula", "magbilang", "pabalik", "pangkat", "ibinahagi", "pantay", "bawat",
        "natitirang", "natira", "una", "pagkatapos", "isahan", "sampuan", "pagsamahin",
        "pangwakas", "malaking", "titik", "hindi", "malinaw", "nakitang", "braille",
        "pindutin", "basahin", "malabo", "larawan", "tuldok", "pahina", "kamera",
        "grade", "greyd", "breyl", "breyls", "ai",
    })

    return {t for t in tokens if len(t) >= 1 and any(c.isalnum() for c in t)}


async def _synthesize_edge_tts(
    text: str,
    voice: str,
    out_path: str,
    carrier_slice: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> bool:
    """Synthesize text via edge-tts, decode MP3 to PCM via miniaudio, resample to 22050 mono, standardize."""
    import edge_tts
    import miniaudio

    async def _do():
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text, voice)
                data = bytearray()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        data.extend(chunk["data"])
                if not data:
                    await asyncio.sleep(0.5)
                    continue
                decoded = miniaudio.mp3_read_s16(bytes(data))
                samples = np.frombuffer(decoded.samples, dtype=np.int16)
                if decoded.nchannels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
                num_samples = int(len(samples) * 22050 / decoded.sample_rate)
                resampled = np.interp(
                    np.linspace(0, len(samples), num_samples, endpoint=False),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.int16)

                if carrier_slice:
                    sliced = slice_carrier_audio(resampled, 22050)
                    resampled = np.asarray(sliced, dtype=np.int16)

                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(resampled.tobytes())

                standardize_wav_file(out_path)
                return True
            except Exception as exc:
                await asyncio.sleep(0.8 * (attempt + 1))
        return False

    if semaphore:
        async with semaphore:
            return await _do()
    return await _do()


def build_filipino_bank(
    engine: FilipinoEngine,
    skip_existing: bool = False,
    voice_bank: str = FILIPINO_BANK_VOICE,
    voice_keystroke: str = FILIPINO_KEYSTROKE_VOICE,
) -> int:
    """
    Build the complete Filipino audio bank using:
    - fil-PH-BlessicaNeural for words, mode intros, sentences, AI offline answers.
    - en-PH-RosaNeural with carrier-slicing for keystrokes (letters A-Z, 0-9, operators, punctuation).
    - Falls back to sherpa-onnx if edge-tts is unavailable.
    """
    os.makedirs(FIL_DIR, exist_ok=True)
    
    # 0. Delete stale sentence_<md5>.wav files from old runs
    for f in os.listdir(FIL_DIR):
        if f.startswith("sentence_") and f.endswith(".wav"):
            try:
                os.remove(os.path.join(FIL_DIR, f))
            except Exception:
                pass

    has_edge_tts = False
    try:
        import edge_tts  # noqa: F401
        import miniaudio  # noqa: F401
        has_edge_tts = True
    except ImportError:
        has_edge_tts = False

    if not has_edge_tts:
        print("edge-tts / miniaudio not available. Falling back to offline sherpa-onnx synthesis...")
        return _build_filipino_bank_sherpa(engine, skip_existing)

    print(f"Building Filipino Audio Bank via edge-tts (Bank={voice_bank}, Keys={voice_keystroke})...")
    sem = asyncio.Semaphore(12)
    tasks = []

    # 1. Fixed phrases / mode intros / announcements
    fixed_files = {
        "mode_free_input.wav": STRINGS["MODE_INTRO_FREE_INPUT"]["fil"],
        "mode_math.wav": STRINGS["MODE_INTRO_MATH"]["fil"],
        "mode_ai.wav": STRINGS["MODE_INTRO_AI"]["fil"],
        "mode_scanner.wav": STRINGS["MODE_INTRO_SCANNER"]["fil"],
        "switched_fil.wav": STRINGS["LANG_SWITCHED_FIL"]["fil"],
        "thinking_fil.wav": STRINGS["AI_THINKING"]["fil"],
        "scanning_fil.wav": STRINGS["SCANNER_SCANNING"]["fil"],
        "no_braille_fil.wav": STRINGS["SCANNER_NO_BRAILLE"]["fil"],
    }
    for fname, phrase in fixed_files.items():
        out_path = os.path.join(FIL_DIR, fname)
        if not (skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000):
            tasks.append(_synthesize_edge_tts(phrase, voice_bank, out_path, semaphore=sem))

    # Also build AI offline Q&A responses into pre-cached WAVs
    for q_key, q_resp in FILIPINO_AI_OFFLINE_RESPONSES.items():
        slug = re.sub(r"[^a-z0-9]", "_", q_key.lower()).strip("_")
        out_path = os.path.join(FIL_DIR, f"ai_{slug}.wav")
        if not (skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000):
            tasks.append(_synthesize_edge_tts(q_resp, voice_bank, out_path, semaphore=sem))

    # 2. Keystroke character files (letters, digits, math operators, punctuation)
    for ch, target in PLAIN_KEYSTROKE_TARGETS.items():
        fname = get_filipino_keystroke_filename(ch)
        out_path = os.path.join(FIL_DIR, fname)
        if not (skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000):
            carrier_text = f"Please say: {target}."
            tasks.append(_synthesize_edge_tts(carrier_text, voice_keystroke, out_path, carrier_slice=True, semaphore=sem))

    # 3. All words from manifest
    words = sorted(collect_filipino_words())
    print(f"Queueing {len(words)} Filipino word WAVs...")
    for w in words:
        slug = re.sub(r"[^a-z0-9\-]", "", w.lower().replace("'", ""))
        if not slug:
            continue
        out_path = os.path.join(FIL_DIR, f"word_{slug}.wav")
        if not (skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000):
            # Render with a trailing period for natural falling prosody
            spoken_word = f"{w}."
            tasks.append(_synthesize_edge_tts(spoken_word, voice_bank, out_path, semaphore=sem))

    async def _run_all():
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return sum(1 for r in results if r is True)

    generated_count = asyncio.run(_run_all())
    print(f"Generated/Updated {generated_count} WAVs via edge-tts.")

    # Duplicate uppercase char files to lowercase char files if distinct on filesystem
    for c in range(ord("A"), ord("Z") + 1):
        src = os.path.join(FIL_DIR, f"char_{chr(c)}.wav")
        dst = os.path.join(FIL_DIR, f"char_{chr(c).lower()}.wav")
        if os.path.isfile(src) and os.path.normcase(src) != os.path.normcase(dst):
            try:
                shutil.copy2(src, dst)
            except shutil.SameFileError:
                pass

    # Phonetic aliases for code-switched terms
    aliases = {
        "word_braille.wav": "word_breyl.wav",
        "word_brailce.wav": "word_breyls.wav",
        "word_ai.wav": "word_ey-ay.wav",
        "word_grade.wav": "word_greyd.wav",
    }
    for src_name, dst_name in aliases.items():
        src_path = os.path.join(FIL_DIR, src_name)
        dst_path = os.path.join(FIL_DIR, dst_name)
        if os.path.isfile(src_path) and os.path.normcase(src_path) != os.path.normcase(dst_path):
            try:
                shutil.copy2(src_path, dst_path)
            except shutil.SameFileError:
                pass

    # Standardize all files in FIL_DIR
    for fname in sorted(os.listdir(FIL_DIR)):
        if fname.lower().endswith(".wav"):
            standardize_wav_file(
                os.path.join(FIL_DIR, fname),
                pre_ms=DEFAULT_PRE_SILENCE_MS,
                post_ms=DEFAULT_POST_SILENCE_MS,
            )

    print("Filipino audio bank rebuild complete and standardized.")
    return 0


def _build_filipino_bank_sherpa(engine: FilipinoEngine, skip_existing: bool) -> int:
    """Offline fallback builder using sherpa-onnx MMS-TTS."""
    count = 0
    words = sorted(collect_filipino_words())
    for w in words:
        slug = re.sub(r"[^a-z0-9\-]", "", w.lower().replace("'", ""))
        if not slug:
            continue
        out_path = os.path.join(FIL_DIR, f"word_{slug}.wav")
        if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            continue
        if engine.synthesize_to_wav(w, out_path):
            count += 1
    return 0


def normalize_english_bank() -> int:
    """Standardize loudness, sample rate (22050Hz mono s16), and silence padding of existing EN WAVs."""
    count = 0
    for d in (EN_CHAR_DIR, EN_WORD_DIR):
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith(".wav"):
                if standardize_wav_file(
                    os.path.join(d, fname),
                    pre_ms=DEFAULT_PRE_SILENCE_MS,
                    post_ms=DEFAULT_POST_SILENCE_MS,
                ):
                    count += 1
    print(f"Normalized {count} existing English WAV files.")
    return 0


def build_english_bank(piper: PiperEngine) -> int:
    """Build EN character + ~1,000 word cache using Piper carrier slicing when Piper is available."""
    if not piper.is_available():
        print("Piper binary/model not installed locally; normalizing existing English WAV files only.")
        return normalize_english_bank()

    os.makedirs(EN_WORD_DIR, exist_ok=True)
    count = 0
    for word in ENGLISH_CORE_WORDS:
        out_path = os.path.join(EN_WORD_DIR, f"word_{word}.wav")
        if piper.synthesize_to_wav(word, out_path, carrier_slice=True):
            count += 1
    print(f"Built {count} English words via Piper carrier slicing.")
    normalize_english_bank()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build & standardize Brailce audio banks.")
    parser.add_argument("--filipino-only", action="store_true", help="Regenerate Filipino audio bank")
    parser.add_argument("--normalize-en-only", action="store_true", help="Normalize existing English WAVs in-place")
    parser.add_argument("--skip-existing", action="store_true", help="Only synthesize missing files")
    parser.add_argument("--all", action="store_true", help="Rebuild both Filipino and English banks")
    args = parser.parse_args()

    if not (args.filipino_only or args.normalize_en_only or args.all):
        args.all = True

    rc = 0
    if args.filipino_only or args.all:
        rc |= build_filipino_bank(FilipinoEngine(), skip_existing=args.skip_existing)
    if args.normalize_en_only:
        rc |= normalize_english_bank()
    elif args.all:
        rc |= build_english_bank(PiperEngine())
    return rc


if __name__ == "__main__":
    sys.exit(main())
