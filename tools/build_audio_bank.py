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
  - Filipino: Offline sherpa-onnx MMS-TTS tgl VITS (brailce/models/filipino/model.onnx)

All generated/updated WAV files are standardized via standardize_wav_file():
  22050 Hz, mono (1 ch), 16-bit PCM, active RMS ~4200, 120ms pre / 180ms post silence.

Usage:
  python tools/build_audio_bank.py --filipino-only
  python tools/build_audio_bank.py --all
  python tools/build_audio_bank.py --normalize-en-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Set

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

# Filipino character pronunciations (English terms with natural Filipino accent)
FILIPINO_CHAR_TEXT: Dict[str, str] = {
    **FILIPINO_KEY_LETTERS_ENGLISH,
    **FILIPINO_KEY_DIGITS,
    **FILIPINO_KEY_OPERATORS,
    **FILIPINO_KEY_PUNCTUATION,
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
    })

    return {t for t in tokens if len(t) >= 1 and any(c.isalnum() for c in t)}


def build_filipino_bank(engine: FilipinoEngine, skip_existing: bool = False) -> int:
    """Regenerate all Filipino WAVs from the single sherpa-onnx MMS-TTS tgl voice."""
    os.makedirs(FIL_DIR, exist_ok=True)
    if engine._get_sherpa_tts() is None:
        print("ERROR: sherpa_onnx Filipino model not found at brailce/models/filipino/model.onnx", file=sys.stderr)
        return 1

    count = 0
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
        if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            continue
        if engine.synthesize_to_wav(phrase, out_path):
            count += 1

    # Handle switched_en.wav cleanly if Piper is absent (preserve/standardize existing)
    switched_en_path = os.path.join(FIL_DIR, "switched_en.wav")
    if os.path.isfile(switched_en_path):
        standardize_wav_file(switched_en_path, pre_ms=DEFAULT_PRE_SILENCE_MS, post_ms=DEFAULT_POST_SILENCE_MS)

    # 2. Filipino characters/digits/symbols
    for ch, spoken in FILIPINO_CHAR_TEXT.items():
        fname = get_filipino_keystroke_filename(ch)
        out_path = os.path.join(FIL_DIR, fname)
        if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            continue
        if engine.synthesize_to_wav(spoken, out_path):
            count += 1

    # 3. Every Filipino word from manifest
    words = sorted(collect_filipino_words())
    print(f"Building {len(words)} Filipino words into {FIL_DIR} (skip_existing={skip_existing})...")
    for w in words:
        slug = re.sub(r"[^a-z0-9\-]", "", w.lower().replace("'", ""))
        if not slug:
            continue
        out_path = os.path.join(FIL_DIR, f"word_{slug}.wav")
        if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            continue
        if engine.synthesize_to_wav(w, out_path):
            count += 1

    # Ensure ALL WAVs in FIL_DIR are standardized (removes any stale unnormalized files)
    for fname in sorted(os.listdir(FIL_DIR)):
        if fname.lower().endswith(".wav"):
            standardize_wav_file(
                os.path.join(FIL_DIR, fname),
                pre_ms=DEFAULT_PRE_SILENCE_MS,
                post_ms=DEFAULT_POST_SILENCE_MS,
            )

    print(f"Filipino audio bank ready: {count} generated/updated.")
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
    parser.add_argument("--filipino-only", action="store_true", help="Regenerate Filipino audio bank via sherpa-onnx")
    parser.add_argument("--normalize-en-only", action="store_true", help="Normalize existing English WAVs in-place")
    parser.add_argument("--skip-existing", action="store_true", help="Only synthesize missing Filipino files")
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
