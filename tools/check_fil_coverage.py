"""
Check Filipino Audio Coverage
==============================

Identifies which words typed or spoken in Filipino Free Input, Math, AI,
and Scanner modes are missing from `brailce/audio/filipino/word_<word>.wav`.

Checked vocabulary sources:
  1. English common/function words typed in Free Input:
     "is", "his", "this", "the", "was", "what", "when", "where", "why", "who", "which",
     "it", "its", "are", "you", "they", "we", "can", "do", "have", "for", "with", "not",
     "on", "at", "by", "from", "as", "be", "to", "of", "and", "in", "that", "so", "up",
     "out", "if", "or", "an", "my", "one", "all", "there", "their", "get", "go", "me",
     "cat", "dog", "book", "page", "read", "braille", "help"
  2. All Filipino words from `brailce/i18n/strings.py`
  3. SafeArithmetic explanation words (`SafeArithmetic.solve()`)
  4. AI offline Q&A responses (`FILIPINO_AI_OFFLINE_RESPONSES` and `FACTUAL_QA_RESPONSES`)
  5. Approved loanwords from `brailce/i18n/fil_lexicon.py`
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from brailce.ai.ollama_client import FACTUAL_QA_RESPONSES
from brailce.braille.arithmetic import SafeArithmetic
from brailce.i18n.fil_lexicon import APPROVED_FILIPINO_LOANWORDS, FILIPINO_PRONUNCIATION_LEXICON
from brailce.i18n.strings import STRINGS
from brailce.modes.ai_assistant import FILIPINO_AI_OFFLINE_RESPONSES
from brailce.speech.filipino_engine import (
    FILIPINO_AUDIO_DIR,
    TAGALOG_HUNDREDS,
    TAGALOG_ONES,
    TAGALOG_TEENS,
    TAGALOG_TENS,
    get_filipino_word_path,
    normalize_for_sherpa_filipino,
)

ENGLISH_FREE_INPUT_WORDS = [
    "is", "his", "this", "the", "was", "what", "when", "where", "why", "who", "which",
    "it", "its", "are", "you", "they", "we", "can", "do", "have", "for", "with", "not",
    "on", "at", "by", "from", "as", "be", "to", "of", "and", "in", "that", "so", "up",
    "out", "if", "or", "an", "my", "one", "all", "there", "their", "get", "go", "me",
    "cat", "dog", "book", "page", "read", "braille", "brailce", "help",
]


def collect_target_vocabulary() -> Set[str]:
    vocab: Set[str] = set()

    # 1. English free input words
    for w in ENGLISH_FREE_INPUT_WORDS:
        vocab.add(w.lower().strip())

    # 2. Approved loanwords and lexicon surface forms
    for w in APPROVED_FILIPINO_LOANWORDS:
        vocab.add(w.lower().strip())
    for w in FILIPINO_PRONUNCIATION_LEXICON.keys():
        vocab.add(w.lower().strip())

    # 3. Number words 0-999
    for v in TAGALOG_ONES.values():
        vocab.add(v)
    for v in TAGALOG_TEENS.values():
        vocab.add(v)
    for v in TAGALOG_TENS.values():
        vocab.add(v)
        for o_idx, o_val in TAGALOG_ONES.items():
            if o_idx > 0:
                vocab.add(f"{v}'t")
                vocab.add(o_val)
    for v in TAGALOG_HUNDREDS.values():
        for w in v.split():
            vocab.add(w)

    # 4. Strings from STRINGS
    phrases: List[str] = []
    for entry in STRINGS.values():
        if "fil" in entry:
            phrases.append(entry["fil"])

    # 5. Arithmetic explanations
    for ex in (
        "0 + 0", "5 + 0", "0 + 5", "5 + 1", "2 + 3", "12 + 23", "18 + 15", "120 + 25",
        "5 - 0", "5 - 5", "6 - 1", "9 - 4", "45 - 12", "42 - 17", "150 - 20", "3 - 5",
        "3 * 0", "1 * 6", "3 * 4", "8 * 9", "8 / 2", "7 / 2", "8 / 0", "2 + 3 + 4",
    ):
        sol = SafeArithmetic.solve(ex, language="fil")
        phrases.append(sol.explanation)
        if sol.short_step:
            phrases.append(sol.short_step)

    # 6. AI responses
    phrases.extend(FILIPINO_AI_OFFLINE_RESPONSES.values())
    for k, v in FACTUAL_QA_RESPONSES.items():
        if any(marker in k for marker in ("ano", "sino", "kailan", "ilang", "paano", "kumusta", "saang", "magandang", "salamat")):
            phrases.append(v)

    for p in phrases:
        norm = normalize_for_sherpa_filipino(p)
        for tok in norm.split():
            clean = re.sub(r"[^a-z0-9\-']", "", tok.lower())
            if clean and any(c.isalnum() for c in clean):
                vocab.add(clean)

    return vocab


def check_coverage() -> Tuple[List[str], List[str]]:
    vocab = sorted(collect_target_vocabulary())
    covered: List[str] = []
    missing: List[str] = []

    for word in vocab:
        p = get_filipino_word_path(word)
        if p and os.path.isfile(p) and os.path.getsize(p) > 500:
            covered.append(word)
        else:
            missing.append(word)

    return covered, missing


def main() -> int:
    print("Checking Filipino audio bank coverage across all system vocabulary...")
    covered, missing = check_coverage()
    print(f"\nTotal vocabulary target tokens: {len(covered) + len(missing)}")
    print(f"Covered with cached WAV: {len(covered)}")
    print(f"Missing from audio bank: {len(missing)}")

    if missing:
        print("\nMISSING WORDS (falling through to MMS-TTS fallback at runtime):")
        for m in missing:
            print(f"  - {m}")
        return 1
    else:
        print("\nAll target words are 100% covered in the Filipino audio bank!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
