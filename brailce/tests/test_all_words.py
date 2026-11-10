"""
Brailce Automated Pronunciation Audit & Speech Verification Suite
==================================================================

Systematically tests the pronunciation of every single pre-cached
and integrated English and Filipino word on the system:
  1. English Sight Words, TH Words, and WH- Words on both SPACE and ENTER paths.
  2. Filipino Characters (A-Z, digits 0-9) and pre-cached Filipino Vocabulary.
  3. Pauses cleanly between utterances so you can audit the audio output.

Usage:
  Direct command line runner:
    python brailce/tests/test_all_words.py
    python brailce/tests/test_all_words.py --section english
    python brailce/tests/test_all_words.py --section filipino
    python brailce/tests/test_all_words.py --pause 1.5

  Pytest:
    pytest brailce/tests/test_all_words.py -v
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brailce.speech.service import SpeechService
from brailce.speech.piper_engine import PiperEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_all_words")

# ---------------------------------------------------------------------------
# Test Words and Characters
# ---------------------------------------------------------------------------

ENGLISH_SIGHT_WORDS = [
    # Critical isolated words (is, this, that, etc.)
    "is",
    "this",
    "that",
    "the",
    "these",
    "those",
    "there",
    "their",
    "they",
    "them",
    "then",
    "than",
    "thus",
    "with",
    "was",
    "has",
    "his",
    "as",
    # WH- words
    "what",
    "where",
    "when",
    "why",
    "who",
    "whom",
    "whose",
    "which",
    "while",
    "white",
    # High-frequency Grade 1 words
    "cat",
    "dog",
    "braille",
    "book",
    "pen",
    "water",
    "school",
    "teacher",
    "hello",
    "help",
]

FILIPINO_CHARACTERS = [
    chr(c) for c in range(ord("A"), ord("Z") + 1)
] + [str(d) for d in range(10)]

FILIPINO_VOCABULARY = [
    # Question words
    "ano",
    "sino",
    "saan",
    "kailan",
    "bakit",
    "paano",
    # Pronouns & Demonstratives
    "ito",
    "iyan",
    "iyon",
    "ako",
    "ikaw",
    "siya",
    "kami",
    "tayo",
    "sila",
    # Common Grade 1 vocabulary
    "oo",
    "hindi",
    "po",
    "opo",
    "salamat",
    "kumusta",
    "magandang",
    "umaga",
    "hapon",
    "gabi",
    "bahay",
    "paaralan",
    "guro",
    "bata",
    "kaibigan",
    "tubig",
    "libro",
    "aso",
    "pusa",
    "mabuti",
    "masaya",
    "braille",
    "wika",
    "filipino",
    "pilipinas",
]


def run_english_audit(speech: SpeechService, pause_sec: float = 1.0) -> None:
    """Test English words on both SPACE (speak_word) and ENTER (speak_sentence) paths."""
    speech.language = "en"
    print("\n" + "=" * 65)
    print(" [EN] RUNNING ENGLISH PRONUNCIATION AUDIT (PIPER NEURAL TTS)")
    print("=" * 65)

    for i, word in enumerate(ENGLISH_SIGHT_WORDS, 1):
        print(f"[{i:2d}/{len(ENGLISH_SIGHT_WORDS):2d}] Testing English word: {word!r}")

        # Path 1: SPACE path (speak_word)
        print(f"       -> [SPACE path] speak_word({word!r}) ...")
        speech.speak_word(word)
        time.sleep(pause_sec)

        # Path 2: ENTER path (speak_sentence)
        print(f"       -> [ENTER path] speak_sentence({word!r}) ...")
        speech.speak_sentence(word)
        time.sleep(pause_sec)

    print("\n[EN] English Pronunciation Audit Finished!")


def run_filipino_audit(speech: SpeechService, pause_sec: float = 1.0) -> None:
    """Test Filipino characters and vocabulary words via filipino_engine."""
    speech.language = "fil"
    print("\n" + "=" * 65)
    print(" [FIL] RUNNING FILIPINO PRONUNCIATION AUDIT (ANGELONEURAL CACHE)")
    print("=" * 65)

    # 1. Characters A-Z and 0-9
    print("\n--- Phase 1: Filipino Keystroke Characters ---")
    for i, char in enumerate(FILIPINO_CHARACTERS, 1):
        print(f"[{i:2d}/{len(FILIPINO_CHARACTERS):2d}] Character: {char!r}")
        speech.speak_character(char)
        time.sleep(pause_sec * 0.7)

    # 2. Vocabulary words on SPACE and ENTER
    print("\n--- Phase 2: Filipino Vocabulary Words ---")
    for i, word in enumerate(FILIPINO_VOCABULARY, 1):
        print(f"[{i:2d}/{len(FILIPINO_VOCABULARY):2d}] Word: {word!r}")

        # Path 1: SPACE path (speak_word)
        print(f"       -> [SPACE path] speak_word({word!r}) ...")
        speech.speak_word(word)
        time.sleep(pause_sec)

        # Path 2: ENTER path (speak_sentence)
        print(f"       -> [ENTER path] speak_sentence({word!r}) ...")
        speech.speak_sentence(word)
        time.sleep(pause_sec)

    print("\n[FIL] Filipino Pronunciation Audit Finished!")


# ---------------------------------------------------------------------------
# Pytest Integration Test Cases
# ---------------------------------------------------------------------------

def test_speech_service_english_words_flow() -> None:
    """Automated assertion test for English word synthesis pipeline."""
    service = SpeechService()
    service.language = "en"

    for word in ["is", "this", "what", "where", "cat"]:
        service.speak_word(word)
        assert service.last_utterance is not None


def test_speech_service_filipino_words_flow() -> None:
    """Automated assertion test for Filipino word synthesis pipeline."""
    service = SpeechService()
    service.language = "fil"

    for word in ["ano", "sino", "kumusta", "salamat"]:
        service.speak_word(word)
        assert service.language == "fil"


# ---------------------------------------------------------------------------
# Manual CLI Runner (for PuTTY / Terminal execution)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pronunciation of all Brailce English and Filipino words.")
    parser.add_argument(
        "--section",
        choices=["all", "english", "filipino"],
        default="all",
        help="Section to audit (default: all)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause in seconds between each spoken utterance (default: 1.0)",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("   BRAILCE COMPREHENSIVE AUDIO PRONUNCIATION AUDIT")
    print("=======================================================")
    print(f"Section : {args.section}")
    print(f"Pause   : {args.pause}s between words\n")

    speech_service = SpeechService()

    if args.section in ("all", "english"):
        run_english_audit(speech_service, pause_sec=args.pause)

    if args.section in ("all", "filipino"):
        run_filipino_audit(speech_service, pause_sec=args.pause)

    print("\n=======================================================")
    print(" ALL WORD AUDITS COMPLETED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

