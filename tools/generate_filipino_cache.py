"""
Generate 100% Offline Grade 1 Filipino Audio Bank
=================================================

Delegates directly to tools/build_audio_bank.py (--filipino-only) using
sherpa-onnx MMS-TTS tgl VITS so the entire Filipino audio bank comes from
one consistent offline voice and stays synchronized with i18n, math, and AI.
"""

import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from build_audio_bank import build_filipino_bank
from brailce.speech.filipino_engine import FilipinoEngine


def main() -> int:
    return build_filipino_bank(FilipinoEngine())


if __name__ == "__main__":
    sys.exit(main())
