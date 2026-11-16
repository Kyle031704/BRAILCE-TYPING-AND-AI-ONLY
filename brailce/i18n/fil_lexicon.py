"""
Filipino Pronunciation Lexicon (Brailce i18n)
=============================================

Maps English loanwords, technical terms, and acronyms to phonetic spellings
tailored for natural Filipino neural TTS synthesis.

The LCD displays the authentic written spelling (e.g., "Enter", "AI", "Braille"),
while `apply_filipino_lexicon()` translates text into the exact phonetics
spoken by the Filipino voice.
"""

from __future__ import annotations

import re
from typing import Dict, Set

# Surface form (lowercase) -> spoken phonetic replacement for Filipino TTS
FILIPINO_PRONUNCIATION_LEXICON: Dict[str, str] = {
    "brailce": "breyls",
    "braille": "breyl",
    "ai": "ey-ay",
    "enter": "enter",
    "space": "ispeys",
    "backspace": "bak-ispeys",
    "tab": "tab",
    "scanner": "iskaner",
    "mode": "mowd",
    "usb": "yu-es-bi",
    "lcd": "el-si-di",
    "led": "el-i-di",
    "wifi": "way-fay",
    "grade": "greyd",
    "type": "tayp",
    "ok": "okay",
    "camera": "kamera",
}

# Approved technical / universal terms permitted in Filipino strings
APPROVED_FILIPINO_LOANWORDS: Set[str] = {
    "enter", "ai", "braille", "brailce", "grade", "ok", "usb", "lcd", "led",
    "space", "backspace", "tab", "scanner", "mode", "wifi", "camera",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
}


def apply_filipino_lexicon(text: str) -> str:
    """
    Transform surface written Filipino text into TTS-friendly pronunciation.
    Preserves punctuation and casing where appropriate.
    """
    if not text:
        return ""

    result = text
    # Replace longer multi-word phrases first
    for surface, spoken in sorted(FILIPINO_PRONUNCIATION_LEXICON.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(surface) + r"\b"
        result = re.sub(pattern, spoken, result, flags=re.IGNORECASE)

    return result

