"""
Filipino Keystroke Table (Brailce i18n)
=======================================

Single source of truth for Filipino spoken keystroke names (letters, digits,
math operators, and punctuation).

Supports:
- "english_names" (default): Standard modern Filipino classroom letter names in
  Tagalog orthography (ey, bi, si, di, ...), spoken with a natural Filipino accent.
- "abakada": Traditional Abakada letter names (ah, ba, ka, da, ...).
- Digits 0-9: Tagalog counting names (sero, isa, dalawa, ...).
- Math operators: dagdag (+), bawas (-), beses (*), hatiin (/), katumbas (=).
- Punctuation: tuldok (.), kuwit (,), tandang pananong (?), tandang padamdam (!),
  senyas ng numero (#).
"""

from __future__ import annotations

from typing import Dict, Optional

# Standard English-derived letter names in Tagalog phonetic orthography
FILIPINO_KEY_LETTERS_ENGLISH: Dict[str, str] = {
    "A": "ey",
    "B": "bi",
    "C": "si",
    "D": "di",
    "E": "i",
    "F": "ef",
    "G": "dyi",
    "H": "eych",
    "I": "ay",
    "J": "dyey",
    "K": "key",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "o",
    "P": "pi",
    "Q": "kyu",
    "R": "ar",
    "S": "es",
    "T": "ti",
    "U": "yu",
    "V": "vi",
    "W": "dobolyu",
    "X": "eks",
    "Y": "way",
    "Z": "zi",
}

# Traditional Abakada letter sounds
FILIPINO_KEY_LETTERS_ABAKADA: Dict[str, str] = {
    "A": "ah",
    "B": "ba",
    "C": "ka",
    "D": "da",
    "E": "eh",
    "F": "fa",
    "G": "ga",
    "H": "ha",
    "I": "ee",
    "J": "dya",
    "K": "ka",
    "L": "la",
    "M": "ma",
    "N": "na",
    "O": "oh",
    "P": "pa",
    "Q": "kwa",
    "R": "ra",
    "S": "sa",
    "T": "ta",
    "U": "oo",
    "V": "va",
    "W": "wa",
    "X": "ekis",
    "Y": "ya",
    "Z": "za",
}

FILIPINO_KEY_DIGITS: Dict[str, str] = {
    "0": "sero",
    "1": "isa",
    "2": "dalawa",
    "3": "tatlo",
    "4": "apat",
    "5": "lima",
    "6": "anim",
    "7": "pito",
    "8": "walo",
    "9": "siyam",
}

FILIPINO_KEY_OPERATORS: Dict[str, str] = {
    "+": "dagdag",
    "-": "bawas",
    "*": "beses",
    "/": "hatiin",
    "=": "katumbas",
}

FILIPINO_KEY_PUNCTUATION: Dict[str, str] = {
    ".": "tuldok",
    ",": "kuwit",
    "?": "tandang pananong",
    "!": "tandang padamdam",
    "#": "senyas ng numero",
    " ": "espasyo",
}

KEYSTROKE_FILENAME_MAP: Dict[str, str] = {
    "+": "char_plus.wav",
    "-": "char_minus.wav",
    "*": "char_times.wav",
    "/": "char_divide.wav",
    "=": "char_equals.wav",
    ".": "char_period.wav",
    ",": "char_comma.wav",
    "?": "char_question.wav",
    "!": "char_exclamation.wav",
    "#": "char_hash.wav",
    " ": "char_space.wav",
}


def get_filipino_keystroke_name(char: str, style: str = "english_names") -> str:
    """Return the natural spoken Filipino keystroke name for any character."""
    if not char:
        return ""
    c = char.strip()
    if not c:
        return FILIPINO_KEY_PUNCTUATION.get(" ", "espasyo")

    cu = c.upper()
    if cu in FILIPINO_KEY_LETTERS_ENGLISH:
        if (style or "").lower() == "abakada":
            return FILIPINO_KEY_LETTERS_ABAKADA.get(cu, cu.lower())
        return FILIPINO_KEY_LETTERS_ENGLISH.get(cu, cu.lower())

    if c in FILIPINO_KEY_DIGITS:
        return FILIPINO_KEY_DIGITS[c]

    if c in FILIPINO_KEY_OPERATORS:
        return FILIPINO_KEY_OPERATORS[c]

    if c in FILIPINO_KEY_PUNCTUATION:
        return FILIPINO_KEY_PUNCTUATION[c]

    return c


def get_filipino_keystroke_filename(char: str) -> str:
    """Return the WAV filename for a given character in the Filipino audio bank."""
    if not char or char == " ":
        return "char_space.wav"
    c = char.strip()
    if c in KEYSTROKE_FILENAME_MAP:
        return KEYSTROKE_FILENAME_MAP[c]
    cu = c.upper()
    if cu.isalpha() or cu.isdigit():
        return f"char_{cu}.wav"
    return f"char_{ord(c)}.wav"


def get_all_filipino_keystroke_items(style: str = "english_names") -> Dict[str, Tuple[str, str]]:
    """
    Return all supported keystrokes as a dict:
    char -> (spoken_name, filename.wav)
    """
    items: Dict[str, Tuple[str, str]] = {}
    # Letters A-Z
    for code in range(ord("A"), ord("Z") + 1):
        ch = chr(code)
        items[ch] = (get_filipino_keystroke_name(ch, style=style), get_filipino_keystroke_filename(ch))
    # Digits 0-9
    for d in range(10):
        ch = str(d)
        items[ch] = (get_filipino_keystroke_name(ch, style=style), get_filipino_keystroke_filename(ch))
    # Operators
    for op in ("+", "-", "*", "/", "="):
        items[op] = (get_filipino_keystroke_name(op, style=style), get_filipino_keystroke_filename(op))
    # Punctuation
    for p in (".", ",", "?", "!", "#"):
        items[p] = (get_filipino_keystroke_name(p, style=style), get_filipino_keystroke_filename(p))
    return items

