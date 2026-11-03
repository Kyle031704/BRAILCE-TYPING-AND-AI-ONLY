"""
Brailce Speech Text Normalizer and Preprocessor Filter
======================================================

Clean, standard text normalization for Piper Neural TTS (en_US-ryan-medium.onnx):
- Single letters A-Z: lowercase letter with period ("a.", "b.", ..., "z.")
- Digits 0-9: spoken words ("zero.", "one.", ..., "nine.")
- Math & Punctuation: clean English words ("plus.", "minus.", "period.", etc.)
- All-caps strings (>1 char) converted to lowercase
- Hyphens (" - ") replaced with ". " for clean spoken pauses
- Words on SPACE/ENTER: trimmed, lowercased, ending with a period (e.g., "apple.")
- Zero phonetic letter respellings
"""

from __future__ import annotations

import re
from typing import Dict, Set

# ---------------------------------------------------------------------------
# Clean English mappings (NO phonetic respellings)
# ---------------------------------------------------------------------------

LETTER_MAP: Dict[str, str] = {
    chr(c): f"{chr(c).lower()}." for c in range(ord("A"), ord("Z") + 1)
}
LETTER_MAP.update({
    chr(c): f"{chr(c)}." for c in range(ord("a"), ord("z") + 1)
})

DIGIT_MAP: Dict[str, str] = {
    "0": "zero.",
    "1": "one.",
    "2": "two.",
    "3": "three.",
    "4": "four.",
    "5": "five.",
    "6": "six.",
    "7": "seven.",
    "8": "eight.",
    "9": "nine.",
}

MATH_OPERATORS: Dict[str, str] = {
    "+": "plus.",
    "-": "minus.",
    "*": "times.",
    "×": "times.",
    "/": "divided by.",
    "÷": "divided by.",
    "=": "equals.",
}

PUNCTUATION_MAP: Dict[str, str] = {
    ".": "period.",
    ",": "comma.",
    "?": "question mark.",
    "!": "exclamation mark.",
    "#": "number sign.",
    "'": "apostrophe.",
    '"': "quote.",
    ":": "colon.",
    ";": "semicolon.",
}

CHARACTER_FILENAME_MAP: Dict[str, str] = {
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
}


def is_keystroke_feedback(text: str) -> bool:
    """Check if a string represents an individual keystroke (single letter, digit, operator, symbol)."""
    if not text:
        return False
    trimmed = text.strip()
    if len(trimmed) == 1:
        return True
    upper = trimmed.upper().rstrip(".")
    if len(upper) == 1:
        return True
    if upper in (
        "PLUS", "MINUS", "TIMES", "DIVIDED BY", "EQUALS",
        "PERIOD", "COMMA", "QUESTION MARK", "EXCLAMATION MARK", "NUMBER SIGN"
    ):
        return True
    return False


def normalize_for_espeak_character(character: str) -> str:
    """Legacy helper maintained for backward compatibility. Returns clean token."""
    return normalize_for_piper(character, is_character=True)


def normalize_for_piper(
    text: str,
    context: str = "general",
    is_word: bool = False,
    is_math: bool = False,
    is_character: bool = False,
) -> str:
    """
    Normalize text for Piper Neural TTS (en_US-ryan-medium.onnx).

    Rules:
    - Single letters A-Z -> "a.", "b.", ..., "z."
    - Digits 0-9 -> "zero.", "one.", ..., "nine."
    - Math & Punctuation -> "plus.", "minus.", "period.", etc.
    - All-caps strings (>1 char) converted to lowercase
    - Hyphens (" - ") replaced with ". " for clean pauses
    - Single words on SPACE -> trimmed, lowercase with period (e.g., "apple.")
    - Sentences on ENTER -> trimmed, lowercase sentence with final punctuation.
    """
    if not text:
        return ""

    raw = text.strip()
    if not raw:
        return ""

    # 1. Single Characters (letters, digits, math operators, punctuation)
    if is_character or len(raw) == 1:
        if raw in LETTER_MAP:
            return LETTER_MAP[raw]
        if raw in DIGIT_MAP:
            return DIGIT_MAP[raw]
        if raw in MATH_OPERATORS:
            return MATH_OPERATORS[raw]
        if raw in PUNCTUATION_MAP:
            return PUNCTUATION_MAP[raw]

    # Explicit symbol word checks (e.g. "+", "PLUS", "period")
    upper_raw = raw.upper().rstrip(".")
    if raw in MATH_OPERATORS:
        return MATH_OPERATORS[raw]
    if raw in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[raw]
    if upper_raw in ("PLUS", "MINUS", "TIMES", "EQUALS"):
        return f"{upper_raw.lower()}."
    if upper_raw == "DIVIDED BY":
        return "divided by."
    if upper_raw in ("PERIOD", "COMMA", "QUESTION MARK", "EXCLAMATION MARK", "NUMBER SIGN"):
        return f"{upper_raw.lower()}."

    # If the entire input is solitary raw punctuation characters, return its clean spoken name
    if len(raw) <= 2 and all(c in PUNCTUATION_MAP for c in raw):
        return PUNCTUATION_MAP.get(raw[0], "")

    # 2. Markdown, HTML, and reasoning tag stripping
    s = raw
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"```[a-zA-Z]*\n?([\s\S]*?)```", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*[•\*\-\+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*>\s*", "", s, flags=re.MULTILINE)

    # 3. Math Mode conversions
    has_math = is_math or context == "math" or bool(re.search(r"\b\d+\s*[\+\-\*\/\=×÷]\s*\d+\b", s))
    if has_math:
        s = s.replace("×", " times ").replace("÷", " divided by ")
        s = re.sub(r"\s*\+\s*", " plus ", s)
        s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", s)
        s = re.sub(r"\s+-\s+", " minus ", s)
        s = re.sub(r"\s*\*\s*", " times ", s)
        s = re.sub(r"\s*\/\s*", " divided by ", s)
        s = re.sub(r"\s*\=\s*", " equals ", s)
        s = re.sub(r"\s*<\s*", " is less than ", s)
        s = re.sub(r"\s*>\s*", " is greater than ", s)
        s = re.sub(r"\b(\d+)\s*R\s*(\d+)\b", r"\1 with a remainder of \2", s)
    else:
        # Replace hyphens with clean sentence pauses (" - " -> ". ")
        s = s.replace(" - ", ". ").replace(" – ", ". ").replace(" — ", ". ")

    # Currency, percentages, parentheses
    s = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", s)
    s = s.replace("$", " dollars ")
    s = s.replace("%", " percent ")
    s = s.replace("&", " and ")
    s = s.replace("@", " at ")
    s = re.sub(r"#\s*(\d+)", r"number \1", s)
    s = s.replace("#", " number sign ")
    s = re.sub(r"[\(\[\{]", ", ", s)
    s = re.sub(r"[\)\]\}]", ", ", s)
    s = re.sub(r'["“”]', '', s)
    s = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", "", s)
    s = s.replace(":", ", ").replace(";", ", ")
    s = re.sub(r"\.{2,}", ", ", s)
    s = re.sub(r"[\^~\\\|_]", " ", s)

    # 5. Convert ALL-CAPS strings (longer than 1 character) to lowercase
    # Example: "NO BRAILLE DETECTED. ADJUST SHEET" -> "no braille detected. adjust sheet"
    if len(s) > 1 and s.isupper():
        s = s.lower()

    # 6. Words on SPACE: strip whitespace, lowercase, end with period (e.g. "apple.")
    words = s.split()
    if is_word or context == "word" or len(words) == 1:
        clean_word = s.strip().lower()
        if clean_word and not clean_word.endswith((".", "!", "?")):
            clean_word += "."
        return clean_word

    # 7. Sentences: ensure first letter capitalized, end with terminal punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,\.!\?])", r"\1", s)
    s = re.sub(r"(,\s*)+,", ", ", s)

    # Capitalize first letter of sentence
    if s:
        s = s[:1].upper() + s[1:]
        s = re.sub(r"\bi\b", "I", s)

    if s and not s.endswith((".", "!", "?")):
        s += "."

    return s.strip()
