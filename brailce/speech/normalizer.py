"""
Brailce Speech Text Normalizer and Preprocessor Filter
======================================================

Provides robust text normalization for:
1. Piper Neural TTS (en_US-ryan-medium.onnx):
   - Natural words on Space (converts Braille uppercase "ARM" to "arm." to avoid acronym spelling).
   - Natural sentences on Enter (converts uppercase to sentence case, preserves cadence punctuation).
   - Math expressions (converts +, -, *, /, = into spoken words, teacher counting pauses).
   - AI responses (strips Markdown, code fences, think tags, converts symbols).
   - Prevents literal vocalization of punctuation (NEVER says "dot", "comma", "parenthesis").

2. eSpeak NG:
   - Standardized single-character keystroke mapping for A–Z, 0–9, operators, and punctuation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

# ---------------------------------------------------------------------------
# Keystroke letter names for instant eSpeak NG pronunciation
# ---------------------------------------------------------------------------

ESPEAK_LETTER_NAMES: Dict[str, str] = {
    "A": "A.",
    "B": "B.",
    "C": "C.",
    "D": "D.",
    "E": "E.",
    "F": "F.",
    "G": "G.",
    "H": "H.",
    "I": "I.",
    "J": "J.",
    "K": "K.",
    "L": "L.",
    "M": "M.",
    "N": "N.",
    "O": "O.",
    "P": "P.",
    "Q": "Q.",
    "R": "R.",
    "S": "S.",
    "T": "T.",
    "U": "U.",
    "V": "V.",
    "W": "W.",
    "X": "X.",
    "Y": "Y.",
    "Z": "Z.",
}

# Aliases used in Braille mapping files or modes
ESPEAK_NAME_TO_CHAR: Dict[str, str] = {
    "A": "A",
    "BEE": "B",
    "SEE": "C",
    "DEE": "D",
    "EE": "E",
    "EFF": "F",
    "EF": "F",
    "JEE": "G",
    "AITCH": "H",
    "EYE": "I",
    "JAY": "J",
    "KAY": "K",
    "ELL": "L",
    "EM": "M",
    "EN": "N",
    "OH": "O",
    "PEE": "P",
    "CUE": "Q",
    "ARE": "R",
    "AR": "R",
    "ESS": "S",
    "TEE": "T",
    "YOU": "U",
    "VEE": "V",
    "DOUBLE YOU": "W",
    "EX": "X",
    "WHY": "Y",
    "ZEE": "Z",
}

ESPEAK_DIGIT_NAMES: Dict[str, str] = {
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

ESPEAK_DIGIT_WORDS_TO_DIGIT: Dict[str, str] = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
}

ESPEAK_MATH_OPERATORS: Dict[str, str] = {
    "+": "plus.",
    "-": "minus.",
    "*": "times.",
    "×": "times.",
    "/": "divided by.",
    "÷": "divided by.",
    "=": "equals.",
}

ESPEAK_PUNCTUATION: Dict[str, str] = {
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

# Set of strings that represent isolated keystroke sounds (not words)
KEYSTROKE_SOUND_NAMES: Set[str] = (
    set(chr(c) for c in range(ord("A"), ord("Z") + 1))
    | set(chr(c) for c in range(ord("0"), ord("9") + 1))
    | set(ESPEAK_NAME_TO_CHAR.keys())
    | set(ESPEAK_DIGIT_WORDS_TO_DIGIT.keys())
    | {
        "PLUS", "MINUS", "TIMES", "DIVIDED BY", "EQUALS",
        "PERIOD", "COMMA", "QUESTION MARK", "EXCLAMATION MARK", "NUMBER SIGN"
    }
)


# ---------------------------------------------------------------------------
# Preprocessor Functions
# ---------------------------------------------------------------------------

def is_keystroke_feedback(text: str) -> bool:
    """
    Check if a string represents an individual keystroke (single letter, digit,
    operator, or explicit letter/symbol name) rather than a word or sentence.
    """
    if not text:
        return False
    trimmed = text.strip()
    if len(trimmed) == 1:
        return True
    upper = trimmed.upper().rstrip(".")
    if len(upper) == 1:
        return True
    if upper in KEYSTROKE_SOUND_NAMES:
        return True
    return False


def normalize_for_espeak_character(character: str) -> str:
    """
    Produce clean, unambiguous keystroke feedback for eSpeak NG.
    Returns pronunciation ending with a period for definitive falling pitch.
    """
    if not character:
        return ""
    clean = character.strip()
    if not clean:
        return ""

    upper = clean.upper().rstrip(".")

    # 1. Letters A–Z (or spoken names like 'Bee', 'Aitch')
    if upper in ESPEAK_NAME_TO_CHAR:
        char_key = ESPEAK_NAME_TO_CHAR[upper]
        return ESPEAK_LETTER_NAMES.get(char_key, f"{char_key}.")
    if len(clean) == 1 and clean.upper() in ESPEAK_LETTER_NAMES:
        return ESPEAK_LETTER_NAMES[clean.upper()]

    # 2. Digits 0–9 (or words like 'FIVE')
    if upper in ESPEAK_DIGIT_WORDS_TO_DIGIT:
        digit_key = ESPEAK_DIGIT_WORDS_TO_DIGIT[upper]
        return ESPEAK_DIGIT_NAMES.get(digit_key, f"{digit_key}.")
    if clean in ESPEAK_DIGIT_NAMES:
        return ESPEAK_DIGIT_NAMES[clean]

    # 3. Math operators
    if clean in ESPEAK_MATH_OPERATORS:
        return ESPEAK_MATH_OPERATORS[clean]
    if upper in ("PLUS", "MINUS", "TIMES", "DIVIDED BY", "EQUALS"):
        return f"{upper.lower()}."

    # 4. Punctuation
    if clean in ESPEAK_PUNCTUATION:
        return ESPEAK_PUNCTUATION[clean]
    if upper in ("PERIOD", "COMMA", "QUESTION MARK", "EXCLAMATION MARK", "NUMBER SIGN"):
        return f"{upper.lower()}."

    return f"{clean}."


def normalize_for_piper(
    text: str,
    context: str = "general",
    is_word: bool = False,
    is_math: bool = False,
) -> str:
    """
    Normalize text for Piper Neural TTS (en_US-ryan-medium.onnx).
    
    Transforms raw Braille text, math expressions, and AI tutor outputs into
    natural, human-grade conversational English without robotic artifacts.
    
    Args:
        text: Raw input string.
        context: Context hint ("general", "math", "word", "ai", "scanner").
        is_word: True if pronouncing a completed word (Space in Free Input Mode).
        is_math: True if processing a math equation or explanation.

    Returns:
        Cleaned, normalized string ready for Piper synthesis.
    """
    if not text:
        return ""

    raw = text.strip()
    if not raw:
        return ""

    # -----------------------------------------------------------------------
    # 0. Single Character / Symbol Resolution
    # -----------------------------------------------------------------------
    # Direct mappings for isolated math operators or digits
    if raw in ESPEAK_MATH_OPERATORS:
        return ESPEAK_MATH_OPERATORS[raw]
    if raw in ESPEAK_DIGIT_NAMES:
        return ESPEAK_DIGIT_NAMES[raw]
    if len(raw) == 1 and raw.isalpha():
        return f"{raw.upper()}."

    # If the entire input is solitary punctuation (e.g. "." or "," or "?"),
    # do not synthesize it as text!
    if set(raw).issubset(set(".,!?;:#'\"`~^-_/\\|@$%=+*()[]{}<>")):
        return ""

    # -----------------------------------------------------------------------
    # 1. Clean Markdown, HTML tags, and AI artifacts
    # -----------------------------------------------------------------------
    s = raw

    # Strip Ollama internal reasoning block (<think>...</think>)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)

    # Strip HTML tags
    s = re.sub(r"<[^>]+>", "", s)

    # Strip code fences and inline backticks
    s = re.sub(r"```[a-zA-Z]*\n?([\s\S]*?)```", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Strip bold / italics markers
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)

    # Strip Markdown headers: '### Header' -> 'Header'
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)

    # Strip list item bullets at line beginnings
    s = re.sub(r"^\s*[•\*\-\+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)

    # Replace blockquotes
    s = re.sub(r"^\s*>\s*", "", s, flags=re.MULTILINE)

    # -----------------------------------------------------------------------
    # 2. Math Mode Operators & Natural Teacher Pacing
    # -----------------------------------------------------------------------
    has_math = is_math or context == "math" or bool(re.search(r"\b\d+\s*[\+\-\*\/\=×÷]\s*\d+\b", s))

    if has_math:
        # Replace mathematical operator signs with spoken words
        # Use spaces around words to prevent merging
        s = s.replace("×", " times ").replace("÷", " divided by ")
        s = re.sub(r"\s*\+\s*", " plus ", s)
        # Minus: handle subtraction between numbers or spaces (e.g. 9 - 4 -> 9 minus 4)
        s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", s)
        s = re.sub(r"\s+-\s+", " minus ", s)
        s = re.sub(r"\s*\*\s*", " times ", s)
        s = re.sub(r"\s*\/\s*", " divided by ", s)
        s = re.sub(r"\s*\=\s*", " equals ", s)
        s = re.sub(r"\s*<\s*", " is less than ", s)
        s = re.sub(r"\s*>\s*", " is greater than ", s)

        # Handle remainder notation: e.g. "3 R1" or "3 R 1" -> "3 with a remainder of 1"
        s = re.sub(r"\b(\d+)\s*R\s*(\d+)\b", r"\1 with a remainder of \2", s)

    # -----------------------------------------------------------------------
    # 3. Currency, Percentage, and Symbol Expansion
    # -----------------------------------------------------------------------
    # Dollars: "$10" -> "10 dollars"
    s = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", s)
    s = s.replace("$", " dollars ")
    s = s.replace("%", " percent ")
    s = s.replace("&", " and ")
    s = s.replace("@", " at ")
    # Number sign before digit: "#5" -> "number 5"
    s = re.sub(r"#\s*(\d+)", r"number \1", s)
    s = s.replace("#", "")

    # -----------------------------------------------------------------------
    # 4. Spoken Parentheses and Structural Punctuation
    # -----------------------------------------------------------------------
    # Replace parentheses and brackets with commas for natural breathing pauses
    # (prevents Piper from saying "open parenthesis" or "close bracket")
    s = re.sub(r"[\(\[\{]", ", ", s)
    s = re.sub(r"[\)\]\}]", ", ", s)

    # Strip quotation marks (both straight and curly)
    s = re.sub(r'["“”]', '', s)
    # Strip standalone single quotes/backticks (preserve apostrophes in contractions like don't)
    s = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", "", s)

    # Replace colons and semicolons with commas for natural pauses
    s = s.replace(":", ", ").replace(";", ", ")

    # Replace em-dashes and en-dashes with comma pauses
    s = s.replace("—", ", ").replace("–", ", ")

    # Convert ellipsis (...) to a single pause/comma
    s = re.sub(r"\.{2,}", ", ", s)

    # Strip stray symbols that Piper might stumble on
    s = re.sub(r"[\^~\\\|_]", " ", s)

    # -----------------------------------------------------------------------
    # 5. Free Input Mode: Word Normalization (When SPACE is pressed)
    # -----------------------------------------------------------------------
    # When user types Braille, letters are uppercase (e.g. 'A', 'R', 'M' -> 'ARM')
    # If passed to Piper as 'ARM', it will spell it out as an acronym ('A-R-M')
    # or choppy syllables. Normalizing to lowercase ('arm.') makes Piper pronounce
    # the coherent English word with full natural duration.
    words = s.split()

    if is_word or context == "word" or len(words) == 1:
        single_word = s.strip()
        # Single-letter words: 'A' and 'I' remain capitalized in English
        if single_word.upper() in ("A", "I"):
            return f"{single_word.upper()}."
        # If the word is all uppercase (e.g. "ARM", "CAT", "HELLO", "DOG")
        if single_word.isupper() and single_word.isalpha():
            clean_word = single_word.lower()
            return f"{clean_word}."
        # If single word without sentence punctuation, append period for natural cadence
        if single_word and not single_word.endswith((".", "!", "?")):
            return f"{single_word.lower()}."
        return single_word

    # -----------------------------------------------------------------------
    # 6. Full Sentence Normalization (When ENTER is pressed)
    # -----------------------------------------------------------------------
    # If the user typed an entire sentence in uppercase (e.g. "ARMS ARE IMPORTANT"):
    # Convert to sentence case ("Arms are important.") so Piper synthesizes a
    # smooth, conversational human voice rather than shouting acronyms.
    if s.isupper():
        # Sentence case: capitalize first letter, lowercase the rest
        # Preserve capitalization of the word "I"
        lower_s = s.lower()
        # Capitalize the first letter
        sentence_cased = lower_s[:1].upper() + lower_s[1:]
        # Capitalize isolated 'i' -> 'I'
        sentence_cased = re.sub(r"\bi\b", "I", sentence_cased)
        s = sentence_cased

    # Ensure counting sequences have natural commas (e.g. "6, 7, 8")
    s = re.sub(r"(\b\d+)\s*,\s*(?=\d+\b)", r"\1, ", s)

    # Clean multiple spaces and whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Clean punctuation spacing: "word , word" -> "word, word"
    s = re.sub(r"\s+([,\.!\?])", r"\1", s)

    # Remove repeated commas: ", ," -> ","
    s = re.sub(r"(,\s*)+,", ", ", s)

    # Ensure final punctuation exists for terminal intonation
    if s and not s.endswith((".", "!", "?")):
        s += "."

    return s.strip()

