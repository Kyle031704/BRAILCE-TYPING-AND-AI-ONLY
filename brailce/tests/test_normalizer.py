"""
Brailce Unit Tests – Text Normalization and Preprocessor Filter
==============================================================

Verifies the normalization rules for:
1. Piper Neural TTS (words, sentences, math expressions, AI responses)
2. eSpeak NG (zero-latency single keystrokes)
"""

import pytest

from brailce.speech.normalizer import (
    is_keystroke_feedback,
    normalize_for_espeak_character,
    normalize_for_piper,
)


class TestPiperNormalization:
    """Tests for normalize_for_piper."""

    def test_word_normalization_on_space(self) -> None:
        """Uppercase Braille words normalized to lowercase coherent words with falling pitch."""
        assert normalize_for_piper("ARM", is_word=True) == "arm."
        assert normalize_for_piper("CAT", is_word=True) == "cat."
        assert normalize_for_piper("DOG", is_word=True) == "dog."
        assert normalize_for_piper("HELLO", is_word=True) == "hello."
        assert normalize_for_piper("WATER", is_word=True) == "water."

    def test_single_letter_words(self) -> None:
        """Single-letter English words ('A' and 'I') remain capitalized."""
        assert normalize_for_piper("A", is_word=True) == "A."
        assert normalize_for_piper("I", is_word=True) == "I."

    def test_sentence_normalization_on_enter(self) -> None:
        """All-caps sentences converted to natural sentence case with terminal punctuation."""
        assert normalize_for_piper("ARMS ARE IMPORTANT") == "Arms are important."
        assert normalize_for_piper("WHAT IS A CAT?") == "What is a cat?"
        assert normalize_for_piper("I LOVE BRAILLE") == "I love braille."

    def test_solitary_punctuation_suppression(self) -> None:
        """Piper must never say 'dot' or 'comma' when handed solitary punctuation."""
        assert normalize_for_piper(".") == ""
        assert normalize_for_piper(",") == ""
        assert normalize_for_piper("?") == ""
        assert normalize_for_piper("!") == ""
        assert normalize_for_piper("...") == ""

    def test_math_operators_to_spoken_words(self) -> None:
        """Operators +, -, *, /, = converted into natural spoken words in math context."""
        assert normalize_for_piper("5 + 3 = 8", is_math=True) == "5 plus 3 equals 8."
        assert normalize_for_piper("9 - 4", is_math=True) == "9 minus 4."
        assert normalize_for_piper("4 * 3", is_math=True) == "4 times 3."
        assert normalize_for_piper("12 / 3", is_math=True) == "12 divided by 3."
        assert normalize_for_piper("4 × 3", is_math=True) == "4 times 3."
        assert normalize_for_piper("12 ÷ 3", is_math=True) == "12 divided by 3."

    def test_math_counting_and_remainders(self) -> None:
        """Counting sequences and remainders normalized for smooth teacher voice."""
        explanation = "Start at 5, and count up 3 more: 6, 7, 8"
        norm = normalize_for_piper(explanation, is_math=True)
        assert "6, 7, 8" in norm
        assert "Start at 5" in norm

        remainder_expr = "7 / 2 = 3 R1"
        norm_r = normalize_for_piper(remainder_expr, is_math=True)
        assert "with a remainder of 1" in norm_r

    def test_markdown_and_ai_stripping(self) -> None:
        """Markdown bold, italics, headers, bullets, and <think> tags stripped."""
        raw = "**Braille** is _great_!\n\n<think>Internal thought</think>\n# Title\n* Bullet item"
        norm = normalize_for_piper(raw)
        assert "Braille is great!" in norm
        assert "<think>" not in norm
        assert "Internal thought" not in norm
        assert "Title" in norm
        assert "Bullet item" in norm

    def test_symbols_and_parentheses(self) -> None:
        """Parentheses converted to comma pauses; currency expanded."""
        raw = "Cost is $10 (ten dollars) & 50% discount."
        norm = normalize_for_piper(raw)
        assert "10 dollars" in norm
        assert "50 percent" in norm
        assert " and " in norm
        assert "open parenthesis" not in norm


class TestEspeakNormalization:
    """Tests for normalize_for_espeak_character."""

    def test_letter_names(self) -> None:
        """Letters produce clean letter names with falling terminal pitch (no artificial respellings)."""
        assert normalize_for_espeak_character("A") == "A."
        assert normalize_for_espeak_character("B") == "B."
        assert normalize_for_espeak_character("C") == "C."
        assert normalize_for_espeak_character("H") == "H."
        assert normalize_for_espeak_character("W") == "W."
        assert normalize_for_espeak_character("Z") == "Z."

    def test_spoken_letter_aliases(self) -> None:
        """Spoken letter names from legacy mappings produce standard letter sound."""
        assert normalize_for_espeak_character("Bee") == "B."
        assert normalize_for_espeak_character("Aitch") == "H."
        assert normalize_for_espeak_character("Double you") == "W."

    def test_digits(self) -> None:
        """Digits 0–9 produce word pronunciations."""
        assert normalize_for_espeak_character("0") == "zero."
        assert normalize_for_espeak_character("5") == "five."
        assert normalize_for_espeak_character("9") == "nine."

    def test_operators_and_punctuation(self) -> None:
        """Operators and punctuation marks produce proper spoken names."""
        assert normalize_for_espeak_character("+") == "plus."
        assert normalize_for_espeak_character("-") == "minus."
        assert normalize_for_espeak_character("*") == "times."
        assert normalize_for_espeak_character("/") == "divided by."
        assert normalize_for_espeak_character(".") == "period."
        assert normalize_for_espeak_character(",") == "comma."
        assert normalize_for_espeak_character("?") == "question mark."
        assert normalize_for_espeak_character("!") == "exclamation mark."
        assert normalize_for_espeak_character("#") == "number sign."


class TestKeystrokeDetection:
    """Tests for is_keystroke_feedback."""

    def test_single_characters(self) -> None:
        assert is_keystroke_feedback("A") is True
        assert is_keystroke_feedback("5") is True
        assert is_keystroke_feedback("+") is True
        assert is_keystroke_feedback(".") is True

    def test_letter_names(self) -> None:
        assert is_keystroke_feedback("Bee") is True
        assert is_keystroke_feedback("Aitch") is True
        assert is_keystroke_feedback("five") is True
        assert is_keystroke_feedback("plus") is True
        assert is_keystroke_feedback("period") is True

    def test_words_and_sentences_not_keystrokes(self) -> None:
        assert is_keystroke_feedback("ARM") is False
        assert is_keystroke_feedback("cat") is False
        assert is_keystroke_feedback("hello") is False
        assert is_keystroke_feedback("WHAT IS A CAT?") is False
