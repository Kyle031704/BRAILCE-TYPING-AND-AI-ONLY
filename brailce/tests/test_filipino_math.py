"""
Section B Tests: Filipino Math Mode, Tagalog Numbers, i18n LCD Labels, & Audio Coverage
=======================================================================================
"""

from __future__ import annotations

import os
import re
from unittest.mock import MagicMock

import pytest

from brailce.braille.arithmetic import SafeArithmetic
from brailce.core.app import BrailceApp
from brailce.core.events import EventType, InputEvent
from brailce.core.state import ApplicationState
from brailce.display.console_display import ConsoleDisplay
from brailce.i18n.strings import STRINGS, get_string
from brailce.modes.ai_assistant import FILIPINO_AI_OFFLINE_RESPONSES
from brailce.modes.math_mode import MathMode
from brailce.speech.filipino_engine import (
    get_filipino_word_path,
    normalize_for_sherpa_filipino,
    tagalog_number,
)
from brailce.speech.service import SpeechService


class TestTagalogNumberToWords:
    """Verify Tagalog 0-999 rules (labing-, -'t contraction, -ng / na linkers)."""

    @pytest.mark.parametrize(
        "num,expected",
        [
            (0, "sero"),
            (1, "isa"),
            (4, "apat"),
            (6, "anim"),
            (9, "siyam"),
            (10, "sampu"),
            (11, "labing-isa"),
            (12, "labindalawa"),
            (13, "labintatlo"),
            (14, "labing-apat"),
            (15, "labinlima"),
            (16, "labing-anim"),
            (17, "labimpito"),
            (18, "labing-walo"),
            (19, "labinsiyam"),
            (20, "dalawampu"),
            (21, "dalawampu't isa"),
            (35, "tatlumpu't lima"),
            (48, "apatnapu't walo"),
            (99, "siyamnapu't siyam"),
            (100, "isang daan"),
            (105, "isang daan at lima"),
            (114, "isang daan at labing-apat"),
            (200, "dalawang daan"),
            (345, "tatlong daan at apatnapu't lima"),
            (400, "apat na daan"),
            (512, "limang daan at labindalawa"),
            (600, "anim na daan"),
            (789, "pitong daan at walumpu't siyam"),
            (801, "walong daan at isa"),
            (900, "siyam na daan"),
            (999, "siyam na daan at siyamnapu't siyam"),
        ],
    )
    def test_tagalog_number_table(self, num: int, expected: str) -> None:
        assert tagalog_number(num) == expected


class TestSafeArithmeticFilipino:
    """Verify Grade 1 Filipino math explanations match pedagogical structure."""

    def test_filipino_counting_on_addition(self) -> None:
        sol = SafeArithmetic.solve("2 + 3", language="fil")
        assert sol.is_valid
        assert sol.result_str == "5"
        assert sol.explanation == "Ang 2 dagdag 3 ay 5. Magsimula sa 2, magbilang ng 3 pa: 3, 4, 5. Ang sagot ay 5."

    def test_filipino_counting_back_subtraction(self) -> None:
        sol = SafeArithmetic.solve("9 - 4", language="fil")
        assert sol.is_valid
        assert sol.result_str == "5"
        assert "Ang 9 bawas 4 ay 5." in sol.explanation
        assert "Magsimula sa 9, magbilang pabalik ng 4: 8, 7, 6, 5." in sol.explanation
        assert "Ang sagot ay 5." in sol.explanation

    def test_filipino_negative_subtraction_refusal(self) -> None:
        sol = SafeArithmetic.solve("3 - 5", language="fil")
        assert not sol.is_valid
        assert "hindi natin maaaring bawasan ang 3 ng 5" in sol.explanation

    def test_filipino_multiplication(self) -> None:
        sol = SafeArithmetic.solve("3 * 2", language="fil")
        assert sol.is_valid
        assert sol.result_str == "6"
        assert "Ang 3 beses 2 ay 6." in sol.explanation
        assert "3 pangkat ng 2" in sol.explanation

    def test_filipino_division_exact_and_remainder_and_zero(self) -> None:
        sol_exact = SafeArithmetic.solve("8 / 2", language="fil")
        assert sol_exact.is_valid
        assert sol_exact.result_str == "4"
        assert "Ang 8 hatiin sa 2 ay 4." in sol_exact.explanation
        assert "pantay na pangkat" in sol_exact.explanation

        sol_rem = SafeArithmetic.solve("7 / 2", language="fil")
        assert sol_rem.is_valid
        assert sol_rem.result_str == "3 R1"
        assert "natitirang 1" in sol_rem.explanation

        sol_zero = SafeArithmetic.solve("8 / 0", language="fil")
        assert not sol_zero.is_valid
        assert "Hindi maaaring hatiin sa sero" in sol_zero.explanation

    def test_filipino_chained_and_incomplete(self) -> None:
        sol_chain = SafeArithmetic.solve("2 + 3 + 4", language="fil")
        assert sol_chain.is_valid
        assert sol_chain.result_str == "9"
        assert "Ang 2 + 3 + 4 ay 9." in sol_chain.explanation

        sol_inc = SafeArithmetic.solve("5 +", language="fil")
        assert not sol_inc.is_valid
        assert "Kulang ang problema sa matematika" in sol_inc.explanation


class TestFilipinoMathLCDAndSpeechParity:
    """Verify LCD receives the exact same Filipino explanation spoken by SpeechService."""

    def test_math_mode_filipino_lcd_and_speech_use_same_string(self, capsys) -> None:
        app = MagicMock(spec=BrailceApp)
        app.state = ApplicationState()
        app.state.language = "fil"
        app.display = MagicMock(wraps=ConsoleDisplay(use_colors=False))
        app.speech = MagicMock(spec=SpeechService)
        app.encoder = MagicMock()
        app.encoder.encode_text.return_value = []
        app.config = MagicMock()
        app.config.get.return_value = True

        mode = MathMode(app)
        app.state.math_buffer = "2 + 3"
        mode.handle_event(InputEvent(EventType.ENTER_PRESSED))

        # Check show_math_result called with language="fil" and matching Filipino explanation
        app.display.show_math_result.assert_called_once()
        kwargs = app.display.show_math_result.call_args.kwargs
        assert kwargs["language"] == "fil"
        expected_exp = "Ang 2 dagdag 3 ay 5. Magsimula sa 2, magbilang ng 3 pa: 3, 4, 5. Ang sagot ay 5."
        assert kwargs["explanation"] == expected_exp

        # Check SpeechService spoken text is identical to LCD explanation
        app.speech.speak.assert_called_once_with(expected_exp)

        # Check ConsoleDisplay output uses translated i18n card labels
        out = capsys.readouterr().out
        assert get_string("MATH_CARD_TITLE", "fil") in out
        assert get_string("MATH_LABEL_EQUATION", "fil") in out
        assert get_string("MATH_LABEL_EXPLANATION", "fil") in out

    def test_hyphenated_word_cache_lookup(self) -> None:
        """labing-isa and other hyphenated Tagalog numbers must resolve cleanly to disk cache."""
        p = get_filipino_word_path("labing-isa")
        assert p is not None and os.path.isfile(p)

    def test_all_filipino_math_i18n_ai_words_have_cached_audio(self) -> None:
        """
        Scan SafeArithmetic(fil), i18n Filipino strings, and FILIPINO_AI_OFFLINE_RESPONSES
        and assert every word token resolves to a cached non-empty WAV in brailce/audio/filipino/.
        """
        phrases = []
        for entry in STRINGS.values():
            if "fil" in entry:
                phrases.append(entry["fil"])

        for ex in (
            "0 + 0", "5 + 0", "0 + 5", "5 + 1", "2 + 3", "12 + 23",
            "5 - 0", "5 - 5", "6 - 1", "9 - 4", "3 - 5",
            "3 * 0", "1 * 6", "3 * 2", "8 / 2", "7 / 2", "8 / 0", "2 + 3 + 4",
        ):
            phrases.append(SafeArithmetic.solve(ex, language="fil").explanation)

        phrases.extend(FILIPINO_AI_OFFLINE_RESPONSES.values())

        missing = []
        for phrase in phrases:
            expanded = normalize_for_sherpa_filipino(phrase)
            for tok in expanded.split():
                clean = re.sub(r"[^a-z0-9\-']", "", tok.lower())
                if not clean:
                    continue
                if get_filipino_word_path(clean) is None:
                    missing.append(clean)

        assert not missing, f"Missing cached Filipino WAVs for tokens: {sorted(set(missing))}"

