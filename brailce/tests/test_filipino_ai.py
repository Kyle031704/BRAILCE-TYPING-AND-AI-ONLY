"""
Section C Tests: Filipino AI Assistant (Language Reliability, Retry, & TTS Path Parity)
======================================================================================
"""

from __future__ import annotations

import json
import os
import re
from unittest.mock import MagicMock, patch

import pytest

from brailce.ai.ollama_client import (
    FACTUAL_QA_RESPONSES,
    FILIPINO_SYSTEM_PROMPT,
    OllamaClient,
    clean_ai_response,
    is_mostly_english,
)
from brailce.core.app import BrailceApp
from brailce.core.state import ApplicationState
from brailce.i18n.strings import get_string
from brailce.modes.ai_assistant import AIAssistantMode, FILIPINO_AI_OFFLINE_RESPONSES
from brailce.speech.filipino_engine import (
    FilipinoEngine,
    get_filipino_word_path,
    normalize_for_sherpa_filipino,
)
from brailce.speech.service import SpeechService

FILIPINO_TEST_QUESTIONS = [
    "Kumusta?",
    "Kumusta ka?",
    "Ano ang pangalan mo?",
    "Sino ka?",
    "Magandang umaga",
    "Magandang hapon",
    "Magandang gabi",
    "Salamat",
    "Ano ang Braille?",
    "Sino ang gumawa ng Braille?",
    "Sino nag-imbento ng Braille?",
    "Sino si Louis Braille?",
    "Kailan naimbento ang Braille?",
    "Ano ang Grade 1 Braille?",
    "Ilang tuldok mayroon sa isang Braille cell?",
    "Ano ang kabisera ng Pilipinas?",
    "Ano ang pinakamalaking planeta?",
    "Ano ang pinakamalapit na bituin sa mundo?",
    "Saang planeta tayo nakatira?",
    "Ilang araw mayroon sa isang taon?",
    "Ilang oras mayroon sa isang araw?",
    "Ano ang pambansang ibon ng Pilipinas?",
    "Ano ang pambansang bulaklak ng Pilipinas?",
    "Ano ang pambansang bayani ng Pilipinas?",
    "Ano ang buwan?",
    "Ano ang araw?",
    "Ano ang tubig?",
    "Ano ang Brailce?",
    "Paano magbasa ng Braille?",
    "Ilan ang titik sa alpabetong Filipino?",
    "Ano ang 2 + 3?",
]


class TestFilipinoAIAssistant:
    """Tests for Section C: pure Filipino replies, retry/fallback, and LCD/speech parity."""

    def test_system_prompt_contains_few_shot_filipino_examples(self) -> None:
        assert "Mga Halimbawa:" in FILIPINO_SYSTEM_PROMPT
        assert "Tanong:" in FILIPINO_SYSTEM_PROMPT
        assert "Sagot:" in FILIPINO_SYSTEM_PROMPT

    def test_clean_ai_response_strips_markdown_emoji_and_prefixes(self) -> None:
        raw = "Sagot: **Ang Braille** ay mahalaga! 😊 Makakatulong ito sa pagbabasa. Ito ay sobrang galing."
        cleaned = clean_ai_response(raw)
        assert "Sagot:" not in cleaned
        assert "**" not in cleaned
        assert "😊" not in cleaned
        # At most 2 sentences retained
        assert cleaned == "Ang Braille ay mahalaga! Makakatulong ito sa pagbabasa."

    @pytest.mark.parametrize("question", FILIPINO_TEST_QUESTIONS)
    def test_31_filipino_questions_are_filipino_and_speakable(self, question: str) -> None:
        client = OllamaClient()
        reply = client.generate(question, language="fil")
        assert reply, f"Empty reply for {question!r}"
        assert not is_mostly_english(reply), f"Reply for {question!r} looks English: {reply!r}"

        # Assert every token after digit expansion is resolvable in Filipino cache
        expanded = normalize_for_sherpa_filipino(reply)
        assert expanded
        missing = []
        for tok in expanded.split():
            clean = re.sub(r"[^a-z0-9\-']", "", tok.lower())
            if clean and get_filipino_word_path(clean) is None:
                missing.append(clean)
        assert not missing, f"Unspeakable uncached tokens for {question!r}: {missing}"

    def test_mocked_taglish_reply_retries_and_recovers_filipino(self) -> None:
        """If Ollama answers in English/Taglish on first try, retry once with strict prompt."""
        client = OllamaClient()
        with patch.object(client, "resolve_model", return_value="lfm2.5-1.2b-instruct"), \
             patch("urllib.request.urlopen") as mock_urlopen:
            r1 = MagicMock()
            r1.status = 200
            r1.read.return_value = json.dumps({"response": "The ocean is blue because water scatters light."}).encode("utf-8")

            r2 = MagicMock()
            r2.status = 200
            r2.read.return_value = json.dumps({"response": "Kulay asul ang dagat dahil sa liwanag ng araw."}).encode("utf-8")

            ctx1 = MagicMock()
            ctx1.__enter__.return_value = r1
            ctx2 = MagicMock()
            ctx2.__enter__.return_value = r2
            mock_urlopen.side_effect = [ctx1, ctx2]

            reply = client.generate("Bakit asul ang dagat?", language="fil")
            assert reply == "Kulay asul ang dagat dahil sa liwanag ng araw."
            assert not is_mostly_english(reply)
            assert mock_urlopen.call_count == 2

    def test_mocked_persistent_english_falls_back_to_filipino_fallback(self) -> None:
        """If Ollama keeps answering in English after retry, return safe Filipino fallback."""
        client = OllamaClient()
        with patch.object(client, "resolve_model", return_value="lfm2.5-1.2b-instruct"), \
             patch("urllib.request.urlopen") as mock_urlopen:
            r = MagicMock()
            r.status = 200
            r.read.return_value = json.dumps({"response": "This is an English only sentence."}).encode("utf-8")
            ctx = MagicMock()
            ctx.__enter__.return_value = r
            mock_urlopen.return_value = ctx

            reply = client.generate("Ano ang kahulugan ng buhay?", language="fil")
            assert reply == get_string("AI_EMPTY_FALLBACK", "fil")
            assert not is_mostly_english(reply)

    def test_ai_mode_filipino_lcd_and_speech_parity_and_single_voice(self) -> None:
        """LCD text and SpeechService input are identical; thinking prompt uses i18n Filipino string."""
        app = MagicMock(spec=BrailceApp)
        app.state = ApplicationState()
        app.state.language = "fil"
        app.display = MagicMock()
        app.speech = MagicMock(spec=SpeechService)
        app.encoder = MagicMock()
        app.ai_client = MagicMock()
        app.ai_client.generate.return_value = "Ang Maynila ang kabisera ng Pilipinas."
        app.config = MagicMock()
        app.config.get.return_value = True

        mode = AIAssistantMode(app)
        mode._process_query("Ano ang kabisera ng Pilipinas?")

        # First speech call was Filipino thinking prompt from i18n table
        first_spoken = app.speech.speak_text.call_args_list[0][0][0]
        assert first_spoken == get_string("AI_THINKING", "fil")

        # Final answer on LCD and spoken text are identical
        lcd_resp = app.display.show_ai_response.call_args[0][0]
        last_spoken = app.speech.speak_text.call_args_list[-1][0][0]
        assert lcd_resp == last_spoken == "Ang Maynila ang kabisera ng Pilipinas."

