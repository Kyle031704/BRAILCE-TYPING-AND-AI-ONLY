"""
Regression test suite ensuring English behavior, speech normalization,
audio files, and string definitions remain frozen and byte-for-byte identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from unittest.mock import MagicMock

import pytest

from brailce.i18n.strings import STRINGS
from brailce.speech.normalizer import prepare_spoken_word, normalize_for_piper
from brailce.speech.service import SpeechService


def test_english_audio_files_manifest():
    """Verify that all English audio files exist and match their SHA-256 hashes."""
    manifest_path = os.path.join(os.path.dirname(__file__), "english_audio_manifest.sha256")
    assert os.path.isfile(manifest_path), f"Manifest file missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0, "English audio manifest is empty"

    for line in lines:
        expected_hash, rel_path = line.split("  ", 1)
        rel_path = rel_path.strip().replace("/", os.sep)
        full_path = os.path.abspath(rel_path)

        assert os.path.isfile(full_path), f"Frozen English audio file missing: {rel_path}"

        with open(full_path, "rb") as af:
            actual_hash = hashlib.sha256(af.read()).hexdigest()

        assert actual_hash == expected_hash, (
            f"Frozen English audio file was modified! File: {rel_path}\n"
            f"Expected SHA256: {expected_hash}\n"
            f"Actual SHA256:   {actual_hash}"
        )


def test_english_strings_frozen():
    """Verify that all existing English STRINGS remain identical to baseline snapshot."""
    snapshot_path = os.path.join(os.path.dirname(__file__), "english_strings_snapshot.json")
    assert os.path.isfile(snapshot_path), f"Snapshot file missing: {snapshot_path}"

    with open(snapshot_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    for key, expected_val in baseline.items():
        assert key in STRINGS, f"Existing English key removed: {key}"
        assert "en" in STRINGS[key], f"Missing 'en' variant for key: {key}"
        assert STRINGS[key]["en"] == expected_val, (
            f"English string modified for key '{key}'!\n"
            f"Expected: {expected_val!r}\n"
            f"Actual:   {STRINGS[key]['en']!r}"
        )


def test_prepare_spoken_word_english():
    """Verify prepare_spoken_word outputs for English inputs."""
    test_cases = [
        ("hello", "hello."),
        ("world", "world."),
        ("HELLO", "hello."),
        ("  test  ", "test."),
        ("apple.", "apple."),
        ("what?", "what?"),
        ("stop!", "stop!"),
        ("wait;", "wait,"),
        ("hello, world", "hello, world."),
    ]
    for raw, expected in test_cases:
        actual = prepare_spoken_word(raw, lang="en")
        assert actual == expected, f"Failed for {raw!r}: expected {expected!r}, got {actual!r}"


def test_speech_service_english_mock():
    """Verify that SpeechService calls mock character/text engines correctly in English mode."""
    mock_char_engine = MagicMock()
    mock_char_engine.is_available.return_value = True
    mock_char_engine.play_cached_character.return_value = False

    mock_text_engine = MagicMock()
    mock_text_engine.is_available.return_value = True
    mock_text_engine.play_cached_word.return_value = False

    mock_filipino_engine = MagicMock()

    service = SpeechService(
        character_engine=mock_char_engine,
        text_engine=mock_text_engine,
    )
    service._filipino_engine = mock_filipino_engine
    service.language = "en"

    # 1. Speak character 'A' -> character_engine.speak('a.', interrupt=False)
    service.speak_character("A")
    mock_char_engine.speak.assert_called_with("a.", interrupt=False)
    mock_filipino_engine.enqueue_character.assert_not_called()

    # 2. Speak digit '1' -> character_engine.speak('one.', interrupt=False)
    mock_char_engine.reset_mock()
    service.speak_character("1")
    mock_char_engine.speak.assert_called_with("one.", interrupt=False)

    # 3. Speak word 'hello' -> text_engine.speak('hello.', interrupt=False)
    mock_text_engine.reset_mock()
    service.speak_word_clean("hello")
    mock_text_engine.speak.assert_called_with("hello.", interrupt=False)
    mock_filipino_engine.enqueue_text.assert_not_called()

    # 4. Speak sentence 'Hello world.' -> text_engine.speak('Hello world.', interrupt=False)
    mock_text_engine.reset_mock()
    service.speak_sentence("Hello world.")
    mock_text_engine.speak.assert_called_with("Hello world.", interrupt=False)
    mock_filipino_engine.enqueue_text.assert_not_called()
