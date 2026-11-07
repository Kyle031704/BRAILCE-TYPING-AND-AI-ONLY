"""
Brailce Phase 5 Tests – Speech Engine
========================================

Tests the eSpeak NG speech engine backend and queuing behavior.
"""

import queue
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

from brailce.speech.espeak_engine import EspeakEngine


class TestEspeakEngine:
    def test_engine_name(self) -> None:
        engine = EspeakEngine()
        assert engine.engine_name == "eSpeak NG"

    def test_set_voice_rate_volume(self) -> None:
        engine = EspeakEngine()
        
        engine.set_voice("es")
        assert engine._voice == "es"
        
        engine.set_rate(300)
        assert engine._rate == 300
        
        engine.set_volume(50)
        assert engine._volume == 50

    def test_rate_and_volume_caps(self) -> None:
        engine = EspeakEngine()
        
        engine.set_rate(1000)
        assert engine._rate == 500  # Capped max
        
        engine.set_rate(10)
        assert engine._rate == 80  # Floored min
        
        engine.set_volume(250)
        assert engine._volume == 200
        
        engine.set_volume(-10)
        assert engine._volume == 0

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        engine = EspeakEngine()
        assert engine.is_available() is True
        
    @patch("subprocess.run")
    def test_is_available_false(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        engine = EspeakEngine()
        assert engine.is_available() is False

    @patch("subprocess.run")
    def test_is_available_fallback(self, mock_run: MagicMock) -> None:
        # First call raises FileNotFoundError, second succeeds
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.side_effect = [FileNotFoundError(), mock_result]
        
        engine = EspeakEngine()
        assert engine.is_available() is True
        assert mock_run.call_count == 2

    @patch("sys.platform", "linux")
    @patch("subprocess.run")
    def test_speak_sync(self, mock_run: MagicMock) -> None:
        engine = EspeakEngine(voice="en", rate=160, volume=75)
        engine._speak_sync("Hello")
        
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "espeak" in args[0]
        assert args[1] == "-v"
        assert args[2] == "en"
        assert args[3] == "-s"
        assert args[4] == "160"
        assert args[5] == "-a"
        assert args[6] == "75"
        assert args[-1] == "Hello"

    @pytest.mark.skipif(sys.platform != "win32", reason="SAPI fallback is Windows-only")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_speak_sync_fallback_to_sapi(self, mock_run: MagicMock) -> None:
        import sys
        mock_comtypes = MagicMock()
        mock_client = MagicMock()
        mock_sapi_engine = MagicMock()
        mock_client.CreateObject.return_value = mock_sapi_engine
        mock_comtypes.client = mock_client

        with patch.dict(sys.modules, {"comtypes": mock_comtypes, "comtypes.client": mock_client}):
            engine = EspeakEngine(voice="en", rate=160, volume=75)
            engine._speak_sync("Fallback test")

            mock_client.CreateObject.assert_called_once_with("SAPI.SpVoice")
            mock_sapi_engine.Speak.assert_called_once_with("Fallback test", 0)

    def test_queue_overflow_protection(self) -> None:
        """Test that engine handles full queue without crashing."""
        engine = EspeakEngine(max_queue_size=2)
        # Manually force running true so it doesn't try to start worker during test setup
        engine._running = True 
        
        # We don't start the worker so queue just fills up
        engine.speak("One")
        engine.speak("Two")
        
        # Third one should log warning but not crash
        engine.speak("Three")
        
        assert engine._queue.qsize() == 2

    @patch("brailce.speech.espeak_engine.EspeakEngine._speak_sync")
    def test_worker_thread(self, mock_speak_sync: MagicMock) -> None:
        """Test that the background thread processes queued items."""
        engine = EspeakEngine()
        
        engine.speak("First")
        engine.speak("Second")
        
        # Wait a moment for worker thread to process
        time.sleep(0.5)
        
        engine.stop()
        
        # Both should have been processed
        assert mock_speak_sync.call_count >= 2
        mock_speak_sync.assert_any_call("First")
        mock_speak_sync.assert_any_call("Second")

    def test_stop_clears_queue_and_joins(self) -> None:
        engine = EspeakEngine()
        
        engine._running = True
        engine._queue.put("Waiting")
        
        # Stop should clear the queue
        engine.stop()
        
        assert engine._queue.empty()
        assert not engine._running
        assert engine._worker is None


def test_piper_slice_carrier_audio_preserves_short_words() -> None:
    """Verify that slice_carrier_audio cuts safely inside the colon pause before target word."""
    from brailce.speech.piper_engine import slice_carrier_audio
    import math

    sr = 22050
    # 1. Lead-in silence (20ms)
    s1 = [0] * int(sr * 0.02)
    # 2. "Please say:" speech (450ms, loud speech energy)
    s2 = [int(12000 * math.sin(2 * math.pi * 300 * (t / sr))) for t in range(int(sr * 0.45))]
    # 3. Colon pause (70ms, lower energy valley)
    s3 = [int(300 * math.sin(2 * math.pi * 100 * (t / sr))) for t in range(int(sr * 0.07))]
    # 4. Target word "is" (lax vowel ~100ms, sibilant ~150ms)
    vowel = [int(2500 * math.sin(2 * math.pi * 200 * (t / sr))) for t in range(int(sr * 0.10))]
    sibilant = [int(6000 * math.sin(2 * math.pi * 4000 * (t / sr))) for t in range(int(sr * 0.15))]
    # 5. Trailing silence (100ms)
    s5 = [0] * int(sr * 0.10)

    samples = s1 + s2 + s3 + vowel + sibilant + s5
    sliced = slice_carrier_audio(samples, sr=sr)

    # Word starts after s1 + s2 + s3
    word_start = len(s1) + len(s2) + len(s3)
    cut_idx = len(samples) - len(sliced)

    # The cut must occur inside the colon pause BEFORE word_start
    assert cut_idx <= word_start, f"Cut point {cut_idx} must not cut into the word at {word_start}"
    # The cut must occur AFTER 'Please say:' main speech
    please_say_end = len(s1) + int(len(s2) * 0.85)
    assert cut_idx >= please_say_end, f"Cut point {cut_idx} must remove 'Please say:' which ended near {please_say_end}"
    # All of the vowel and sibilant must be retained
    assert len(sliced) >= len(vowel) + len(sibilant)
