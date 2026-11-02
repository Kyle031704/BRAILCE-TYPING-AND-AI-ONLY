"""
Brailce Piper Neural TTS Engine – Natural Human Voice
======================================================

High-fidelity, offline neural text-to-speech powered by Piper (VITS).
Runs locally on Raspberry Pi 5 with studio-grade human pronunciation,
fluid co-articulation, and natural intonation.
Uses the preferred voice model: en_US-ryan-medium.onnx.

Features:
- Instant, zero-latency playback for single characters via pre-rendered
  WAV cache (exact same voice, pitch, and timbre as sentences).
- Dynamic neural synthesis for completed words, sentences, math explanations,
  and AI tutor responses.
- Shared audio device coordination to prevent ALSA I2S DAC lockups.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

from brailce.speech.audio_device import AudioDeviceCoordinator
from brailce.speech.base import SpeechEngine
from brailce.speech.normalizer import (
    CHARACTER_FILENAME_MAP,
    ESPEAK_DIGIT_WORDS_TO_DIGIT,
    ESPEAK_NAME_TO_CHAR,
    is_keystroke_feedback,
    normalize_for_piper,
)

logger = logging.getLogger(__name__)


def _find_cache_dir() -> Optional[str]:
    """Locate the pre-rendered character WAV audio cache directory."""
    # 1. Project directory: brailce/audio/characters
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "characters"))
    if os.path.isdir(base):
        return base

    # 2. Raspberry Pi standard installation location
    pi_dir = "/home/raspberrypi5/brailce/brailce/audio/characters"
    if os.path.isdir(pi_dir):
        return pi_dir

    return None


def _find_piper_binary() -> Optional[str]:
    """Locate the piper binary on the system."""
    # 1. System PATH
    found = shutil.which("piper")
    if found:
        return found

    # 2. Common install locations on Raspberry Pi
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "piper", "piper", "piper"),
        os.path.join(home, "piper", "piper"),
        "/usr/local/bin/piper",
        "/usr/bin/piper",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _find_piper_model() -> Optional[str]:
    """Locate the preferred Piper ONNX voice model."""
    home = os.path.expanduser("~")
    search_dirs = [
        os.path.join(home, "piper"),
        os.path.join(home, "piper", "piper"),
        os.path.join(home, ".local", "share", "piper"),
        "/usr/local/share/piper",
    ]
    # Preferred natural models in order of warmth (ryan-medium is default)
    preferred_names = [
        "en_US-ryan-medium.onnx",
        "en_US-bryce-medium.onnx",
        "en_US-lessac-medium.onnx",
        "en_GB-alan-medium.onnx",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in preferred_names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        # Any .onnx file in the directory
        try:
            for f in os.listdir(d):
                if f.endswith(".onnx"):
                    return os.path.join(d, f)
        except OSError:
            pass
    return None


class PiperEngine(SpeechEngine):
    """
    Piper Neural TTS Engine.
    
    Produces smooth, studio-quality human voice completely offline
    with negligible latency on Raspberry Pi 5.
    Used for characters (pre-cached), words, sentences, math, and AI.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        rate: int = 160,
        volume: int = 100,
        max_queue_size: int = 50,
    ) -> None:
        self._binary = binary_path or _find_piper_binary()
        self._model = model_path or _find_piper_model()
        self._rate = rate
        self._volume = volume
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=max_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._current_process: Optional[subprocess.Popen] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        return "Piper Neural TTS (Studio Voice)"

    def is_available(self) -> bool:
        """Check if piper binary and a voice model are available."""
        if not self._binary:
            self._binary = _find_piper_binary()
        if not self._model:
            self._model = _find_piper_model()
        return bool(self._binary and os.path.exists(self._binary) and self._model and os.path.exists(self._model))

    def set_voice(self, voice: str) -> None:
        if os.path.exists(voice):
            self._model = voice

    def set_rate(self, rate: int) -> None:
        self._rate = max(80, min(300, rate))

    def set_volume(self, volume: int) -> None:
        self._volume = max(0, min(100, volume))

    def stop(self) -> None:
        """Immediately stop currently playing audio and drain queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        AudioDeviceCoordinator.interrupt_active("piper")
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass
                self._current_process = None

    def play_cached_character(self, character: str) -> bool:
        """
        Play pre-rendered high-fidelity Piper Ryan WAV for an individual keystroke.
        Provides instant zero-latency feedback with the exact same voice quality,
        timbre, and accent as full-sentence speech.
        """
        cache_dir = _find_cache_dir()
        if not cache_dir:
            return False

        clean = character.strip()
        if not clean:
            return False

        upper = clean.upper().rstrip(".")

        fname = None
        if clean in CHARACTER_FILENAME_MAP:
            fname = CHARACTER_FILENAME_MAP[clean]
        elif upper in CHARACTER_FILENAME_MAP:
            fname = CHARACTER_FILENAME_MAP[upper]
        elif len(upper) == 1 and upper.isalnum():
            fname = f"char_{upper}.wav"
        elif upper in ESPEAK_NAME_TO_CHAR:
            char_key = ESPEAK_NAME_TO_CHAR[upper]
            fname = f"char_{char_key}.wav"
        elif upper in ESPEAK_DIGIT_WORDS_TO_DIGIT:
            digit_key = ESPEAK_DIGIT_WORDS_TO_DIGIT[upper]
            fname = f"char_{digit_key}.wav"
        elif upper in ("PLUS", "MINUS", "TIMES", "DIVIDED BY", "EQUALS"):
            name_map = {
                "PLUS": "char_plus.wav",
                "MINUS": "char_minus.wav",
                "TIMES": "char_times.wav",
                "DIVIDED BY": "char_divide.wav",
                "EQUALS": "char_equals.wav",
            }
            fname = name_map[upper]
        elif upper in ("PERIOD", "COMMA", "QUESTION MARK", "EXCLAMATION MARK", "NUMBER SIGN"):
            name_map = {
                "PERIOD": "char_period.wav",
                "COMMA": "char_comma.wav",
                "QUESTION MARK": "char_question.wav",
                "EXCLAMATION MARK": "char_exclamation.wav",
                "NUMBER SIGN": "char_hash.wav",
            }
            fname = name_map[upper]

        if not fname:
            return False

        wav_path = os.path.join(cache_dir, fname)
        if not os.path.isfile(wav_path):
            return False

        # On Linux (Raspberry Pi 5): aplay directly to MAX98357A I2S DAC
        if sys.platform != "win32":
            audio_lock = AudioDeviceCoordinator.get_lock()
            with audio_lock:
                try:
                    proc = subprocess.Popen(
                        ["aplay", "-q", wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    AudioDeviceCoordinator.register_process(proc, "piper_cache")
                    with self._lock:
                        self._current_process = proc
                    proc.wait(timeout=5)
                    return True
                except Exception as exc:
                    logger.debug("Cached audio aplay failed: %s", exc)
                    return False
                finally:
                    with self._lock:
                        self._current_process = None
                    AudioDeviceCoordinator.unregister_process()
        else:
            # On Windows: play synchronously in this thread via winsound
            try:
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                return True
            except Exception as exc:
                logger.debug("Cached audio winsound failed: %s", exc)
                return False

    def speak(self, text: str, interrupt: Optional[bool] = None) -> None:
        """
        Enqueue text for natural speech synthesis.
        For single keystrokes with pre-rendered cache, plays immediately.
        """
        if not text:
            return

        trimmed = text.strip()
        if not trimmed:
            return

        # Fast path: check pre-rendered character WAV cache
        if is_keystroke_feedback(trimmed):
            if self.play_cached_character(trimmed):
                return

        normalized = normalize_for_piper(text)
        if not normalized:
            return

        if self._worker is None or not self._worker.is_alive():
            self._start_worker()

        # Drain previous queue items ONLY if explicit interrupt requested
        if interrupt is True:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            AudioDeviceCoordinator.interrupt_active("piper")

        try:
            self._queue.put_nowait(normalized)
        except queue.Full:
            logger.warning("Piper speech queue full — dropping item")

    def _start_worker(self) -> None:
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="piper-worker",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:
                break

            self._speak_sync(text)

    def _speak_sync(self, text: str) -> None:
        if not self.is_available():
            logger.warning("Piper not available, unable to speak: %s", text)
            return

        speech_text = normalize_for_piper(text)
        if not speech_text:
            return

        # -------------------------------------------------------------------
        # Piper pipeline:
        #   piper --model model.onnx --output-raw --length-scale 1.0 | aplay ...
        #
        # Acquires shared AudioDeviceCoordinator lock so Piper and eSpeak
        # never conflict or collide on the MAX98357A I2S DAC.
        # -------------------------------------------------------------------
        cmd_piper = [
            self._binary,
            "--model", self._model,
            "--output-raw",
            "--length-scale", "1.0",
        ]

        # 150ms leading silence (6616 bytes) for MAX98357A I2S DAC clock lock
        # 200ms trailing silence (8820 bytes) to ensure full DMA buffer drain without clipping
        # 22050 samples/s * 2 bytes/sample = 44100 bytes/s
        pre_silence = b"\x00" * 6616
        post_silence = b"\x00" * 8820

        audio_lock = AudioDeviceCoordinator.get_lock()
        with audio_lock:
            for attempt in range(3):
                p_piper = None
                p_aplay = None
                try:
                    p_piper = subprocess.Popen(
                        cmd_piper,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    raw_pcm, _ = p_piper.communicate(
                        speech_text.encode("utf-8") + b"\n", timeout=15
                    )
                    if not raw_pcm:
                        continue

                    padded_pcm = pre_silence + raw_pcm + post_silence

                    if sys.platform != "win32":
                        p_aplay = subprocess.Popen(
                            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q", "-"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        with self._lock:
                            self._current_process = p_aplay
                        AudioDeviceCoordinator.register_process(p_aplay, "piper")

                        p_aplay.communicate(padded_pcm, timeout=20)
                        if p_aplay.returncode == 0:
                            break
                        else:
                            time.sleep(0.08)
                    else:
                        # Windows simulation fallback
                        import tempfile, wave, winsound
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                            t_path = tf.name
                        with wave.open(t_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(22050)
                            wf.writeframes(padded_pcm)
                        winsound.PlaySound(t_path, winsound.SND_FILENAME)
                        try:
                            os.remove(t_path)
                        except Exception:
                            pass
                        break
                except Exception as e:
                    logger.debug("Piper speech attempt %d failed: %s", attempt, e)
                    time.sleep(0.08)
                finally:
                    if p_piper and p_piper.poll() is None:
                        try:
                            p_piper.kill()
                        except Exception:
                            pass
                    if p_aplay and p_aplay.poll() is None:
                        try:
                            p_aplay.kill()
                        except Exception:
                            pass
                    with self._lock:
                        self._current_process = None
                    AudioDeviceCoordinator.unregister_process()
