"""
Brailce Piper Neural TTS Engine – Natural Human Voice
======================================================

High-fidelity, offline neural text-to-speech powered by Piper (VITS).
Runs locally on Raspberry Pi 5 with studio-grade human pronunciation,
fluid co-articulation, and natural intonation.
Uses the preferred voice model: en_US-ryan-medium.onnx.

Unified Audio Pipeline:
- Direct Piper neural synthesis for all speech: words, sentences, characters, and announcements.
- Instant pre-cached WAV playback for single character keystroke feedback.
- Locked VITS parameters: length_scale 1.08, noise_scale 0.5, noise_w 0.6.
- Pure wave module silence padding (120ms pre, 150ms post).
- Played strictly with `aplay -q <tmp_wav_path>`. Never uses `aplay -t raw`.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Optional

from brailce.speech.audio_device import AudioDeviceCoordinator
from brailce.speech.base import SpeechEngine
from brailce.speech.normalizer import (
    CHARACTER_FILENAME_MAP,
    normalize_for_piper,
)

logger = logging.getLogger(__name__)

# Standard VITS parameters for stable American English vowels and Grade 1 pacing
PIPER_LENGTH_SCALE = "1.08"
PIPER_NOISE_SCALE = "0.5"
PIPER_NOISE_W = "0.6"


def _find_cache_dir() -> Optional[str]:
    """Locate the pre-rendered character WAV audio cache directory."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "characters"))
    if os.path.isdir(base):
        return base
    pi_dir = "/home/raspberrypi5/brailce/brailce/audio/characters"
    if os.path.isdir(pi_dir):
        return pi_dir
    return None


def _find_piper_binary() -> Optional[str]:
    """Locate the piper binary on the system."""
    found = shutil.which("piper")
    if found:
        return found

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
        try:
            for f in os.listdir(d):
                if f.endswith(".onnx"):
                    return os.path.join(d, f)
        except OSError:
            pass
    return None


def slice_carrier_audio(samples: list[int], sr: int = 22050) -> list[int]:
    """
    Backward-Scanning Carrier Audio Slicer:
    - Synthesizes: "Please say: <target>." with Piper.
    - Scans backward in the first half of the audio (between 0.40s and 0.95s from speech onset)
      to locate the silence gap separating "Please say:" from <target>.
    - Cuts 50ms (5 windows) BEFORE the target word starts.
    - Keeps 100% of remaining audio to the very end of the file.
    """
    win = int(sr * 0.010)  # 10ms window = 220 samples at 22050Hz
    if win <= 0:
        win = 220

    rms_list = []
    for i in range(0, len(samples) - win, win):
        chunk = samples[i : i + win]
        r = math.sqrt(sum(s * s for s in chunk) / win)
        rms_list.append(r)

    total_windows = len(rms_list)
    if total_windows < 40:
        return samples

    first_voiced = next((i for i, r in enumerate(rms_list) if r > 300), 0)
    last_voiced = max((i for i, r in enumerate(rms_list) if r > 500), default=total_windows - 1)

    search_start = min(total_windows - 10, first_voiced + 30)
    search_end = max(search_start + 1, last_voiced - 8)

    min_win = min(range(search_start, search_end), key=lambda k: rms_list[k])

    target_onset = min_win
    for j in range(min_win, min(total_windows, min_win + 30)):
        if rms_list[j] > 500:
            target_onset = j
            break

    cut_window = max(first_voiced + 25, target_onset - 4)
    cut_sample = cut_window * win
    sliced = samples[cut_sample:]
    if not sliced or len(sliced) < int(sr * 0.025) or max(abs(s) for s in sliced) < 500:
        return samples
    return sliced


def _process_carrier_wav(carrier_wav: str, pre_s: float = 0.12, post_s: float = 0.15) -> bool:
    try:
        with wave.open(carrier_wav, "rb") as wf:
            params = wf.getparams()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if params.nchannels != 1 or params.sampwidth != 2 or nframes == 0:
            return False

        num_samples = len(raw) // 2
        samples = list(struct.unpack(f"<{num_samples}h", raw))
        target_samples = slice_carrier_audio(samples, sr=params.framerate)
        if not target_samples or max(abs(s) for s in target_samples) < 500:
            return False

        target_pcm = struct.pack(f"<{len(target_samples)}h", *target_samples)
        pre_silence = b"\x00\x00" * int(params.framerate * pre_s)
        post_silence = b"\x00\x00" * int(params.framerate * post_s)
        padded = pre_silence + target_pcm + post_silence

        with wave.open(carrier_wav, "wb") as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(padded)

        return True
    except Exception as exc:
        logger.debug("Carrier slicing error: %s", exc)
        return False


def _pad_wav_with_silence(wav_path: str, pre_ms: int = 100, post_ms: int = 150) -> None:
    """Pad WAV file with silence using Python's wave module."""
    try:
        with wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        pre_silence = b"\x00\x00" * int(params.framerate * (pre_ms / 1000.0))
        post_silence = b"\x00\x00" * int(params.framerate * (post_ms / 1000.0))

        with wave.open(wav_path, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(pre_silence + frames + post_silence)
    except Exception as exc:
        logger.debug("Silence padding error: %s", exc)


class PiperEngine(SpeechEngine):
    """
    Piper Neural TTS Engine.
    Direct synthesis pipeline for sentences, words, and characters with
    wave silence padding and playback via `aplay -q`.
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

    def stop_current(self) -> None:
        self.stop()

    def play_cached_character(self, character: str) -> bool:
        """
        Instant zero-latency playback for single characters using pre-rendered WAV cache.
        Returns True if played successfully, False if file missing.
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

        if not fname:
            return False

        wav_path = os.path.join(cache_dir, fname)
        if not os.path.isfile(wav_path):
            return False

        return AudioDeviceCoordinator.play_wav_file(wav_path, owner="piper_cache", timeout=5.0)

    def speak(self, text: str, interrupt: Optional[bool] = None) -> None:
        if not text:
            return

        normalized = normalize_for_piper(text)
        if not normalized:
            return

        if self._worker is None or not self._worker.is_alive():
            self._start_worker()

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
        """
        Synthesize speech:
          - Single words on SPACE: synthesized via Carrier-Sentence ("Please say. <target>.")
            and sliced at the longest 550ms silence gap, with automatic direct fallback.
          - Multi-word sentences, AI responses, mode intros: synthesized directly.
          - Play strictly via aplay -q <tmp_wav_path>.
        """
        if not self.is_available():
            logger.warning("Piper not available, unable to speak: %s", text)
            return

        speech_text = normalize_for_piper(text)
        if not speech_text:
            return

        words = speech_text.split()
        is_single_word = len(words) == 1

        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="brailce_piper_", suffix=".wav")
            os.close(tmp_fd)
            tmp_fd = None

            audio_lock = AudioDeviceCoordinator.get_lock()
            with audio_lock:
                for attempt in range(3):
                    p_piper = None
                    try:
                        if is_single_word:
                            target_word = speech_text.strip().rstrip(".")
                            carrier_prompt = f"Please say: {target_word}."
                            cmd_piper = [
                                self._binary,
                                "--model", self._model,
                                "--output_file", tmp_path,
                                "--length_scale", PIPER_LENGTH_SCALE,
                                "--noise_scale", PIPER_NOISE_SCALE,
                                "--noise_w", PIPER_NOISE_W,
                            ]
                            p_piper = subprocess.Popen(
                                cmd_piper,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            p_piper.communicate(carrier_prompt.encode("utf-8") + b"\n", timeout=15)

                            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                                time.sleep(0.08)
                                continue

                            sliced = _process_carrier_wav(tmp_path, pre_s=0.12, post_s=0.15)
                            if not sliced:
                                # Safety fallback: direct synthesis without carrier
                                cmd_direct = [
                                    self._binary,
                                    "--model", self._model,
                                    "--output_file", tmp_path,
                                    "--length_scale", PIPER_LENGTH_SCALE,
                                    "--noise_scale", PIPER_NOISE_SCALE,
                                    "--noise_w", PIPER_NOISE_W,
                                ]
                                p_dir = subprocess.Popen(
                                    cmd_direct,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                                p_dir.communicate(speech_text.encode("utf-8") + b"\n", timeout=15)
                                _pad_wav_with_silence(tmp_path, pre_ms=120, post_ms=150)
                        else:
                            # Multi-word sentence / utterance
                            cmd_piper = [
                                self._binary,
                                "--model", self._model,
                                "--output_file", tmp_path,
                                "--length_scale", PIPER_LENGTH_SCALE,
                                "--noise_scale", PIPER_NOISE_SCALE,
                                "--noise_w", PIPER_NOISE_W,
                            ]
                            p_piper = subprocess.Popen(
                                cmd_piper,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            p_piper.communicate(speech_text.encode("utf-8") + b"\n", timeout=15)

                            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                                time.sleep(0.08)
                                continue

                            _pad_wav_with_silence(tmp_path, pre_ms=100, post_ms=150)

                        # Play strictly with `aplay -q <tmp_path>`
                        success = AudioDeviceCoordinator.play_wav_file(tmp_path, owner="piper", timeout=30.0)
                        if success:
                            break
                        else:
                            time.sleep(0.08)

                    except Exception as e:
                        logger.debug("Piper speech attempt %d failed: %s", attempt, e)
                        time.sleep(0.08)
                    finally:
                        if p_piper and p_piper.poll() is None:
                            try:
                                p_piper.kill()
                            except Exception:
                                pass
                        with self._lock:
                            self._current_process = None
                        AudioDeviceCoordinator.unregister_process()

        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except Exception:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
