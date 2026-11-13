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
- Locked VITS parameters: length_scale 1.05, noise_scale 0.5, noise_w 0.6.
- Pure wave module silence padding (120ms pre, 180ms post).
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
from brailce.speech.audio_normalize import (
    DEFAULT_POST_SILENCE_MS,
    DEFAULT_PRE_SILENCE_MS,
    slice_carrier_audio,
    standardize_wav_file,
)
from brailce.speech.base import SpeechEngine
from brailce.speech.normalizer import (
    CHARACTER_FILENAME_MAP,
    normalize_for_piper,
)

logger = logging.getLogger(__name__)

# Single fixed VITS parameter set for ALL English output (characters, words, sentences, intros).
# Option 1 (carrier-sentence slicing "Please say: <word>." + brailce/audio/words_en/ cache) is used
# rather than Option 2 ([[ phonemes ]] overrides) because it generalizes to arbitrary vocabulary
# without Piper build-flag dependencies while keeping "is" stressed and /w/, /wh/ onsets intact.
PIPER_LENGTH_SCALE = "1.05"
PIPER_NOISE_SCALE = "0.5"
PIPER_NOISE_W = "0.6"


def _find_cache_dir() -> Optional[str]:
    """Locate the pre-rendered character WAV audio cache directory."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "characters"))
    if os.path.isdir(base):
        return base
    home_dir = os.path.expanduser(os.path.join("~", "brailce", "brailce", "audio", "characters"))
    if os.path.isdir(home_dir):
        return home_dir
    return None


def _find_words_cache_dir(create: bool = False) -> str:
    """Locate (or create) the English word WAV cache directory (brailce/audio/words_en/)."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "words_en"))
    if create:
        os.makedirs(base, exist_ok=True)
    return base



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


def _pad_wav_with_silence(wav_path: str, pre_ms: int = 150, post_ms: int = 200) -> None:
    """Prepend 150ms pure digital zero PCM silence and append 200ms post-silence to WAV file."""
    try:
        with wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        bytes_per_sample = params.sampwidth * params.nchannels
        pre_silence = b"\x00" * bytes_per_sample * int(params.framerate * (pre_ms / 1000.0))
        post_silence = b"\x00" * bytes_per_sample * int(params.framerate * (post_ms / 1000.0))

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

        spoken_name_map = {
            "PLUS": "char_plus.wav",
            "MINUS": "char_minus.wav",
            "TIMES": "char_times.wav",
            "DIVIDED BY": "char_divide.wav",
            "EQUALS": "char_equals.wav",
            "PERIOD": "char_period.wav",
            "COMMA": "char_comma.wav",
            "QUESTION MARK": "char_question.wav",
            "EXCLAMATION MARK": "char_exclamation.wav",
            "NUMBER SIGN": "char_hash.wav",
            "ZERO": "char_0.wav", "ONE": "char_1.wav", "TWO": "char_2.wav",
            "THREE": "char_3.wav", "FOUR": "char_4.wav", "FIVE": "char_5.wav",
            "SIX": "char_6.wav", "SEVEN": "char_7.wav", "EIGHT": "char_8.wav", "NINE": "char_9.wav",
        }

        fname = None
        if clean in CHARACTER_FILENAME_MAP:
            fname = CHARACTER_FILENAME_MAP[clean]
        elif upper in CHARACTER_FILENAME_MAP:
            fname = CHARACTER_FILENAME_MAP[upper]
        elif upper in spoken_name_map:
            fname = spoken_name_map[upper]
        elif len(upper) == 1 and upper.isalnum():
            fname = f"char_{upper}.wav"

        if not fname:
            return False

        wav_path = os.path.join(cache_dir, fname)
        if not os.path.isfile(wav_path):
            return False

        return AudioDeviceCoordinator.play_wav_file(wav_path, owner="piper_cache", timeout=5.0)

    def get_cached_word_path(self, word: str) -> Optional[str]:
        """Return path to pre-rendered / cached English word WAV if it exists."""
        clean = "".join(ch for ch in word.strip().lower() if ch.isalnum() or ch == "_")
        if not clean:
            return None
        words_dir = _find_words_cache_dir(create=False)
        candidate = os.path.join(words_dir, f"word_{clean}.wav")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 44:
            return candidate
        return None

    def play_cached_word(self, word: str) -> bool:
        """Play English word from brailce/audio/words_en/word_<word>.wav if present."""
        wav_path = self.get_cached_word_path(word)
        if not wav_path:
            return False
        return AudioDeviceCoordinator.play_wav_file(wav_path, owner="piper_word_cache", timeout=10.0)

    def synthesize_to_wav(
        self,
        text: str,
        out_wav_path: str,
        carrier_slice: bool = False,
        sentence_silence: Optional[str] = None,
    ) -> bool:
        """
        Synthesize speech to out_wav_path using locked Piper VITS parameters.
        If carrier_slice=True, wraps text in "Please say: <text>." with a colon pause
        and extracts the target word cleanly with slice_carrier_audio(), then applies
        uniform 22050 Hz mono s16 conversion, RMS loudness normalization, and padding.
        """
        if not self.is_available():
            return False

        clean = text.strip()
        if not clean:
            return False

        if carrier_slice:
            synth_input = f"Please say: {clean.rstrip('.!?,;:') }."
            sil = sentence_silence or "0.55"
        else:
            synth_input = clean
            sil = sentence_silence or "0.35"

        for attempt in range(3):
            p_piper = None
            try:
                cmd_piper = [
                    self._binary,
                    "--model", self._model,
                    "--output_file", out_wav_path,
                    "--length_scale", PIPER_LENGTH_SCALE,
                    "--noise_scale", PIPER_NOISE_SCALE,
                    "--noise_w", PIPER_NOISE_W,
                    "--sentence_silence", sil,
                ]
                p_piper = subprocess.Popen(
                    cmd_piper,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                p_piper.communicate(synth_input.encode("utf-8") + b"\n", timeout=15)

                if not os.path.isfile(out_wav_path) or os.path.getsize(out_wav_path) <= 44:
                    time.sleep(0.08)
                    continue

                if carrier_slice:
                    with wave.open(out_wav_path, "rb") as wf:
                        params = wf.getparams()
                        raw = wf.readframes(wf.getnframes())
                    if params.sampwidth == 2 and len(raw) >= 2:
                        total_samples = len(raw) // (params.sampwidth * params.nchannels)
                        samples = struct.unpack(f"<{total_samples * params.nchannels}h", raw)
                        if params.nchannels > 1:
                            samples = samples[::params.nchannels]
                        sliced = slice_carrier_audio(samples, sr=params.framerate)
                        sliced_bytes = struct.pack(f"<{len(sliced)}h", *sliced)
                        with wave.open(out_wav_path, "wb") as wf_out:
                            wf_out.setnchannels(1)
                            wf_out.setsampwidth(2)
                            wf_out.setframerate(params.framerate)
                            wf_out.writeframes(sliced_bytes)

                if standardize_wav_file(
                    out_wav_path,
                    pre_ms=DEFAULT_PRE_SILENCE_MS,
                    post_ms=DEFAULT_POST_SILENCE_MS,
                ):
                    return True
            except Exception as exc:
                logger.debug("Piper synthesize_to_wav attempt %d failed: %s", attempt, exc)
                time.sleep(0.08)
            finally:
                if p_piper and p_piper.poll() is None:
                    try:
                        p_piper.kill()
                    except Exception:
                        pass
        return False

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
        Unified Piper synthesis pipeline:
          - Single character -> cached char_*.wav.
          - Single isolated word -> disk cache brailce/audio/words_en/word_<w>.wav,
            built on demand via carrier-sentence slicing ("Please say: <word>.") so
            isolated function words ("is", "this", "his") and /w/, /wh/ onsets
            sound identical to in-sentence speech.
          - Multi-word sentence -> direct Piper synthesis with the same fixed VITS
            parameters and uniform RMS + silence standardization.
        """
        speech_text = normalize_for_piper(text)
        if not speech_text:
            return

        trimmed = speech_text.strip()

        # Single character / keystroke -> cached char_*.wav on worker queue
        if self.play_cached_character(trimmed):
            return

        words = speech_text.split()
        target_word = speech_text.strip().rstrip(".!?,;:")
        is_single_word = (len(words) == 1) and (len(target_word) > 1)

        if is_single_word:
            clean_word = target_word.lower()
            # 1. Play immediately from word disk cache if available (even if live piper binary missing)
            if self.play_cached_word(clean_word):
                return

        if not self.is_available():
            logger.warning("Piper not available, unable to speak: %s", text)
            return

        tmp_fd = None
        tmp_path = None
        try:
            audio_lock = AudioDeviceCoordinator.get_lock()
            with audio_lock:
                if is_single_word:
                    clean_word = "".join(ch for ch in target_word.lower() if ch.isalnum() or ch == "_")
                    if clean_word:
                        words_dir = _find_words_cache_dir(create=True)
                        cache_path = os.path.join(words_dir, f"word_{clean_word}.wav")
                        if self.synthesize_to_wav(target_word.lower(), cache_path, carrier_slice=True):
                            AudioDeviceCoordinator.play_wav_file(cache_path, owner="piper_word_cache", timeout=15.0)
                            return

                tmp_fd, tmp_path = tempfile.mkstemp(prefix="brailce_piper_", suffix=".wav")
                os.close(tmp_fd)
                tmp_fd = None

                synth_ok = self.synthesize_to_wav(
                    speech_text,
                    tmp_path,
                    carrier_slice=is_single_word,
                )
                if synth_ok:
                    AudioDeviceCoordinator.play_wav_file(tmp_path, owner="piper", timeout=30.0)
        finally:
            with self._lock:
                self._current_process = None
            AudioDeviceCoordinator.unregister_process()
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
