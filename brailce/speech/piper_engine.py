"""
Brailce Piper Neural TTS Engine – Natural Human Voice
======================================================

High-fidelity, offline neural text-to-speech powered by Piper (VITS).
Runs locally on Raspberry Pi 5 with studio-grade human pronunciation,
fluid co-articulation, and natural intonation.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
from typing import Optional

from brailce.speech.base import SpeechEngine
from brailce.speech.espeak_engine import clean_speech_text

logger = logging.getLogger(__name__)


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
    # Preferred natural models in order of warmth
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
        for f in os.listdir(d):
            if f.endswith(".onnx"):
                return os.path.join(d, f)
    return None


class PiperEngine(SpeechEngine):
    """
    Piper Neural TTS Engine.
    
    Produces smooth, studio-quality human voice completely offline
    with negligible latency on Raspberry Pi 5.
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
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None

    def speak(self, text: str, interrupt: Optional[bool] = None) -> None:
        if not text:
            return

        trimmed = text.strip()
        if not trimmed:
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

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

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

        # ──────────────────────────────────────────────────────────────────
        # COMPREHENSIVE NATURAL ENGLISH WORD MAP
        # ──────────────────────────────────────────────────────────────────
        # Single letters, digits, and isolated symbols are converted into
        # clean, natural character pronunciations (without "the letter").
        # Period endings give Piper falling terminal pitch for natural cadence.
        # ──────────────────────────────────────────────────────────────────

        NATURAL_WORD_MAP = {
            # ── Letters (clean, straight to character, natural cadence) ──
            "A": "A.",
            "B": "Bee.",
            "C": "See.",
            "D": "Dee.",
            "E": "Ee.",
            "F": "Eff.",
            "G": "Jee.",
            "H": "Aitch.",
            "I": "Eye.",
            "J": "Jay.",
            "K": "Kay.",
            "L": "Ell.",
            "M": "M.",
            "N": "N.",
            "O": "Oh.",
            "P": "Pee.",
            "Q": "Cue.",
            "R": "Ar.",
            "S": "S.",
            "T": "Tee.",
            "U": "You.",
            "V": "Vee.",
            "W": "Double you.",
            "X": "Ex.",
            "Y": "Why.",
            "Z": "Zee.",
            # Also support spoken labels from mapping files / modes
            "AY": "A.",
            "BEE": "Bee.",
            "SEE": "See.",
            "DEE": "Dee.",
            "EE": "Ee.",
            "EFF": "Eff.",
            "EF": "Eff.",
            "JEE": "Jee.",
            "AITCH": "Aitch.",
            "EYE": "Eye.",
            "JAY": "Jay.",
            "KAY": "Kay.",
            "ELL": "Ell.",
            "EM": "M.",
            "EN": "N.",
            "OH": "Oh.",
            "PEE": "Pee.",
            "CUE": "Cue.",
            "ARE": "Ar.",
            "AR": "Ar.",
            "ESS": "S.",
            "TEE": "Tee.",
            "YOU": "You.",
            "VEE": "Vee.",
            "DOUBLE YOU": "Double you.",
            "EX": "Ex.",
            "WHY": "Why.",
            "ZEE": "Zee.",
            # ── Digits ──
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
            "ZERO": "zero.",
            "ONE": "one.",
            "TWO": "two.",
            "THREE": "three.",
            "FOUR": "four.",
            "FIVE": "five.",
            "SIX": "six.",
            "SEVEN": "seven.",
            "EIGHT": "eight.",
            "NINE": "nine.",
            # ── Math Operators ──
            "+": "plus.",
            "PLUS": "plus.",
            "-": "minus.",
            "MINUS": "minus.",
            "*": "multiplied by.",
            "TIMES": "multiplied by.",
            "/": "divided by.",
            "DIVIDED BY": "divided by.",
            "=": "equals.",
            "EQUALS": "equals.",
            # ── Punctuation ──
            "?": "question mark.",
            "QUESTION MARK": "question mark.",
            "!": "exclamation mark.",
            "EXCLAMATION MARK": "exclamation mark.",
            ".": "dot.",
            "PERIOD": "dot.",
            "DOT": "dot.",
            ",": "comma.",
            "COMMA": "comma.",
            "#": "number sign.",
            "NUMBER SIGN": "number sign.",
            ":": "colon.",
            "COLON": "colon.",
            ";": "semicolon.",
            "SEMICOLON": "semicolon.",
            "'": "apostrophe.",
            "APOSTROPHE": "apostrophe.",
            '"': "quote.",
            "QUOTE": "quote.",
            "QUOTATION MARK": "quote.",
            "(": "open parenthesis.",
            ")": "close parenthesis.",
            "@": "at sign.",
            "&": "and.",
            "%": "percent.",
            "$": "dollar sign.",
            "_": "underscore.",
            "~": "tilde.",
            "^": "caret.",
            "\\": "backslash.",
            "|": "vertical bar.",
            "{": "open brace.",
            "}": "close brace.",
            "[": "open bracket.",
            "]": "close bracket.",
            "<": "less than.",
            ">": "greater than.",
        }

        raw = text.strip()
        if not raw:
            return

        # ── Step 1: Single-character fast path ──
        if len(raw) == 1:
            key = raw.upper() if raw.isalpha() else raw
            speech_text = NATURAL_WORD_MAP.get(key, raw)
        else:
            # ── Step 2: Check if the whole string is a known speech label ──
            upper_check = raw.upper()
            if upper_check in NATURAL_WORD_MAP:
                speech_text = NATURAL_WORD_MAP[upper_check]
            else:
                # ── Step 3: For words / sentences, clean and ensure natural cadence ──
                speech_text = clean_speech_text(raw)
                if not speech_text:
                    return
                # If single word (like "cat"), lowercase and add period for full vowel pronunciation
                if " " not in speech_text:
                    speech_text = speech_text.lower()
                    if not speech_text.endswith((".", "!", "?")):
                        speech_text += "."

        # ──────────────────────────────────────────────────────────────────
        # Piper pipeline:
        #   echo <text> | piper --model model.onnx --output-raw | aplay …
        #
        # Use --length-scale 1.0 for natural, consistent American cadence
        # across typing, mode announcements, and AI responses.
        # ──────────────────────────────────────────────────────────────────
        cmd_piper = [
            self._binary,
            "--model", self._model,
            "--output-raw",
            "--length-scale", "1.0",
        ]

        import time
        for attempt in range(3):
            try:
                p_piper = subprocess.Popen(
                    cmd_piper,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                p_aplay = subprocess.Popen(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q", "-"],
                    stdin=p_piper.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_process = p_aplay

                if p_piper.stdin:
                    p_piper.stdin.write(speech_text.encode("utf-8") + b"\n")
                    p_piper.stdin.close()

                p_aplay.wait(timeout=15)
                p_piper.wait(timeout=5)

                if p_aplay.returncode == 0:
                    break
                else:
                    time.sleep(0.08)
            except Exception as e:
                logger.debug("Piper speech attempt %d failed: %s", attempt, e)
                time.sleep(0.08)
            finally:
                self._current_process = None
