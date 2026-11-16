"""
Brailce – 100% Offline Grade 1 Filipino Playback Engine
=======================================================

Provides pure offline vocal feedback for Grade 1 Filipino (FIL) using ONE
consistent Filipino voice (sherpa-onnx MMS-TTS tgl VITS model):
1. Pre-rendered Studio Audio Bank (brailce/audio/filipino/word_<word>.wav,
   mode_*.wav, switched_*.wav, thinking_fil.wav, char_*.wav) generated from
   sherpa-onnx MMS-TTS tgl VITS and loudness/format-standardized to 22050 Hz mono s16.
2. Live offline synthesis via sherpa-onnx MMS-TTS tgl VITS (brailce/models/filipino/model.onnx)
   for any unseen Filipino sentence, math explanation, or AI response, cached to disk
   and played strictly through AudioDeviceCoordinator.play_wav_file().
3. Pure offline cached-word concatenation fallback if sherpa_onnx binary/model is missing
   (never falls back to online edge-tts or English Ryan phonetic respelling).
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import tempfile
import threading
import wave
from typing import Optional, List, Tuple, Any

from brailce.i18n.fil_lexicon import apply_filipino_lexicon
from brailce.speech.audio_device import AudioDeviceCoordinator
from brailce.speech.audio_normalize import (
    DEFAULT_POST_SILENCE_MS,
    DEFAULT_PRE_SILENCE_MS,
    TARGET_SAMPLE_RATE,
    resample_mono_pcm,
    standardize_wav_file,
)
from brailce.speech.base import SpeechEngine

logger = logging.getLogger(__name__)


def pitch_shift_pcm(samples: Any, semitones: float, sr: int = TARGET_SAMPLE_RATE) -> Any:
    """
    Fast, dependency-light phase-vocoder pitch shifter in pure numpy.
    Raises pitch by semitones (+3 to +4 semitones) while preserving duration and formants.
    """
    if abs(semitones) < 0.05 or getattr(samples, "size", 0) < 512:
        return samples
    try:
        import numpy as np

        factor = 2.0 ** (semitones / 12.0)
        target_len = samples.size
        n_fft = 1024
        hop = n_fft // 4
        synth_hop = int(round(hop / factor))
        if synth_hop < 1:
            return samples

        window = np.hanning(n_fft)
        padded = np.pad(samples.astype(np.float64), (0, n_fft), mode="constant")
        num_frames = samples.size // hop
        if num_frames < 2:
            return samples

        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(num_frames, n_fft),
            strides=(padded.strides[0] * hop, padded.strides[0]),
        )
        stft = np.fft.rfft(frames * window, axis=1)

        phase_adv = np.linspace(0, np.pi * 2 * hop, n_fft // 2 + 1)
        out_stft = np.zeros_like(stft)
        phase = np.angle(stft[0])
        out_stft[0] = stft[0]

        for i in range(1, num_frames):
            mag = np.abs(stft[i])
            dphase = np.angle(stft[i]) - np.angle(stft[i - 1]) - phase_adv
            dphase = dphase - 2 * np.pi * np.round(dphase / (2 * np.pi))
            freq = (phase_adv + dphase) / hop
            phase += freq * synth_hop
            out_stft[i] = mag * np.exp(1j * phase)

        out_len = num_frames * synth_hop + n_fft
        out_arr = np.zeros(out_len, dtype=np.float64)
        win_sum = np.zeros(out_len, dtype=np.float64)
        inv_frames = np.fft.irfft(out_stft, axis=1) * window

        for i in range(num_frames):
            st = i * synth_hop
            out_arr[st : st + n_fft] += inv_frames[i]
            win_sum[st : st + n_fft] += window ** 2

        mask = win_sum > 1e-4
        out_arr[mask] /= win_sum[mask]

        x_old = np.linspace(0, 1, out_arr.size)
        x_new = np.linspace(0, 1, target_len)
        resampled = np.interp(x_new, x_old, out_arr)
        return resampled
    except Exception as exc:
        logger.debug("pitch_shift_pcm error: %s", exc)
        return samples


def _pad_wav_with_silence(wav_path: str, pre_ms: int = 120, post_ms: int = 180) -> None:
    """Prepend digital zero PCM silence and append post-silence to WAV file with gentle envelope."""
    try:
        import numpy as np

        with wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(wf.getnframes())

        if params.sampwidth == 2 and len(raw) >= 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            # Gentle envelope on edges to eliminate click/pop
            fade_len = min(int(params.framerate * 0.005), arr.size // 4)
            if fade_len > 1:
                fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
                fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
                arr[:fade_len] *= fade_in
                arr[-fade_len:] *= fade_out
            pcm = np.clip(np.round(arr), -32767, 32767).astype(np.int16).tobytes()
        else:
            pcm = raw

        bytes_per_sample = params.sampwidth * params.nchannels
        pre_silence = b"\x00" * bytes_per_sample * int(params.framerate * (pre_ms / 1000.0))
        post_silence = b"\x00" * bytes_per_sample * int(params.framerate * (post_ms / 1000.0))

        with wave.open(wav_path, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(pre_silence + pcm + post_silence)
    except Exception as exc:
        logger.debug("Silence padding error: %s", exc)


def _find_filipino_audio_dir() -> str:
    """Locate the pre-rendered Filipino audio cache directory."""
    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "audio", "filipino")
    )
    if os.path.isdir(local_dir):
        return local_dir
    home_dir = os.path.expanduser(os.path.join("~", "brailce", "brailce", "audio", "filipino"))
    if os.path.isdir(home_dir):
        return home_dir
    return local_dir


FILIPINO_AUDIO_DIR = _find_filipino_audio_dir()


def _find_filipino_sherpa_model_dir() -> Optional[str]:
    """Locate the sherpa-onnx Filipino model directory."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "filipino")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "filipino")),
        os.path.expanduser(os.path.join("~", "brailce", "brailce", "models", "filipino")),
    ]
    for d in candidates:
        model_path = os.path.join(d, "model.onnx")
        tokens_path = os.path.join(d, "tokens.txt")
        if os.path.isfile(model_path) and os.path.isfile(tokens_path):
            return d
    return None


FILIPINO_CHAR_FILENAMES = {
    # Letters A-Z
    **{chr(c): f"char_{chr(c)}.wav" for c in range(ord("A"), ord("Z") + 1)},
    **{chr(c).lower(): f"char_{chr(c)}.wav" for c in range(ord("A"), ord("Z") + 1)},
    # Digits 0-9
    **{str(d): f"char_{d}.wav" for d in range(10)},
    # Math & Punctuation
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
    "'": "char_apostrophe.wav",
    ":": "char_colon.wav",
    ";": "char_semicolon.wav",
    " ": "char_space.wav",
}

FILIPINO_MODE_FILENAMES = {
    "free_input": "mode_free_input.wav",
    "math": "mode_math.wav",
    "ai_assistant": "mode_ai.wav",
    "ai": "mode_ai.wav",
    "scanner": "mode_scanner.wav",
}


def get_filipino_character_path(char: str) -> Optional[str]:
    """Return path to cached Filipino character WAV if it exists on disk."""
    if not char:
        return None
    clean = char.strip()
    fname = FILIPINO_CHAR_FILENAMES.get(clean)
    if not fname:
        upper = clean.upper()
        fname = FILIPINO_CHAR_FILENAMES.get(upper)
    if not fname:
        return None
    cache_dir = _find_filipino_audio_dir()
    wav_path = os.path.join(cache_dir, fname)
    if os.path.isfile(wav_path):
        return wav_path
    return None


def get_filipino_mode_path(mode: Any) -> Optional[str]:
    """Return path to cached Filipino mode announcement WAV if it exists on disk."""
    mode_name = str(mode).lower()
    if hasattr(mode, "name"):
        mode_name = mode.name.lower()
    for key, fname in FILIPINO_MODE_FILENAMES.items():
        if key in mode_name:
            cache_dir = _find_filipino_audio_dir()
            wav_path = os.path.join(cache_dir, fname)
            if os.path.isfile(wav_path):
                return wav_path
    return None


def get_filipino_word_path(word: str) -> Optional[str]:
    """
    Return path to cached Filipino word WAV if it exists on disk.
    Preserves internal hyphens (e.g. 'labing-isa', 'nag-iisip') and handles digits.
    """
    raw = word.strip().lower().rstrip(".,!?:;")
    if not raw:
        return None
    cache_dir = _find_filipino_audio_dir()
    # 1. Exact hyphen-preserving slug (e.g. 'labing-isa')
    slug_hyphen = re.sub(r"[^a-z0-9\-]", "", raw.replace("'", ""))
    # 2. Compact alphanumeric slug (e.g. 'dalawamput', 'labingisa')
    slug_plain = re.sub(r"[^a-z0-9]", "", raw)

    candidates = []
    if slug_hyphen:
        candidates.append(f"word_{slug_hyphen}.wav")
    if slug_plain and slug_plain != slug_hyphen:
        candidates.append(f"word_{slug_plain}.wav")

    # Map phonetic pronunciations to cached word WAVs
    phonetic_to_word = {
        "breyl": "braille",
        "breyls": "brailce",
        "greyd": "grade",
        "ey-ay": "ai",
        "eyay": "ai",
        "mowd": "mode",
        "iskaner": "scanner",
        "ispeys": "space",
        "bak-ispeys": "backspace",
        "bakispeys": "backspace",
    }
    orig = phonetic_to_word.get(slug_hyphen) or phonetic_to_word.get(slug_plain)
    if orig:
        candidates.append(f"word_{orig}.wav")

    for fname in candidates:
        wav_path = os.path.join(cache_dir, fname)
        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 44:
            return wav_path
    return None


def get_language_announcement_path(lang: str) -> Optional[str]:
    """Return path to language switch announcement WAV."""
    cache_dir = _find_filipino_audio_dir()
    filename = "switched_fil.wav" if lang.lower() == "fil" else "switched_en.wav"
    wav_path = os.path.join(cache_dir, filename)
    if os.path.isfile(wav_path):
        return wav_path
    return None


def get_thinking_audio_path() -> Optional[str]:
    """Return path to thinking_fil.wav if it exists on disk."""
    cache_dir = _find_filipino_audio_dir()
    wav_path = os.path.join(cache_dir, "thinking_fil.wav")
    if os.path.isfile(wav_path):
        return wav_path
    return None


def filipino_to_phonetic(text: str) -> str:
    """
    100% Offline Filipino-to-Phonetic Transcriber.
    Maps standard 26-letter A-Z Tagalog/Filipino words into phonetic vowel
    syllables so English Piper TTS pronounces Tagalog vowels with natural clarity.
    
    Rules:
      - "mga" -> "mang ah"
      - "ng" -> "nang"
      - Vowels:
          a -> ah
          e -> eh
          i -> ee
          o -> oh
          u -> oo
          ay -> eye
          oy -> oy
          aw -> ow
    """
    tokens = text.strip().split()
    converted_tokens: List[str] = []

    for raw_token in tokens:
        word = raw_token.strip().lower().rstrip(".,!?:;")
        if not word:
            continue

        # Keep "braille" intact as "braille" (no Tagalog phonetic mangling)
        if word == "braille":
            converted_tokens.append("braille")
            continue

        # Keep numbers/digits intact as numbers
        if word.isdigit():
            converted_tokens.append(word)
            continue

        # Common Tagalog grammatical markers
        if word == "mga":
            converted_tokens.append("mang ah")
            continue
        if word == "ng":
            converted_tokens.append("nang")
            continue
        if word == "ang":
            converted_tokens.append("ahng")
            continue
        if word == "sa":
            converted_tokens.append("sah")
            continue
        if word == "at":
            converted_tokens.append("aht")
            continue
        if word == "ay":
            converted_tokens.append("eye")
            continue

        # Diphthongs
        w = word
        w = re.sub(r"ay\b", "__EYE__", w)
        w = re.sub(r"oy\b", "__OY__", w)
        w = re.sub(r"aw\b", "__OW__", w)
        w = re.sub(r"iw\b", "__EEW__", w)

        # Tagalog vowel transcription
        # Map a, e, i, o, u to phonemic equivalents
        w = re.sub(r"a", "ah ", w)
        w = re.sub(r"e", "eh ", w)
        w = re.sub(r"i", "ee ", w)
        w = re.sub(r"o", "oh ", w)
        w = re.sub(r"u", "oo ", w)

        # Restore diphthongs
        w = w.replace("__EYE__", "eye")
        w = w.replace("__OY__", "oy")
        w = w.replace("__OW__", "ow")
        w = w.replace("__EEW__", "eew")

        # Clean spaces
        phonetic_word = " ".join(w.split())
        converted_tokens.append(phonetic_word)

    result = " ".join(converted_tokens).strip()
    if result and not result.endswith("."):
        result += "."
    return result


TAGALOG_ONES = {
    0: "sero",
    1: "isa",
    2: "dalawa",
    3: "tatlo",
    4: "apat",
    5: "lima",
    6: "anim",
    7: "pito",
    8: "walo",
    9: "siyam",
    10: "sampu",
}

TAGALOG_TEENS = {
    11: "labing-isa",
    12: "labindalawa",
    13: "labintatlo",
    14: "labing-apat",
    15: "labinlima",
    16: "labing-anim",
    17: "labimpito",
    18: "labing-walo",
    19: "labinsiyam",
}

TAGALOG_TENS = {
    2: "dalawampu",
    3: "tatlumpu",
    4: "apatnapu",
    5: "limampu",
    6: "animnapu",
    7: "pitumpu",
    8: "walumpu",
    9: "siyamnapu",
}

TAGALOG_HUNDREDS = {
    1: "isang daan",
    2: "dalawang daan",
    3: "tatlong daan",
    4: "apat na daan",
    5: "limang daan",
    6: "anim na daan",
    7: "pitong daan",
    8: "walong daan",
    9: "siyam na daan",
}


def tagalog_number(n: int) -> str:
    """Convert an integer 0-999,999 into natural Tagalog number words."""
    if n < 0:
        return f"negatibong {tagalog_number(-n)}"
    if n in TAGALOG_ONES:
        return TAGALOG_ONES[n]
    if n in TAGALOG_TEENS:
        return TAGALOG_TEENS[n]
    if 20 <= n <= 99:
        tens = n // 10
        rem = n % 10
        if rem == 0:
            return TAGALOG_TENS[tens]
        return f"{TAGALOG_TENS[tens]}'t {TAGALOG_ONES[rem]}"
    if 100 <= n <= 999:
        hundreds = n // 100
        rem = n % 100
        if rem == 0:
            return TAGALOG_HUNDREDS[hundreds]
        return f"{TAGALOG_HUNDREDS[hundreds]} at {tagalog_number(rem)}"
    if 1000 <= n <= 999999:
        thousands = n // 1000
        rem = n % 1000
        t_str = tagalog_number(thousands)
        if t_str[-1] in "aeiou":
            prefix = f"{t_str}ng libo"
        elif t_str.endswith("n"):
            prefix = f"{t_str}g libo"
        else:
            prefix = f"{t_str} na libo"
        if rem == 0:
            return prefix
        return f"{prefix} {tagalog_number(rem)}"
    return str(n)


def normalize_for_sherpa_filipino(text: str) -> str:
    """
    Normalize text for sherpa-onnx Filipino VITS TTS:
    - Applies Filipino pronunciation lexicon (loanwords, mode names, symbols).
    - Converts math operators: '+' -> 'dagdag', '=' -> 'ay', '-' -> 'bawas', etc.
    - Converts digits (0-999,999) into Tagalog number words (e.g. 7 -> pito, 14 -> labing-apat, 1000 -> isang libo).
    - Expands 'mga' -> 'manga' for natural phonetic realization.
    - Lowercases string (.lower()) so all characters match tokens.txt.
    - Strips unsupported characters not in tokens.txt.
    """
    if not text:
        return ""

    s = apply_filipino_lexicon(text)

    # 1. Convert math operators / symbols
    s = s.replace("+", " dagdag ")
    s = s.replace("=", " ay ")
    s = s.replace("×", " beses ").replace("*", " beses ")
    s = s.replace("÷", " hatiin sa ").replace("/", " hatiin sa ")
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " bawas ", s)
    s = re.sub(r"\s+-\s+", " bawas ", s)

    # 2. Respell 'mga' -> 'manga'
    s = re.sub(r"\bmga\b", "manga", s, flags=re.IGNORECASE)

    # 3. Convert digits 0-999,999 into Tagalog words
    def _replace_num(match: re.Match) -> str:
        val = int(match.group(0))
        if 0 <= val <= 999999:
            return f" {tagalog_number(val)} "
        return match.group(0)

    s = re.sub(r"\b\d{1,6}\b", _replace_num, s)

    # 4. Lowercase
    s = s.lower()

    # 5. Remove punctuation not present in tokens.txt
    s = re.sub(r"[^\w\s\-\'–]", " ", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


class FilipinoEngine:
    """
    Grade 1 Filipino Speech Controller.
    Synthesis hierarchy:
    1. Pre-cached word/character audio bank (brailce/audio/filipino/).
    2. Online Edge-TTS (fil-PH-AngeloNeural, 3.0s timeout).
    3. Offline Sherpa-ONNX (brailce/models/filipino/model.onnx, num_threads=4).
    4. Fallback to concatenative / phonetic Piper synthesis if model.onnx is missing.
    """

    def __init__(
        self,
        fallback_speech: Optional[SpeechEngine] = None,
        max_queue_size: int = 50,
        pitch_semitones: float = 3.5,
    ) -> None:
        self.fallback_speech = fallback_speech
        self.letter_style = "english_names"
        self.pitch_semitones = pitch_semitones
        self._sherpa_tts = None
        self._sherpa_initialized = False
        self._queue: queue.Queue[Optional[Tuple[str, Any]]] = queue.Queue(maxsize=max_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._queue_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        with self._queue_lock:
            if self._worker is None or not self._worker.is_alive():
                self._running = True
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name="filipino-speech-worker",
                )
                self._worker.start()

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, payload = item
            try:
                if kind == "wav":
                    AudioDeviceCoordinator.play_wav_file(payload, owner="filipino", timeout=30.0)
                elif kind == "char":
                    if len(payload) >= 3:
                        ch, use_fil, style = payload[0], payload[1], payload[2]
                    else:
                        ch, use_fil = payload[0], payload[1]
                        style = None
                    self.speak_character(ch, use_filipino_voice=use_fil, style=style)
                elif kind == "text":
                    self.speak_filipino(payload)
            except Exception as exc:
                logger.debug("Filipino worker error: %s", exc)

    def stop(self) -> None:
        """Clear pending Filipino utterances and interrupt active playback."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        AudioDeviceCoordinator.interrupt_active("filipino")

    def _enqueue(self, kind: str, payload: Any, interrupt: bool = False) -> None:
        self._ensure_worker()
        if interrupt:
            self.stop()
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            logger.warning("Filipino speech queue full — dropping item")

    def enqueue_text(self, text: str, interrupt: bool = False) -> None:
        """Non-blocking queued Filipino speech synthesis/playback."""
        if text and text.strip():
            self._enqueue("text", text, interrupt=interrupt)

    def enqueue_wav(self, wav_path: str, interrupt: bool = False) -> None:
        """Non-blocking queued playback of a pre-rendered WAV on the single Filipino worker."""
        if wav_path and os.path.isfile(wav_path):
            self._enqueue("wav", wav_path, interrupt=interrupt)

    def enqueue_character(self, character: str, use_filipino_voice: bool = True, style: Optional[str] = None) -> None:
        """Non-blocking queued character feedback."""
        if character:
            self._enqueue("char", (character, use_filipino_voice, style), interrupt=False)

    def play_announcement(self, lang: str) -> bool:
        """Play 'Wikang Filipino' or 'English language' announcement."""
        ann_path = get_language_announcement_path(lang)
        if ann_path and os.path.isfile(ann_path):
            return AudioDeviceCoordinator.play_wav_file(ann_path, owner="filipino", timeout=5.0)
        return False

    def play_thinking(self) -> bool:
        """Play 'Nag-iisip, mangyaring maghintay' thinking clip in the same Filipino voice."""
        thinking_path = get_thinking_audio_path()
        if thinking_path and os.path.isfile(thinking_path):
            return AudioDeviceCoordinator.play_wav_file(thinking_path, owner="filipino_thinking", timeout=6.0)
        return self.speak_filipino("Nag-iisip, mangyaring maghintay.")

    def speak_filipino(self, text: str) -> bool:
        """
        Speak Filipino text using ONE consistent offline Filipino voice
        (sherpa-onnx MMS-TTS tgl VITS + pre-rendered word bank):
        1. Single cached word -> play brailce/audio/filipino/word_<word>.wav directly.
        2. Multi-word or unseen text -> check hash cache sentence_<md5>.wav.
        3. Synthesize offline via sherpa-onnx MMS-TTS tgl VITS, normalize & cache to
           sentence_<md5>.wav, and play strictly through AudioDeviceCoordinator.
        4. If sherpa model is unavailable, concatenate cached Filipino word WAVs
           (never falling back to online edge-tts or English Ryan phonetic respelling).
        """
        if not text:
            return False

        trimmed = text.strip()
        if not trimmed:
            return False

        # 1. Single-word cache lookup (preserves hyphens like 'labing-isa')
        words = trimmed.split()
        if len(words) == 1:
            wav_path = get_filipino_word_path(words[0])
            if wav_path and os.path.isfile(wav_path):
                logger.info("FilipinoEngine: Playing cached word '%s' (%s)", words[0], wav_path)
                return AudioDeviceCoordinator.play_wav_file(wav_path, owner="filipino", timeout=10.0)

        # 2. Hash cache for full sentences / math explanations / AI replies
        norm_key = normalize_for_sherpa_filipino(trimmed) or trimmed.lower()
        h = hashlib.md5(norm_key.encode("utf-8")).hexdigest()
        cache_dir = _find_filipino_audio_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cached_sentence = os.path.join(cache_dir, f"sentence_{h}.wav")
        if os.path.isfile(cached_sentence) and os.path.getsize(cached_sentence) > 1000:
            logger.info("FilipinoEngine: Playing hash-cached sentence (%s)", cached_sentence)
            return AudioDeviceCoordinator.play_wav_file(cached_sentence, owner="filipino", timeout=30.0)

        # 3. Live offline synthesis via sherpa-onnx MMS-TTS tgl VITS
        if self.synthesize_to_wav(trimmed, cached_sentence):
            if os.path.isfile(cached_sentence) and os.path.getsize(cached_sentence) > 1000:
                return AudioDeviceCoordinator.play_wav_file(cached_sentence, owner="filipino", timeout=30.0)

        # 4. Offline word-bank concatenation (expand digits/ops first)
        expanded_words = normalize_for_sherpa_filipino(trimmed).split()
        if expanded_words and self._play_offline_concatenated(expanded_words):
            return True

        logger.warning("FilipinoEngine: Unable to synthesize offline Filipino speech for %r", trimmed)
        return False

    def _get_sherpa_tts(self):
        """Lazy loader for sherpa_onnx.OfflineTts using local Filipino MMS-TTS tgl VITS model."""
        if self._sherpa_initialized:
            return self._sherpa_tts

        self._sherpa_initialized = True
        model_dir = _find_filipino_sherpa_model_dir()
        if not model_dir:
            return None

        model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(model_path) or not os.path.isfile(tokens_path):
            return None

        try:
            import sherpa_onnx
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                tokens=tokens_path,
                data_dir="",
                noise_scale=0.333,
                noise_scale_w=0.4,
                length_scale=0.95,
            )
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=4,
                debug=False,
                provider="cpu",
            )
            tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)
            self._sherpa_tts = sherpa_onnx.OfflineTts(tts_config)
            logger.info("Initialized sherpa_onnx.OfflineTts (num_threads=4) with model: %s", model_path)
            return self._sherpa_tts
        except Exception as exc:
            logger.warning("Failed to initialize sherpa_onnx.OfflineTts: %s", exc)
            self._sherpa_tts = None
            return None

    def synthesize_to_wav(self, text: str, out_wav_path: str) -> bool:
        """Public alias to synthesize Filipino text cleanly to a standardized 22050Hz WAV."""
        return self._synthesize_sherpa_filipino(text, out_wav_path)

    def _synthesize_sherpa_filipino(self, text: str, out_wav_path: str) -> bool:
        """
        Synthesize Filipino text using sherpa-onnx OfflineTts (MMS-TTS tgl VITS),
        split gently on sentence boundaries for natural pacing, apply pitch shift (+3.5 st),
        then convert to 22050 Hz mono s16, RMS-normalized, with standard pre/post silence padding.
        """
        tts = self._get_sherpa_tts()
        if not tts:
            return False

        # Tokenize text into clauses and trailing punctuation delimiters
        tokens = re.split(r"([.!?;:\n,]+)", text)
        clauses_with_delims: List[Tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            clause = tokens[i].strip()
            delim = tokens[i + 1] if i + 1 < len(tokens) else ""
            if clause:
                clauses_with_delims.append((clause, delim))
            i += 2

        if not clauses_with_delims:
            return False

        try:
            import numpy as np

            all_chunks: List[np.ndarray] = []
            src_sr = 16000
            for idx, (clause, delim) in enumerate(clauses_with_delims):
                norm_clause = normalize_for_sherpa_filipino(clause)
                if not norm_clause:
                    continue
                # For short single letter names, keep speed slightly crisper to prevent stretching
                speed_val = 1.05 if len(norm_clause) <= 4 else 1.0
                audio = tts.generate(norm_clause, sid=0, speed=speed_val)
                if audio is None or len(audio.samples) == 0:
                    continue
                src_sr = audio.sample_rate
                chunk = np.asarray(audio.samples, dtype=np.float64) * 32767.0
                all_chunks.append(chunk)

                if idx < len(clauses_with_delims) - 1:
                    # Preserve prosodic timing: 400ms for terminal punctuation, 250ms for commas/semicolons
                    if any(c in delim for c in ".?!\n"):
                        pause_sec = 0.400
                    else:
                        pause_sec = 0.250
                    all_chunks.append(np.zeros(int(src_sr * pause_sec), dtype=np.float64))

            if not all_chunks:
                return False

            combined = np.concatenate(all_chunks)

            # Apply pitch shift (+3.5 semitones for friendlier, lighter offline voice)
            if getattr(self, "pitch_semitones", 0.0) != 0.0:
                combined = pitch_shift_pcm(combined, self.pitch_semitones, src_sr)

            pcm_int16 = np.clip(np.round(combined), -32767, 32767).astype(np.int16)

            os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
            with wave.open(out_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(src_sr)
                wf.writeframes(pcm_int16.tobytes())

            if os.path.isfile(out_wav_path) and os.path.getsize(out_wav_path) > 44:
                return standardize_wav_file(
                    out_wav_path,
                    target_sr=TARGET_SAMPLE_RATE,
                    pre_ms=DEFAULT_PRE_SILENCE_MS,
                    post_ms=DEFAULT_POST_SILENCE_MS,
                )
            return False
        except Exception as exc:
            logger.warning("sherpa_onnx generation error for text %r: %s", text, exc)
            return False

    def _play_offline_concatenated(self, words: List[str]) -> bool:
        """
        Concatenate cached Filipino word WAVs (preserving hyphens) with a 50ms gap
        and standardize loudness/sample rate. Zero English phonetic respelling!
        """
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="brailce_fil_concat_", suffix=".wav")
        os.close(tmp_fd)

        try:
            framerate = TARGET_SAMPLE_RATE
            gap_pcm = b"\x00\x00" * int(framerate * 0.050)
            combined_pcm = bytearray()

            for i, raw_word in enumerate(words):
                cached_wav = get_filipino_word_path(raw_word)
                pcm_data = None
                if cached_wav and os.path.isfile(cached_wav):
                    try:
                        with wave.open(cached_wav, "rb") as wf:
                            raw_frames = wf.readframes(wf.getnframes())
                        # Trim outer padding from cached word so stitch is tight
                        import numpy as np
                        arr = np.frombuffer(raw_frames, dtype=np.int16)
                        nz = np.where(np.abs(arr) > 150)[0]
                        if nz.size > 0:
                            pad = int(framerate * 0.025)
                            trimmed = arr[max(0, int(nz[0]) - pad) : min(arr.size, int(nz[-1]) + pad)]
                            pcm_data = trimmed.tobytes()
                        else:
                            pcm_data = raw_frames
                    except Exception:
                        pcm_data = None

                if pcm_data:
                    combined_pcm.extend(pcm_data)
                    if i < len(words) - 1:
                        combined_pcm.extend(gap_pcm)

            if not combined_pcm:
                return False

            with wave.open(tmp_path, "wb") as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(framerate)
                wf_out.writeframes(combined_pcm)

            standardize_wav_file(tmp_path, pre_ms=DEFAULT_PRE_SILENCE_MS, post_ms=DEFAULT_POST_SILENCE_MS)
            return AudioDeviceCoordinator.play_wav_file(tmp_path, owner="filipino", timeout=25.0)
        except Exception as exc:
            logger.warning("Error in offline Filipino concatenation: %s", exc)
            return False
        finally:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def speak_character(self, character: str, use_filipino_voice: bool = True, style: Optional[str] = None) -> bool:
        """
        Play character pronunciation in Filipino mode.
        When use_filipino_voice=True (default), uses pre-rendered Filipino character WAVs
        from the same Filipino voice (brailce/audio/filipino/char_*.wav), falling back
        to sherpa-onnx Tagalog name only if missing.
        """
        if use_filipino_voice:
            wav_path = get_filipino_character_path(character)
            if wav_path and os.path.isfile(wav_path):
                return AudioDeviceCoordinator.play_wav_file(wav_path, owner="filipino_char", timeout=5.0)
            from brailce.i18n.fil_keys import get_filipino_keystroke_name
            ch_name = get_filipino_keystroke_name(character, style=style or self.letter_style)
            if ch_name:
                return self.speak_filipino(ch_name)
        if self.fallback_speech and hasattr(self.fallback_speech, "_character_engine"):
            char_eng = self.fallback_speech._character_engine
            if hasattr(char_eng, "play_cached_character") and char_eng.play_cached_character(character):
                return True
        return False

    def speak_word(self, word: str) -> bool:
        """Play single Filipino word (delegates to speak_filipino)."""
        return self.speak_filipino(word)

    def speak_sentence(self, sentence: str) -> bool:
        """Speak a multi-word Filipino sentence (delegates to speak_filipino)."""
        return self.speak_filipino(sentence)
