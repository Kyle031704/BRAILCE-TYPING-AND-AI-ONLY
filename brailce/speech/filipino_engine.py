"""
Brailce – 100% Offline Grade 1 Filipino Playback Engine
=======================================================

Provides pure offline vocal feedback for Grade 1 Filipino (FIL):
1. Pre-cached Studio Audio Bank (brailce/audio/filipino/word_<word>.wav)
   generated via edge-tts (fil-PH-AngeloNeural).
2. Concatenative Sentence Synthesis: Stitches cached words with 60ms pauses
   into a seamless single aplay call.
3. Offline Phonetic Rule Engine: Converts untracked words into phonetic
   syllables for Piper Neural TTS offline fallback.
4. Language announcements: switched_fil.wav and switched_en.wav.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import wave
from typing import Optional, List, Tuple, Any

from brailce.speech.audio_device import AudioDeviceCoordinator
from brailce.speech.base import SpeechEngine

logger = logging.getLogger(__name__)


def _find_filipino_audio_dir() -> str:
    """Locate the pre-rendered Filipino audio cache directory."""
    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "audio", "filipino")
    )
    if os.path.isdir(local_dir):
        return local_dir
    pi_dir = "/home/raspberrypi5/brailce/brailce/audio/filipino"
    if os.path.isdir(pi_dir):
        return pi_dir
    return local_dir


FILIPINO_AUDIO_DIR = _find_filipino_audio_dir()


def _find_filipino_sherpa_model_dir() -> Optional[str]:
    """Locate the sherpa-onnx Filipino model directory."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "filipino")),
        "/home/raspberrypi5/brailce/brailce/models/filipino",
    ]
    for d in candidates:
        model_path = os.path.join(d, "model.onnx")
        tokens_path = os.path.join(d, "tokens.txt")
        if os.path.isfile(model_path) and os.path.isfile(tokens_path):
            return d
    return None

def _pad_wav_with_silence(wav_path: str, pre_ms: int = 250, post_ms: int = 350) -> None:
    """Prepend 250ms pure digital zero PCM silence and append 350ms post-silence to WAV file."""
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
    """Return path to cached word WAV if it exists on disk."""
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not clean:
        return None
    cache_dir = _find_filipino_audio_dir()
    wav_path = os.path.join(cache_dir, f"word_{clean}.wav")
    if os.path.isfile(wav_path):
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
    """Convert an integer 0-999 into Tagalog number words."""
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
    return str(n)


def normalize_for_sherpa_filipino(text: str) -> str:
    """
    Normalize text for sherpa-onnx Filipino VITS TTS:
    - Converts math operators: '+' -> 'dagdag', '=' -> 'ay', '-' -> 'bawas', etc.
    - Converts digits (0-999) into Tagalog number words (e.g. 7 -> pito, 14 -> labing-apat).
    - Lowercases string (.lower()) so all characters match tokens.txt.
    - Strips unsupported characters not in tokens.txt.
    """
    if not text:
        return ""

    s = text

    # 1. Convert math operators / symbols
    s = s.replace("+", " dagdag ")
    s = s.replace("=", " ay ")
    s = s.replace("×", " beses ").replace("*", " beses ")
    s = s.replace("÷", " hatiin sa ").replace("/", " hatiin sa ")
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " bawas ", s)
    s = re.sub(r"\s+-\s+", " bawas ", s)

    # 2. Convert digits 0-999 into Tagalog words
    def _replace_num(match: re.Match) -> str:
        val = int(match.group(0))
        if 0 <= val <= 999:
            return f" {tagalog_number(val)} "
        return match.group(0)

    s = re.sub(r"\b\d{1,3}\b", _replace_num, s)

    # 3. Lowercase
    s = s.lower()

    # 4. Remove punctuation not present in tokens.txt (periods, commas, question marks, colons, quotes)
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

    def __init__(self, fallback_speech: Optional[SpeechEngine] = None) -> None:
        self.fallback_speech = fallback_speech
        self._sherpa_tts = None
        self._sherpa_initialized = False

    def play_announcement(self, lang: str) -> bool:
        """Play 'Wikang Filipino' or 'English language' announcement."""
        ann_path = get_language_announcement_path(lang)
        if ann_path and os.path.isfile(ann_path):
            return AudioDeviceCoordinator.play_wav_file(ann_path, owner="filipino", timeout=5.0)
        return False

    def play_thinking(self) -> bool:
        """Play 'Nag-iisip, mangyaring maghintay' thinking clip."""
        thinking_path = get_thinking_audio_path()
        if thinking_path and os.path.isfile(thinking_path):
            return AudioDeviceCoordinator.play_wav_file(thinking_path, owner="filipino_thinking", timeout=6.0)
        if self.fallback_speech:
            self.fallback_speech.speak("Nag-iisip, mangyaring maghintay.", interrupt=False)
            return True
        return False

    def speak_filipino(self, text: str) -> bool:
        """
        Speak Filipino text:
        1. If single word in brailce/audio/filipino/word_*.wav, play immediately.
        2. If multi-word sentence or uncached word/response:
           - First try synthesizing with edge-tts (fil-PH-AngeloNeural, timeout 3.0s ->
             ffmpeg 22050Hz mono s16 WAV -> cache by hash in sentence_<md5>.wav -> aplay -q).
           - If offline and not yet cached: for each word, if word_<w>.wav exists, load its
             PCM frames; for uncached words, synthesize pure Filipino phonetic syllables to WAV,
             concatenate segments smoothly with a 50ms gap, and play via aplay -q!
        """
        if not text:
            return False

        trimmed = text.strip()
        if not trimmed:
            return False

        words = trimmed.split()
        if len(words) == 1:
            clean_word = re.sub(r"[^a-zA-Z]", "", words[0]).lower()
            wav_path = get_filipino_word_path(clean_word)
            if wav_path and os.path.isfile(wav_path):
                logger.info("FilipinoEngine: Playing cached word '%s' (%s)", clean_word, wav_path)
                return AudioDeviceCoordinator.play_wav_file(wav_path, owner="filipino", timeout=10.0)

        # Check hash cache for multi-word or uncached sentences/responses
        h = hashlib.md5(trimmed.lower().encode("utf-8")).hexdigest()
        cache_dir = _find_filipino_audio_dir()
        cached_sentence = os.path.join(cache_dir, f"sentence_{h}.wav")
        if os.path.isfile(cached_sentence) and os.path.getsize(cached_sentence) > 1000:
            logger.info("FilipinoEngine: Playing hash-cached sentence (%s)", cached_sentence)
            return AudioDeviceCoordinator.play_wav_file(cached_sentence, owner="filipino", timeout=30.0)

        # Try edge-tts synthesis online
        if self._try_edge_tts_synthesis(trimmed, cached_sentence, timeout=3.5):
            if os.path.isfile(cached_sentence) and os.path.getsize(cached_sentence) > 1000:
                logger.info("FilipinoEngine: Playing synthesized edge-tts sentence (%s)", cached_sentence)
                return AudioDeviceCoordinator.play_wav_file(cached_sentence, owner="filipino", timeout=30.0)

        # When Edge-TTS is unavailable/offline, first check if sherpa model.onnx exists on disk
        model_dir = _find_filipino_sherpa_model_dir()
        sherpa_model_path = os.path.join(model_dir, "model.onnx") if model_dir else None
        if sherpa_model_path and os.path.isfile(sherpa_model_path):
            logger.info("FilipinoEngine: Edge-TTS offline; synthesizing with sherpa_onnx (model.onnx)")
            if self._synthesize_sherpa_filipino(trimmed, cached_sentence):
                if os.path.isfile(cached_sentence) and os.path.getsize(cached_sentence) > 1000:
                    return AudioDeviceCoordinator.play_wav_file(cached_sentence, owner="filipino", timeout=30.0)

        # Keep existing phonetic Piper fallback only if brailce/models/filipino/model.onnx is not found on disk
        logger.info("FilipinoEngine: sherpa model not available on disk; falling back to phonetic Piper synthesis")
        if self._play_offline_concatenated(words):
            return True

        # Last resort: direct phonetic speech through fallback engine
        if self.fallback_speech:
            phonetic_sentence = filipino_to_phonetic(trimmed)
            self.fallback_speech.speak(phonetic_sentence, interrupt=False)
            return True

        return False

    def _get_sherpa_tts(self):
        """Lazy loader for sherpa_onnx.OfflineTts using local Filipino VITS model."""
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
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
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

    def _synthesize_sherpa_filipino(self, text: str, out_wav_path: str) -> bool:
        """Synthesize Filipino text using sherpa-onnx OfflineTts."""
        tts = self._get_sherpa_tts()
        if not tts:
            return False

        norm_text = normalize_for_sherpa_filipino(text)
        if not norm_text:
            return False

        try:
            audio = tts.generate(norm_text, sid=0, speed=1.0)
            if audio is None or len(audio.samples) == 0:
                return False

            sample_rate = audio.sample_rate
            samples = audio.samples

            # Write WAV file using sherpa_onnx helper if available
            try:
                import sherpa_onnx
                if hasattr(sherpa_onnx, "write_wave"):
                    sherpa_onnx.write_wave(out_wav_path, samples, sample_rate)
                    if os.path.isfile(out_wav_path) and os.path.getsize(out_wav_path) > 44:
                        _pad_wav_with_silence(out_wav_path, pre_ms=200, post_ms=100)
                        return True
            except Exception:
                pass

            try:
                import numpy as np
                pcm_bytes = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            except ImportError:
                import struct
                pcm_bytes = bytearray()
                for s in samples:
                    val = max(-32768, min(32767, int(s * 32767.0)))
                    pcm_bytes.extend(struct.pack("<h", val))

            with wave.open(out_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)

            if os.path.isfile(out_wav_path) and os.path.getsize(out_wav_path) > 44:
                _pad_wav_with_silence(out_wav_path, pre_ms=200, post_ms=100)
                return True
            return False
        except Exception as exc:
            logger.warning("sherpa_onnx generation error for text %r: %s", norm_text, exc)
            return False

    def _try_edge_tts_synthesis(self, text: str, out_wav_path: str, timeout: float = 3.0) -> bool:
        """
        Attempt online synthesis using edge-tts (fil-PH-AngeloNeural) with ffmpeg conversion.
        """
        import shutil
        import subprocess
        tmp_mp3 = out_wav_path + ".tmp.mp3"
        try:
            edge_bin = shutil.which("edge-tts")
            ffmpeg_bin = shutil.which("ffmpeg")
            if not edge_bin or not ffmpeg_bin:
                return False

            cmd_tts = [
                edge_bin,
                "--voice", "fil-PH-AngeloNeural",
                "--text", text,
                "--write-media", tmp_mp3,
            ]
            res = subprocess.run(cmd_tts, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not os.path.isfile(tmp_mp3) or os.path.getsize(tmp_mp3) == 0:
                return False

            cmd_ff = [
                ffmpeg_bin, "-y",
                "-i", tmp_mp3,
                "-ar", "22050",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                out_wav_path,
            ]
            ff_res = subprocess.run(cmd_ff, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ff_res.returncode == 0 and os.path.isfile(out_wav_path):
                _pad_wav_with_silence(out_wav_path, pre_ms=200, post_ms=100)
                return True
            return False
        except Exception as exc:
            logger.debug("Edge-TTS synthesis attempt failed: %s", exc)
            return False
        finally:
            if os.path.isfile(tmp_mp3):
                try:
                    os.remove(tmp_mp3)
                except Exception:
                    pass

    def _synthesize_phonetic_pcm(self, word: str) -> Optional[bytes]:
        """Synthesize pure Filipino phonetic syllables to PCM frames."""
        phonetic = filipino_to_phonetic(word)
        if not phonetic:
            return None

        piper_bin = None
        piper_model = None
        if self.fallback_speech:
            engine = getattr(self.fallback_speech, "_engine", self.fallback_speech)
            piper_bin = getattr(engine, "_binary", None)
            piper_model = getattr(engine, "_model", None)

        if not piper_bin or not piper_model:
            from brailce.speech.piper_engine import _find_piper_binary, _find_piper_model
            piper_bin = _find_piper_binary()
            piper_model = _find_piper_model()

        if piper_bin and piper_model and os.path.isfile(piper_bin) and os.path.isfile(piper_model):
            import subprocess
            t_fd, t_path = tempfile.mkstemp(prefix="brailce_fil_phon_", suffix=".wav")
            os.close(t_fd)
            try:
                cmd = [
                    piper_bin,
                    "--model", piper_model,
                    "--output_file", t_path,
                    "--length_scale", "1.08",
                    "--noise_scale", "0.5",
                    "--noise_w", "0.6",
                ]
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                p.communicate(f"{phonetic}.\n".encode("utf-8"), timeout=5.0)
                if os.path.isfile(t_path) and os.path.getsize(t_path) > 44:
                    with wave.open(t_path, "rb") as wf:
                        return wf.readframes(wf.getnframes())
            except Exception:
                pass
            finally:
                if os.path.isfile(t_path):
                    try:
                        os.remove(t_path)
                    except Exception:
                        pass
        return None

    def _play_offline_concatenated(self, words: List[str]) -> bool:
        """
        Concatenate cached word WAVs (or phonetic fallback WAVs) with a 50ms gap.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="brailce_fil_concat_", suffix=".wav")
        os.close(tmp_fd)

        try:
            framerate = 22050
            gap_pcm = b"\x00\x00" * int(framerate * 0.050)  # 50ms gap between segments
            pre_silence = b"\x00\x00" * int(framerate * 0.200)  # 200ms digital zero PCM
            post_silence = b"\x00\x00" * int(framerate * 0.100)  # 100ms digital zero PCM
            combined_pcm = bytearray()

            for i, raw_word in enumerate(words):
                w_clean = re.sub(r"[^a-zA-Z]", "", raw_word).lower()
                cached_wav = get_filipino_word_path(w_clean) if w_clean else None
                pcm_data = None

                if cached_wav and os.path.isfile(cached_wav):
                    try:
                        with wave.open(cached_wav, "rb") as wf:
                            pcm_data = wf.readframes(wf.getnframes())
                    except Exception:
                        pcm_data = None

                if not pcm_data:
                    pcm_data = self._synthesize_phonetic_pcm(raw_word)

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
                wf_out.writeframes(pre_silence + combined_pcm + post_silence)

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

    def speak_character(self, character: str) -> bool:
        """Play clean character pronunciation (routes to clean English letter audio)."""
        if self.fallback_speech and hasattr(self.fallback_speech, "speak_character"):
            self.fallback_speech.speak_character(character)
            return True
        return False

    def speak_word(self, word: str) -> bool:
        """Play single Filipino word (delegates to speak_filipino)."""
        return self.speak_filipino(word)

    def speak_sentence(self, sentence: str) -> bool:
        """Speak a multi-word Filipino sentence (delegates to speak_filipino)."""
        return self.speak_filipino(sentence)
