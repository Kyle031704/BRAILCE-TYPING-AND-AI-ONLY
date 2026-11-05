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

import logging
import os
import re
import tempfile
import wave
from typing import Optional, List, Tuple

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


class FilipinoEngine:
    """
    100% Offline Grade 1 Filipino Speech Controller.
    Routes single words and sentences to pre-cached human voice or concatenative
    synthesis with seamless fallback to offline Piper phonetic translation.
    """

    def __init__(self, fallback_speech: Optional[SpeechEngine] = None) -> None:
        self.fallback_speech = fallback_speech

    def play_announcement(self, lang: str) -> bool:
        """Play 'Wikang Filipino' or 'English language' announcement."""
        ann_path = get_language_announcement_path(lang)
        if ann_path and os.path.isfile(ann_path):
            return AudioDeviceCoordinator.play_wav_file(ann_path, owner="filipino", timeout=5.0)
        return False

    def speak_word(self, word: str) -> bool:
        """
        Play single Filipino word:
        1. Check if pre-cached word WAV exists.
        2. If missing, fall back to offline phonetic Piper synthesis.
        """
        clean = word.strip().rstrip(".,!?").lower()
        if not clean:
            return False

        wav_path = get_filipino_word_path(clean)
        if wav_path:
            logger.info("FilipinoEngine: Playing cached word '%s' (%s)", clean, wav_path)
            return AudioDeviceCoordinator.play_wav_file(wav_path, owner="filipino", timeout=10.0)

        # Offline phonetic fallback
        logger.info("FilipinoEngine: Word '%s' not cached, using phonetic fallback", clean)
        if self.fallback_speech:
            phonetic = filipino_to_phonetic(clean)
            self.fallback_speech.speak(phonetic, interrupt=False)
            return True

        return False

    def speak_sentence(self, sentence: str) -> bool:
        """
        Speak a multi-word Filipino sentence:
        1. If all words are cached, concatenate PCM frames with 60ms pauses
           and play in one smooth aplay call.
        2. If any word is missing, convert sentence to phonetic syllables and
           synthesize offline through Piper.
        """
        words = [w.strip().rstrip(".,!?") for w in sentence.strip().split() if w.strip()]
        if not words:
            return False

        if len(words) == 1:
            return self.speak_word(words[0])

        word_paths = [get_filipino_word_path(w) for w in words]
        all_cached = all(p is not None and os.path.isfile(p) for p in word_paths)

        if all_cached:
            logger.info("FilipinoEngine: Concatenating %d cached words for sentence: %r", len(words), sentence)
            return self._play_concatenated_sentence(word_paths)

        # Fallback to offline phonetic Piper synthesis
        logger.info("FilipinoEngine: Partial cache hit, routing sentence to phonetic Piper synthesis")
        if self.fallback_speech:
            phonetic_sentence = filipino_to_phonetic(sentence)
            self.fallback_speech.speak(phonetic_sentence, interrupt=False)
            return True

        return False

    def _play_concatenated_sentence(self, wav_paths: List[Optional[str]]) -> bool:
        """Concatenate cached word WAVs with 60ms pauses into a single WAV and play."""
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="brailce_fil_sentence_", suffix=".wav")
        os.close(tmp_fd)

        try:
            framerate = 22050
            pause_frames = b"\x00\x00" * int(framerate * 0.060)  # 60ms pause between words
            combined_pcm = bytearray()

            for i, p in enumerate(wav_paths):
                if not p or not os.path.isfile(p):
                    continue
                with wave.open(p, "rb") as wf:
                    framerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    combined_pcm.extend(frames)
                    if i < len(wav_paths) - 1:
                        combined_pcm.extend(pause_frames)

            # Write combined WAV
            with wave.open(tmp_path, "wb") as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(framerate)
                wf_out.writeframes(combined_pcm)

            return AudioDeviceCoordinator.play_wav_file(tmp_path, owner="filipino", timeout=20.0)
        except Exception as exc:
            logger.warning("Error concatenating Filipino sentence: %s", exc)
            return False
        finally:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
