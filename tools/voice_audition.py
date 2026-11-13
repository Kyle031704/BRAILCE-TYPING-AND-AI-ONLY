"""
Brailce Filipino Voice Audition Tool (`tools/voice_audition.py`)
===============================================================

Synthesizes the standard 12 audition phrases across candidate Filipino voices
and saves them to `tools/voice_audition/<candidate>/` for comparative listening:
1. `fil-PH-BlessicaNeural` (Warm, natural Filipino female voice - Recommended for Grade 1)
2. `fil-PH-AngeloNeural` (Clear Filipino male neural voice)
3. `sherpa_mms_tgl` (Offline MMS-TTS tgl VITS voice, if model.onnx is present)

Each phrase is loudness-standardized to 22050 Hz mono s16 PCM.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import wave

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brailce.speech.audio_normalize import (
    DEFAULT_POST_SILENCE_MS,
    DEFAULT_PRE_SILENCE_MS,
    TARGET_SAMPLE_RATE,
    standardize_wav_file,
)
from brailce.speech.filipino_engine import _find_filipino_sherpa_model_dir, normalize_for_sherpa_filipino

AUDITION_DIR = os.path.join(PROJECT_ROOT, "tools", "voice_audition")

TEST_PHRASES = [
    ("01_letter_a", "ey"),
    ("02_digit_two", "dalawa"),
    ("03_operator_plus", "dagdag"),
    ("04_intro_free_input", "Malayang Pagpindot. Pindutin ang mga titik at salita."),
    ("05_intro_math", "Modong Matematika. Pindutin ang mga numero at operasyon."),
    ("06_intro_ai", "Modong Katulong na AI. Pindutin ang iyong tanong, tapos pindutin ang Enter."),
    ("07_intro_scanner", "Modong Panuri ng Braille. Itapat ang pahina, tapos pindutin ang Enter upang basahin."),
    ("08_math_explanation", "Ang 2 dagdag 3 ay 5. Magsimula sa 2, magbilang ng 3 pa: 3, 4, 5. Ang sagot ay 5."),
    ("09_ai_answer", "Ano ang araw? Ang araw ay ang bituin sa gitna ng ating sistemang solar."),
    ("10_exclamation", "Magaling!"),
    ("11_scanner_prompt", "Nakitang Braille. Pindutin ang Enter upang basahin."),
    ("12_question", "Ano ang Braille?"),
]


async def _synthesize_edge(voice: str, text: str, out_wav: str) -> bool:
    """Generate audio via edge-tts."""
    try:
        import edge_tts
        temp_mp3 = out_wav + ".tmp.mp3"
        communicate = edge_tts.Communicate(text, voice, pitch="-1Hz", rate="-2%")
        await communicate.save(temp_mp3)

        if not os.path.isfile(temp_mp3) or os.path.getsize(temp_mp3) == 0:
            return False

        # Convert to 22050Hz mono s16 WAV via soundfile / pydub / miniaudio / PIL / scipy / python
        try:
            import soundfile as sf
            data, sr = sf.read(temp_mp3)
            sf.write(out_wav, data, sr)
        except Exception:
            # Fallback to ffmpeg if present
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", temp_mp3, "-ar", "22050", "-ac", "1", out_wav],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if os.path.isfile(temp_mp3):
            os.remove(temp_mp3)

        if os.path.isfile(out_wav):
            standardize_wav_file(out_wav, target_sr=TARGET_SAMPLE_RATE, pre_ms=DEFAULT_PRE_SILENCE_MS, post_ms=DEFAULT_POST_SILENCE_MS)
            return True
        return False
    except Exception as exc:
        print(f"Edge-TTS synthesis error for {voice}: {exc}", file=sys.stderr)
        return False


def _synthesize_sherpa(text: str, out_wav: str) -> bool:
    """Generate audio via local sherpa-onnx MMS-TTS tgl if available."""
    model_dir = _find_filipino_sherpa_model_dir()
    if not model_dir:
        return False
    try:
        import sherpa_onnx
        import numpy as np
        model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        if not (os.path.isfile(model_path) and os.path.isfile(tokens_path)):
            return False

        vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=model_path, tokens=tokens_path, length_scale=1.08)
        model_cfg = sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4)
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_cfg))

        norm_text = normalize_for_sherpa_filipino(text)
        audio = tts.generate(norm_text, sid=0, speed=0.92)
        if audio is None or len(audio.samples) == 0:
            return False

        pcm = np.clip(np.round(np.asarray(audio.samples, dtype=np.float64) * 32767.0), -32767, 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(audio.sample_rate)
            wf.writeframes(pcm.tobytes())

        standardize_wav_file(out_wav, target_sr=TARGET_SAMPLE_RATE, pre_ms=DEFAULT_PRE_SILENCE_MS, post_ms=DEFAULT_POST_SILENCE_MS)
        return True
    except Exception as exc:
        print(f"Sherpa-ONNX synthesis error: {exc}", file=sys.stderr)
        return False


async def run_audition(candidate: str) -> None:
    cand_dir = os.path.join(AUDITION_DIR, candidate)
    os.makedirs(cand_dir, exist_ok=True)
    print(f"\n--- Auditioning: {candidate} -> {cand_dir} ---")

    for key, phrase in TEST_PHRASES:
        out_file = os.path.join(cand_dir, f"{key}.wav")
        success = False
        if "Blessica" in candidate:
            success = await _synthesize_edge("fil-PH-BlessicaNeural", phrase, out_file)
        elif "Angelo" in candidate:
            success = await _synthesize_edge("fil-PH-AngeloNeural", phrase, out_file)
        elif "sherpa" in candidate or "mms" in candidate:
            success = _synthesize_sherpa(phrase, out_file)

        status = "OK" if success and os.path.isfile(out_file) else "FAILED/SKIPPED"
        print(f"  [{status}] {key}: {phrase[:40]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Audition Filipino TTS candidates.")
    parser.add_argument("--candidate", default="all", choices=["all", "blessica", "angelo", "sherpa_mms_tgl"])
    args = parser.parse_args()

    cands = ["blessica", "angelo", "sherpa_mms_tgl"] if args.candidate == "all" else [args.candidate]
    for c in cands:
        name = "fil-PH-BlessicaNeural" if c == "blessica" else ("fil-PH-AngeloNeural" if c == "angelo" else "sherpa_mms_tgl")
        await run_audition(name)


if __name__ == "__main__":
    asyncio.run(main())

