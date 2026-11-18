"""
Brailce Filipino Letter Probe & Standardizer
============================================
Probes candidate respellings for A-Z with Edge TTS (Rosa/Blessica),
evaluates ASR (faster-whisper) + duration + F0 + spectral properties,
selects the best candidate per letter, normalizes intonation/timing,
saves results to tools/probe_results.json, updates brailce/audio/filipino/char_<L>.wav,
and generates tools/voice_audition/fil_letters/ (alphabet.wav + index.html).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import edge_tts
from faster_whisper import WhisperModel
from brailce.speech.audio_normalize import slice_carrier_audio

FIL_AUDIO_DIR = os.path.join(PROJECT_ROOT, "brailce", "audio", "filipino")
AUDITION_DIR = os.path.join(PROJECT_ROOT, "tools", "voice_audition", "fil_letters")
PROBE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "tools", "probe_results.json")
SR = 22050

# Candidate definitions for each letter A-Z
CANDIDATES: Dict[str, List[Dict[str, str]]] = {
    "A": [
        {"text": "Please say: A.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Please say the letter A.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Ay.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: ay.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "B": [
        {"text": "Please say: bee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Bee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: B.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "C": [
        {"text": "Please say: see.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "See.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: C.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "D": [
        {"text": "Please say: dee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Dee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: D.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "E": [
        {"text": "Please say: E.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Please say the letter E.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Please say: ee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "F": [
        {"text": "Please say: ef.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Ef.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: F.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "G": [
        {"text": "Please say: jee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Jee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: G.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "H": [
        {"text": "Please say: aych.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Aych.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: H.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "I": [
        {"text": "Please say: eye.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Eye.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: I.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "J": [
        {"text": "Please say: jay.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Jay.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: J.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "K": [
        {"text": "Please say: kay.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Kay.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: K.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "L": [
        {"text": "Please say: el.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "El.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: L.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "M": [
        {"text": "Please say: em.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Em.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: M.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "N": [
        {"text": "Please say: en.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "En.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: N.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "O": [
        {"text": "Please say: oh.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Please say: O.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Oh.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say the letter O.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "P": [
        {"text": "Please say: pee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Pee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: P.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "Q": [
        {"text": "Please say: kyoo.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Kyoo.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: Q.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "R": [
        {"text": "Please say: ar.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Ar.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: R.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "S": [
        {"text": "Please say: es.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Es.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: S.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "T": [
        {"text": "Please say: tee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Tee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: T.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "U": [
        {"text": "Please say: yoo.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Yoo.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: U.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "V": [
        {"text": "Please say: vee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Vee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: V.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "W": [
        {"text": "Please say: double-you.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Double-you.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: W.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "X": [
        {"text": "Please say: eks.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Eks.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: X.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "Y": [
        {"text": "Please say: why.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Why.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: Y.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
    "Z": [
        {"text": "Please say: zee.", "voice": "en-PH-RosaNeural", "type": "carrier"},
        {"text": "Zee.", "voice": "en-PH-RosaNeural", "type": "direct"},
        {"text": "Please say: Z.", "voice": "en-PH-RosaNeural", "type": "carrier"},
    ],
}

def estimate_pitch_track(sig: np.ndarray, sr: int = SR) -> Tuple[float, float, int]:
    """Return median F0, pitch slope (Hz/s), and count of voiced frames."""
    frame_len = int(sr * 0.030)
    hop = int(sr * 0.010)
    f0s: List[float] = []
    times: List[float] = []
    min_lag = int(sr / 450)  # max 450 Hz
    max_lag = int(sr / 90)   # min 90 Hz

    max_energy = np.max(np.abs(sig)) if sig.size > 0 else 0.0
    if max_energy < 1e-4:
        return 0.0, 0.0, 0

    for i in range(0, len(sig) - frame_len, hop):
        frame = sig[i : i + frame_len]
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < 0.06 * max_energy:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue
        peak_idx = np.argmax(search_region) + min_lag
        # parabolic interpolation
        if 0 < peak_idx < len(corr) - 1:
            alpha = corr[peak_idx - 1]
            beta = corr[peak_idx]
            gamma = corr[peak_idx + 1]
            denom = alpha - 2 * beta + gamma
            p = peak_idx + 0.5 * (alpha - gamma) / (denom + 1e-12) if abs(denom) > 1e-12 else peak_idx
        else:
            p = float(peak_idx)
        if corr[peak_idx] > 0.40 * corr[0]:
            f0 = sr / p
            f0s.append(f0)
            times.append(i / sr)

    if len(f0s) < 3:
        return 0.0, 0.0, len(f0s)
    med_f0 = float(np.median(f0s))
    slope, _ = np.polyfit(times, f0s, 1)
    return med_f0, float(slope), len(f0s)

def measure_hf_ratio(sig: np.ndarray, sr: int = SR) -> float:
    """Ratio of energy above 2.5 kHz to total energy."""
    if sig.size == 0 or np.max(np.abs(sig)) < 1e-4:
        return 0.0
    freqs, psd = scipy.signal.welch(sig, sr, nperseg=min(512, len(sig)))
    total_e = np.sum(psd)
    if total_e <= 0:
        return 0.0
    hf_e = np.sum(psd[freqs >= 2500.0])
    return float(hf_e / total_e)

def trim_and_pad(
    sig: np.ndarray,
    sr: int = SR,
    target_rms: float = 4000.0,
    lead_ms: int = 15,
    tail_ms: int = 80,
    fade_ms: int = 8,
) -> np.ndarray:
    """Trim silence to active speech and apply fixed lead/tail and smooth fades."""
    if sig.size == 0:
        return sig
    abs_sig = np.abs(sig)
    peak = np.max(abs_sig)
    if peak < 1e-4:
        return sig

    thresh = max(0.03 * peak, 0.01)
    active_idx = np.where(abs_sig > thresh)[0]
    if len(active_idx) == 0:
        return sig

    start = active_idx[0]
    end = active_idx[-1] + 1
    active_clip = sig[start:end]

    # Active RMS normalization
    active_rms = np.sqrt(np.mean(active_clip ** 2))
    scale = (target_rms / 32768.0) / max(active_rms, 1e-6)
    scaled = active_clip * scale
    max_p = np.max(np.abs(scaled))
    if max_p > 0.95:
        scaled = scaled * (0.95 / max_p)

    # Lead and tail
    lead_samples = int(sr * lead_ms / 1000.0)
    tail_samples = int(sr * tail_ms / 1000.0)
    out = np.concatenate([np.zeros(lead_samples), scaled, np.zeros(tail_samples)])

    # Fade-in and fade-out
    fade_len = int(sr * fade_ms / 1000.0)
    if len(out) > 2 * fade_len:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        out[lead_samples : lead_samples + fade_len] *= fade_in
        end_core = lead_samples + len(scaled)
        out[end_core - fade_len : end_core] *= fade_out

    return out

async def synthesize_edge(text: str, voice: str) -> np.ndarray:
    comm = edge_tts.Communicate(text, voice)
    data = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    sig, sr = sf.read(io.BytesIO(data))
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != SR:
        num = int(round(len(sig) * SR / sr))
        sig = scipy.signal.resample(sig, num)
    return sig

async def probe_all_letters():
    print("Loading faster-whisper (tiny.en, cpu, int8)...")
    asr = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    chosen_candidates: Dict[str, Dict] = {}
    audition_audio: Dict[str, np.ndarray] = {}

    print(f"Probing {len(CANDIDATES)} letters...")
    for letter, cand_list in sorted(CANDIDATES.items()):
        results = []
        for cand in cand_list:
            raw = await synthesize_edge(cand["text"], cand["voice"])
            if cand["type"] == "carrier":
                raw16 = (raw * 32767.0).astype(np.int16)
                sliced_int = slice_carrier_audio(raw16, SR)
                sig = np.array(sliced_int, dtype=np.float32) / 32767.0
            else:
                sig = raw

            med_f0, slope, voiced = estimate_pitch_track(sig)
            hf = measure_hf_ratio(sig)
            dur = len(sig) / SR

            # ASR check
            segs, _ = asr.transcribe(sig)
            trans = " ".join(s.text.strip() for s in segs)

            # Score candidate: target name matching, reasonable duration (0.25-0.55s), voiced
            is_voiced = voiced >= 5
            score = 0
            t_low = trans.lower()
            l_low = letter.lower()
            if l_low in t_low or (letter == "W" and "double" in t_low) or (letter == "O" and "o" in t_low):
                score += 10
            if 0.28 <= dur <= (0.65 if letter == "W" else 0.52):
                score += 5
            if letter not in ["S", "X", "H", "F", "C"] and is_voiced:
                score += 5
            if letter in ["S", "X", "H", "F", "C"]:
                score += 3

            results.append({
                "cand": cand,
                "dur": dur,
                "f0": med_f0,
                "slope": slope,
                "voiced": voiced,
                "hf": hf,
                "trans": trans,
                "score": score,
                "sig": sig,
            })

        # Pick highest score, break ties with duration closest to 0.38s
        best = max(results, key=lambda r: (r["score"], -abs(r["dur"] - 0.38)))
        print(f"[{letter}] Picked: '{best['cand']['text']}' | dur={best['dur']:.2f}s | F0={best['f0']:.0f}Hz | ASR='{best['trans']}' (score {best['score']})")

        # Standardize clip
        clean_sig = trim_and_pad(best["sig"], SR, target_rms=4000.0, lead_ms=15, tail_ms=80, fade_ms=8)
        chosen_candidates[letter] = {
            "text": best["cand"]["text"],
            "voice": best["cand"]["voice"],
            "type": best["cand"]["type"],
            "dur": len(clean_sig) / SR,
            "f0": best["f0"],
            "slope": best["slope"],
            "voiced": best["voiced"],
            "hf": best["hf"],
            "trans": best["trans"],
        }
        audition_audio[letter] = clean_sig

        # Save to brailce/audio/filipino/char_<L>.wav
        out_wav = os.path.join(FIL_AUDIO_DIR, f"char_{letter}.wav")
        sf.write(out_wav, clean_sig, SR, subtype="PCM_16")

    # Save probe results JSON
    with open(PROBE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen_candidates, f, indent=2)
    print(f"Saved probe results to {PROBE_RESULTS_PATH}")

    # Build tools/voice_audition/fil_letters/
    os.makedirs(AUDITION_DIR, exist_ok=True)
    alphabet_pieces = []
    silence_gap = np.zeros(int(SR * 0.400), dtype=np.float32)
    for letter in sorted(audition_audio.keys()):
        alphabet_pieces.append(audition_audio[letter])
        alphabet_pieces.append(silence_gap)
    full_alphabet = np.concatenate(alphabet_pieces)
    sf.write(os.path.join(AUDITION_DIR, "alphabet.wav"), full_alphabet, SR, subtype="PCM_16")

    # Generate index.html
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Filipino Letters Audition</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 24px; background: #f8f9fa; }",
        "h1 { color: #202124; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; margin-top: 16px; }",
        ".card { background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }",
        ".letter { font-size: 28px; font-weight: bold; margin-bottom: 6px; }",
        ".meta { font-size: 11px; color: #5f6368; margin-bottom: 8px; }",
        "audio { width: 100%; height: 32px; }",
        ".full { margin: 20px 0; padding: 16px; background: #e8f0fe; border-radius: 8px; }",
        "</style></head><body>",
        "<h1>Filipino Letters (A-Z) Audio Audition</h1>",
        "<div class='full'><h3>Full Alphabet (A to Z with 400ms gaps)</h3>",
        "<audio controls src='alphabet.wav'></audio></div>",
        "<div class='grid'>",
    ]
    for letter, meta in sorted(chosen_candidates.items()):
        # Copy individual file to audition folder
        card_wav = os.path.join(AUDITION_DIR, f"char_{letter}.wav")
        sf.write(card_wav, audition_audio[letter], SR, subtype="PCM_16")
        html.append(
            f"<div class='card'><div class='letter'>{letter}</div>"
            f"<div class='meta'>{meta['dur']:.2f}s | {meta['f0']:.0f}Hz<br>'{meta['trans']}'</div>"
            f"<audio controls src='char_{letter}.wav'></audio></div>"
        )
    html.append("</div></body></html>")

    with open(os.path.join(AUDITION_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Generated audition index in {AUDITION_DIR}")

if __name__ == "__main__":
    asyncio.run(probe_all_letters())
