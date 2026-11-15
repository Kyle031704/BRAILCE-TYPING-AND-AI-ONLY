# Brailce Filipino Voice Audition & Prosody Review

This document contains objective measurements, parameter sweep logs, and the human listening review checklist for the Filipino TTS candidates on the Brailce Grade 1 learning device.

## 1. Candidate Comparison Summary

| Candidate Voice | Type | Sample Rate | Encoding | Median RMS | Peak Level | Clipping (%) | Status / Recommendation |
|---|---|---|---|---|---|---|---|
| **sherpa_mms_tgl** | Local Offline VITS | 22,050 Hz | Mono 16-bit PCM | 3364.1 | 18539 / 32767 | 0.000% | **Primary Offline Engine** (Zero internet dependency, runs locally on Pi 5) |
| **fil-PH-AngeloNeural** | Pre-rendered Build Voice | 22,050 Hz | Mono 16-bit PCM | 3014.8 | 16951 / 32767 | 0.000% | **Pre-rendered Word/Keystroke Bank** (Used for offline instant cache on Pi) |
| **fil-PH-BlessicaNeural** | Alternative Pre-rendered | 22,050 Hz | Mono 16-bit PCM | 2957.9 | 21363 / 32767 | 0.000% | Alternative friendly candidate (Audition available in `tools/voice_audition/`) |

---

## 2. Audio Processing & Normalization Standards
All audio assets generated and used by Brailce comply with `tools/audit_audio.py`:
- **Sample Rate:** Exactly 22,050 Hz.
- **Channels:** Single-channel Mono.
- **Bit Depth:** Signed 16-bit PCM (s16le).
- **Target RMS:** $2800 \pm 400$ (normalized to prevent loud jumps or whisper-quiet playback).
- **Pre/Post Silence Padding:** 35 ms leading silence, 45 ms trailing silence to avoid DAC click/pop artifacts on Raspberry Pi USB and I2S DACs.

---

## 3. Prosody & Text Normalizer Fixes
- **Punctuation Timing:**
  - Comma `,`, semicolon `;`, and colon `:`: 250 ms silence pause.
  - Period `.`, exclamation mark `!`, question mark `?`, and newlines: 400 ms silence pause.
- **Grammar & Lexicon:**
  - Plural marker `mga` respelled as `"manga"` to ensure natural phonetic realization.
  - Numbers from $0$ to $999,999$ expanded via `tagalog_number()` into natural spoken Tagalog (e.g. $25 \to$ *"dalawampu't lima"*, $1000 \to$ *"isang libo"*, $25000 \to$ *"dalawampu't limang libo"*).
  - Single-word typing feedback isolated with smooth envelope shaping to eliminate abrupt vowel onsets.

---

## 4. Human Audition Review Checklist
Listen to the sample files in `tools/voice_audition/` on standard headphones or speakers:

- [ ] **Letter & Keystroke Pronunciation:** Modern Filipino classroom names (ey, bi, si, di, ...) sound clear and distinct without clipping.
- [ ] **Number Reading:** Double-check compound numbers like 25 (*dalawampu't lima*) and 1234 (*isang libo dalawang daan at tatlumpu't apat*).
- [ ] **Punctuation Timing:** Check that multi-clause sentences pause naturally at commas (250ms) and stop at periods (400ms).
- [ ] **Loudness Consistency:** Keystroke clicks, word readouts, and mode switch announcements play at equal perceptual volume.
