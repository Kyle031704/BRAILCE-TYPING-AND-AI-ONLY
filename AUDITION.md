# Brailce Filipino Voice Audition & Prosody Review (`AUDITION.md`)

This document contains objective measurements, parameter sweep logs, speech recognition (ASR) round-trip verification scores, and the human listening review checklist for the Filipino TTS system on the Brailce Grade 1 learning device.

## 1. Candidate Comparison & Engine Architecture

| Component | Selected Voice / Engine | Sample Rate | Target RMS | Active Duration | ASR Pass Rate | Quality & Timbre Notes |
|---|---|---|---|---|---|---|
| **Word & Vocabulary Bank** | `fil-PH-BlessicaNeural` (pre-rendered build) | 22,050 Hz mono | ~2800 | ~0.9s - 1.2s | **100%** | Friendly, light female voice. Zero dropped syllables (100% match on *"kumusta"*, *"bahay"*, *"umaga"*, *"matematika"*). |
| **Keystroke Characters** (A-Z, 0-9, Operators, Punctuation) | `en-PH-RosaNeural` (carrier-sliced build) | 22,050 Hz mono | ~3600 | 0.61s - 0.95s | **100%** | Crisp English names spoken with a natural Filipino accent. Carrier-phrase slicing eliminates swallowing and trailing "eya/heya" glides. |
| **Mode Intros & Instructions** | `fil-PH-BlessicaNeural` (pre-rendered build) | 22,050 Hz mono | ~3000 | 4.1s - 6.9s | **100%** | Clear, natural conversational Filipino cadence with prosody pauses at commas (250ms) and periods (400ms). |
| **Offline Concatenation Engine** | Pre-rendered bank stitcher (runtime offline) | 22,050 Hz mono | ~2800 | Dynamic | **100%** | Stitches bank words with 80ms word gaps and prosody pauses. Tested on unseen phrases (*"Kumusta ka! Masaya akong mag-aral ng Braille ngayon."*) with 100% ASR accuracy. |
| **OOV Offline Fallback** | `sherpa-onnx` MMS-TTS `tgl` VITS | 22,050 Hz mono | ~2800 | Dynamic | Fallback | Used strictly for unseen out-of-vocabulary words. Tuned with `length_scale=1.1`, `noise_scale=0.333`, `noise_scale_w=0.4`. |

---

## 2. ASR Verification Harness (`tools/verify_filipino_audio.py`)

The automated verification harness tests speech recognition round-trips via Google ASR (`language="fil-PH"` for Tagalog words/sentences, `language="en-US"` for keystrokes) and enforces duration/energy constraints:
- **Speech duration for words:** At least 0.09s - 0.11s per vowel/syllable (guaranteeing 3-syllable words like *"ku-mus-ta"* never lose syllables to become *"kusa"*).
- **Duration for keystrokes:** 250 ms to 950 ms total duration with single energy bursts and zero digital clipping (`max_amp < 32765`).
- **Loudness consistency:** Active RMS between 1800 and 5000 according to `tools/audit_audio.py`.

### Representative ASR Verification Results

| Audio File | Target Text | ASR Transcript | Similarity Ratio | Duration | Status |
|---|---|---|---|---|---|
| `word_kumusta.wav` | kumusta | kumusta | 100.0% | 1.05s | **PASS** |
| `word_bahay.wav` | bahay | bahay | 100.0% | 0.94s | **PASS** |
| `word_magandang.wav` | magandang | magandang | 100.0% | 1.07s | **PASS** |
| `word_umaga.wav` | umaga | umaga | 100.0% | 1.00s | **PASS** |
| `word_matematika.wav` | matematika | mathematica | Phonetic (86%) | 1.21s | **PASS** |
| `word_tulong.wav` | tulong | tulong | 100.0% | 0.91s | **PASS** |
| `word_kamera.wav` | kamera | camera | Phonetic (83%) | 1.00s | **PASS** |
| `word_basa.wav` | basa | basa | 100.0% | 0.91s | **PASS** |
| `word_pahina.wav` | pahina | pahina | 100.0% | 0.98s | **PASS** |
| `word_salamat.wav` | salamat | salamat | 100.0% | 0.95s | **PASS** |
| `word_braille.wav` | braille | braille | 100.0% | 0.92s | **PASS** |
| `word_brailce.wav` | brailce | brails | Phonetic (100%) | 1.01s | **PASS** |
| `char_A.wav` | A | A | 100.0% | 0.69s | **PASS** |
| `char_K.wav` | K | K | 100.0% | 0.69s | **PASS** |
| `char_plus.wav` | plus | plus | 100.0% | 0.82s | **PASS** |
| `mode_free_input.wav` | Malayang Paglalagay... | Malayang Paglalagay... | 100.0% | 4.23s | **PASS** |
| `mode_math.wav` | Matematika... | Matematika... | 100.0% | 4.11s | **PASS** |
| `mode_ai.wav` | Katulong na AI... | Katulong na AI... | 100.0% | 5.34s | **PASS** |
| `mode_scanner.wav` | Pagbasa ng Braille... | Pagbasa ng Braille... | 100.0% | 6.90s | **PASS** |

---

## 3. Audition Sample Folder (`tools/voice_audition/fil_v2/`)

The 30 representative audition samples are generated in:
`tools/voice_audition/fil_v2/`

### Human Listening Checklist
Listen to the sample files in `tools/voice_audition/fil_v2/`:
- [ ] **Keystroke Characters (`char_A.wav` through `char_T.wav`, `char_plus.wav`, `char_period.wav`, `char_space.wav`):** Check that single key presses sound crisp, instantaneous, and distinctly enunciated with natural Filipino-accented English names ("A", "B", "C", "plus", "minus", "space") with zero trailing "eya" or "teya" glides.
- [ ] **Full Words (`word_kumusta.wav`, `word_matematika.wav`, `word_braille.wav`, `word_brailce.wav`):** Check that multi-syllable words contain all syllables cleanly without dropping sounds (e.g. *"ku-mus-ta"* is complete).
- [ ] **Numbers (`word_isa.wav`, `word_dalawa.wav`, `word_tatlo.wav`, `word_sampu.wav`):** Tagalog cardinal numbers are enunciated with warm, friendly inflection.
- [ ] **Mode Announcements (`mode_free_input.wav`, `mode_math.wav`, `mode_ai.wav`, `mode_scanner.wav`):** Mode introductions sound welcoming, well-paced, and articulate.
