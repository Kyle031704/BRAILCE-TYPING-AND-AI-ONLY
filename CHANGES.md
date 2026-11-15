# Brailce System Fixes & Architectural Changes (`CHANGES.md`)

## 1. Confirmed Root Causes & Fixes by Area

### Global Cleanup, Configuration & Bilingual i18n
- **Root cause:** Hardcoded Windows paths, Raspberry Pi IP addresses, SSH credentials, and host keys were scattered across `brailce/config/config.json`, root `deploy_to_pi.py`, `tools/deploy_to_pi.py`, `tools/generate_character_cache.py`, `piper_engine.py`, `filipino_engine.py`, and ~12 stale one-off helper scripts under `tools/`.
- **Fix:**
  - Removed hardcoded credentials/IPs/paths from tracked source and config (`vision.test_image_path` cleared to `""`).
  - Deleted stale one-off `tools/` scripts and stale root `deploy_to_pi.py`; updated `tools/deploy_to_pi.py` and `tools/generate_character_cache.py` to read `PI_IP`, `PI_USER`, `PI_PASS`, `PI_REMOTE_DIR`, and `PI_HOSTKEY` from `.env` / environment variables (documented in `.env.example`, ignored via `.gitignore`).
  - Added unified `brailce/i18n/strings.py` (`STRINGS`, `get_string()`, `get_mode_intro()`, `get_mode_name()`) so LCD display text and spoken TTS prompts use the exact same localized `"en"` and `"fil"` strings.

---

### A — Consistent English Word Pronunciation (`Free Input`, `Math`, `AI`, `Scanner`)
- **Root causes:**
  1. Isolated function words (`is`, `his`, `this`, `the`, `was`, `what`, `when`, `where`, `why`, `who`) were routed through inconsistent synthesis calls and hit Piper VITS boundary swallowing at `t = 0` plus `MagicMock` / `hasattr` branching in `free_input.py`.
  2. Multi-parameter mismatches existed across engines instead of one consistent `en_US-ryan-medium` voice configuration.
- **Fix & Rationale (Option 1 vs Option 2):**
  - Locked `PiperEngine` to a single VITS parameter set (`length_scale=1.05`, `noise_scale=0.5`, `noise_w=0.6`) across both isolated words and sentences so timbre stays uniform.
  - Selected **Option 1 (Carrier-Phrase Slicing + Pre-rendered Cache)** over purely raw single-word synthesis with ad-hoc glottal hacks: `SpeechService.speak_word_clean()` resolves lowercase words via `brailce/audio/words_en/<word>.wav` first, and on a miss synthesizes `"Please say: <word>."`, slices cleanly after the colon silence gap via `brailce/speech/audio_normalize.py` (`slice_carrier_audio()`), normalizes active speech RMS (`TARGET_ACTIVE_RMS = 4200.0`, peak cap `30000.0`), and pads with `120 ms` pre / `180 ms` post zero silence.
  - Removed `MagicMock` / `Mock` hacks in `brailce/modes/free_input.py`, `math_mode.py`, and `ai_assistant.py`, unifying all playback on `SpeechService` and `AudioDeviceCoordinator.play_wav_file()` (`aplay -q`).

---

### B — Filipino Math Mode & Arithmetic Visualization
- **Root causes:**
  1. `SafeArithmetic.solve()` generated hardcoded English step/counting explanations even in `"fil"` mode, and Filipino keypad/mode/math card LCD headers remained in English or clipped on `480x320` ILI9488 displays.
  2. `brailce/audio/filipino/word_oo.wav` was silent/near-empty (low RMS).
- **Fix:**
  - Implemented `SafeArithmetic.solve(expr, language=...)` with Tagalog `0–999` number-to-word conversion (including `-ng` / `na` linkers), Filipino Grade 1 counting steps/explanations, and localized error keys (`MATH_ERR_*`) sourced from `brailce/i18n/strings.py`.
  - Updated `ConsoleDisplay` and `ILI9488Display.show_math_solution()` with localized titles (`MATH_CARD_TITLE`, `MATH_LABEL_*`) and 48-character multi-line word wrapping to prevent LCD clipping on `480x320`.
  - Regenerated the Filipino audio cache and normalized `word_oo.wav` and all sibling WAVs uniformly.

---

### C — Accurate Offline Filipino AI Assistant (`lfm2.5-1.2b-instruct`)
- **Root causes:**
  1. Weak Filipino system prompting allowed the `1.2B` model to slip into English paragraphs or emit markdown/emojis/loanword clutter.
  2. Spoken responses read hyphens inconsistently or choked on unexpanded numbers in Filipino TTS.
- **Fix:**
  - Upgraded `FILIPINO_SYSTEM_PROMPT` in `brailce/ai/ollama_client.py` with strict 3-example Tagalog few-shot prompting, plus `is_mostly_english()` detection with one automatic retry and localized `AI_EMPTY_FALLBACK` safety net.
  - Enhanced `clean_ai_response()` to strip markdown/emojis/special tokens and convert hyphens inside words into spaces for clean TTS articulation while preserving natural Braille terms (`Braille`, `Brailce`).
  - Pre-cleaned offline factual responses (`FACTUAL_QA_RESPONSES`, `FILIPINO_AI_OFFLINE_RESPONSES`) in pure Grade 1 Filipino.

---

### D — Single Consistent Voice per Language (Zero Online `edge-tts` / Zero Mixed Ryan-Filipino)
- **Root causes:**
  1. Previously, `FilipinoEngine` relied on `edge-tts` (`ffmpeg` conversion + internet access) or fell back to `filipino_to_phonetic()` spoken through English `PiperEngine` (switching speakers mid-sentence and mispronouncing Tagalog vowels/consonants).
  2. Word files had uneven volume and inconsistent sample rates across WAVs.
- **Fix & Rationale:**
  - Standardized strictly on **one offline voice per language**:
    - English (`"en"`): Piper `en_US-ryan-medium.onnx` via `PiperEngine` (single worker queue).
    - Filipino (`"fil"`): `sherpa-onnx` MMS-TTS Tagalog (`tgl`) VITS offline model (`brailce/models/filipino/model.onnx` + `tokens.txt`) via `FilipinoEngine` (single `queue.Queue` worker), plus `brailce/audio/filipino/` WAVs pre-rendered and RMS-normalized from that same voice via `tools/build_audio_bank.py` and `brailce/speech/audio_normalize.py`.
  - Deleted `filipino_to_phonetic()` runtime fallback and removed `edge-tts` / `ffmpeg` runtime imports from production engines. Added `speech.filipino_letter_voice` switch (`"filipino"` default vs `"english"`).
  - Created `tools/audit_audio.py` to enforce sample rate (`22050 Hz`), 16-bit mono PCM, active RMS consistency (`±3 dB` of cohort median, no silent/empty WAVs), and peak headroom across all cached audio banks.

---

### E — Offline Classical Grade 1 Braille Scanner & Camera Fixes
- **Root causes:**
  1. `PiCameraDevice` configured Picamera2 with `format="BGR888"`, which swaps R/B channels in OpenCV memory (cyan tint), captured single un-settled preview frames (`640x480`), and fell back to generating a fake synthetic Braille image when hardware was missing.
  2. `OBREngine` tried downloading YOLO weights from GitHub at runtime (failing offline) and stripped trailing `'s'` characters (`rstrip("sS")`).
  3. `ScannerMode` ran heavy detection continuously and spoke negative prompts on every cycle without hysteresis/cooldown.
- **Fix:**
  - `brailce/vision/camera.py`: Fixed Picamera2 `format="RGB888"` (matching OpenCV BGR order), added AE/AWB settle & colour-gain lock, `capture_still()`, `capture_sharpest(n=5)` via `compute_sharpness()`, and explicit `CameraNotConnectedError` reporting (`SCANNER_CAMERA_NOT_CONNECTED`) instead of synthetic Braille fabrication.
  - `brailce/vision/obr_engine.py`: Implemented an offline classical dot-level Grade 1 recognizer (quality gates for brightness/Laplacian sharpness/dot resolution, illumination flattening + Difference-of-Gaussians dot detector, border artifact filtering, nearest-neighbor skew estimation & deskew, 2x3 cell lattice fitter, exact `BINARY_DICT` translation returning `?` / `"unknown"` on invalid cells without guessing), removed runtime GitHub downloads and `rstrip("sS")`, and gated optional local YOLO weights behind `use_yolo_secondary=False`.
  - `brailce/modes/scanner_mode.py`: Added ~3.5 Hz lightweight classical presence polling with hysteresis (`3` positive frames / `4` negative frames) and `6 s` cooldown, sharpest still capture on Enter, actionable distance/blur feedback (`SCANNER_BLURRY`, `SCANNER_MOVE_CLOSER`, `SCANNER_MOVE_FARTHER`), systematic interruptible letter-by-letter -> whole-word readout (`CHAR_PACE_S`, `WORD_PAUSE_S`, `LINE_PAUSE_S`), Enter-to-repeat, Clear-to-reset, and any-key speech cancellation.
  - Added `tools/scan_debug.py` visual dot/cell annotation CLI and `brailce/tests/test_obr_classical.py`.

---

## 2. Manual Raspberry Pi 5 Test Checklist

Run these checks directly on the Raspberry Pi 5 hardware after deployment:

### Checklist 1: Audio Bank & Offline Parity
- [ ] Run `python3 tools/audit_audio.py` on Pi and verify `0` failures (`en_characters` and `filipino` PASS).
- [ ] Disconnect Wi-Fi/Ethernet and launch `python3 main.py`; confirm zero network errors at startup and during speech.

### Checklist 2: Free Input & English Word Consistency
- [ ] Press single letter keys `A`–`Z` and confirm crisp, unclipped single-letter pronunciation.
- [ ] Type `is`, `his`, `this`, `the`, `was`, `what`, `when`, `where`, `why`, `who` followed by **Space** or **Enter**; verify natural pronunciation matches sentence playback (`This is when we read.`).
- [ ] Press Backspace/Clear during playback and verify speech stops immediately.

### Checklist 3: Filipino Mode (Language Toggle, Math & AI)
- [ ] Press Language toggle (`F3` / hardware switch) -> verify `"Wikang Filipino."` announcement and `[FIL]` LCD indicator.
- [ ] Switch to **Math Mode**: enter `5+3` and `12-7` -> verify Tagalog counting steps, LCD card wrapping (no text clipped on `480x320`), and clean `20-5` = `15` audio (`labinlima`) with uniform single Filipino voice.
- [ ] Switch to **AI Assistant Mode**: type `"Ano ang araw?"` and `"Ano ang Braille?"` -> verify short pure-Filipino voice replies without English filler, markdown characters, or speaker switching.

### Checklist 4: Scanner Mode (Pi Camera v2 + Embossed Sheet)
- [ ] Disconnect camera ribbon briefly -> enter Scanner Mode and verify LCD shows + speaks `"Camera not connected."` (and Filipino equivalent) without showing fake Braille.
- [ ] Connect Pi Camera v2 under side raking LED light (~15–30° angle, ~12–15 cm close focus). Hold blank paper -> no false `"Braille detected"` spam.
- [ ] Slide Grade 1 embossed sheet (`NEED` / `BRAILLE` / `#1234`) into view -> hear `"Braille detected. Press Enter to read."` once (no repeated nagging).
- [ ] Press **Enter** -> confirm LCD displays each character + Braille dots in sync while speaking letter-by-letter then the whole word (`N` `E` `E` `D` -> `need`).
- [ ] Press any key mid-readout -> playback stops immediately. Press **Enter** again to repeat or **Clear** to reset for a new page.
- [ ] Run `python3 tools/scan_debug.py --camera --out /tmp/scan_debug.png` to inspect dot alignment & Laplacian sharpness if focusing the IMX219 lens.

---

## 3. Phase 1–5 Comprehensive Architectural Upgrades (Round 3)

### Phase 1: Frozen English Baseline & Crash Isolation
- Verified and snapshot 100% frozen byte-for-byte behavior for all English audio files, `PiperEngine`, `EspeakEngine`, and `STRINGS["en"]` via `brailce/tests/test_english_frozen.py`.
- Fixed `UsbWebcamDevice` missing `import threading` and `from typing import Any` (`NameError: name 'threading' is not defined`).
- Resolved Filipino character synthesis regression when keypress events arrive with custom styles.

### Phase 2: Scale-Adaptive Classical OBR Braille Scanner Rewrite
- **Scale-Adaptive Multi-Scale Detection:** Difference-of-Gaussians ($\sigma=1.2, 1.8, 2.4, 3.2$) with shadow-highlight pairing and area filtering (`area >= 0.30 * med_a`) that eliminates edge artifacts while preserving embossed dot shadows.
- **Robust Skew & Dot Pitch Estimation:** Nearest-neighbor vector orientation clustering in $\pm 15^\circ$ range with distance histograms detecting true physical dot pitch ($p_{dot}$).
- **Global Lattice Fitter & Metric Alignment:** Symmetric cell boundary partition at $(p_{cell} + p_{dot}) / 2$ with intra-cell dot column tolerance $[0.82 \cdot p_{dot}, 1.18 \cdot p_{dot}]$ and inter-cell space tolerance $\ge 1.35 \cdot p_{dot}$.
- **Non-Braille Rejection:** Safe rejection of monitor terminal photos (`tools/test_real_cam.jpg`) and random noise pages via unknown-cell ratio limit ($>20\%$) and line density coherence.
- Added comprehensive ground truth suite `tools/synth_braille.py`, `tools/eval_real_scans.py`, and `brailce/tests/test_obr_synthetic.py` (accuracy $\ge 90\%$ across dot diameters 7–12px, rotations $\pm 5^\circ$, and both ink and embossed textures).

### Phase 3: Natural Filipino Prosody, Expanded Lexicon & Audition Suite
- **Single Lexicon Truth:** Mapped letter/number/symbol names via `brailce/i18n/fil_keys.py`.
- **Numbers to 999,999:** Full Tagalog number expansion in `brailce/speech/filipino_engine.py` (`tagalog_number`), supporting thousands (`libo`) and hundreds (`daan` / `raan`).
- **Natural Sentence Rhythm:** Configured 250ms silence pause for commas and semicolons, 400ms for terminal punctuation (`.`, `?`, `!`, `\n`).
- **Normalized Grammar Tokens:** Handled `"mga"` normalized to `"manga"`, and pre-synthesized all missing vocabulary WAVs (`word_libo.wav`, `word_manga.wav`, `word_pagpindot.wav`, `word_panuri.wav`, `word_posisyon.wav`, `word_tapos.wav`).
- Generated complete candidate parameter evaluation and human listening checklist in `AUDITION.md` (all 3 parameter sets achieve 0.000% clipping).

### Phase 4: High-Contrast ILI9488 LCD UI Typography & Layout
- **Bundled Font:** Packaged clean TrueType font `brailce/display/fonts/BrailceDefault.ttf` supporting both standard Latin ASCII and Unicode Braille patterns (U+2800–U+28FF).
- **Type Scale:** Active character scaled to 72px, mode header titles to 20px, labels to 18px, and body text to 15px.
- **Emoji Elimination:** Clean textual headers (`FREE INPUT`, `MATH MODE`, `AI ASSISTANT`, `SCANNER`) with clean ASCII battery meter (`pct%` / `pct%+`).
- Added screenshot verification tool `tools/render_screens.py` generating high-contrast renders across all operational modes.

### Phase 5: Production Deployment & Verification Scripts
- Added Windows PowerShell deployment script `tools/deploy_to_pi.ps1`.
- Added Raspberry Pi automated update script `tools/pi_update.sh`.


