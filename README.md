# Brailce — Offline Bilingual Braille Learning Device

Brailce is a portable **Raspberry Pi 5** assistive learning device for visually impaired Grade 1 learners in **English (`en`)** and **Filipino (`fil`)**. It combines a dedicated physical Braille-embossed keypad, a `480x320` ILI9488 SPI LCD, 100% offline bilingual speech synthesis, a local Ollama AI tutor, and a Raspberry Pi Camera Module v2 Optical Braille Recognition (OBR) scanner.

---

## Core Capabilities & Modes

| Mode | Purpose | Controls |
|------|---------|----------|
| **Free Input** | Immediate letter feedback (`1 key = 1 letter`) + clean whole-word pronunciation on Space/Enter | `A–Z`, `0–9`, `Space`, `Enter`, `Backspace`, `Clear` |
| **Math Mode** | Safe Grade 1 arithmetic (`+`, `-`, `*`, `/`) with bilingual counting explanations and LCD solution card | Digits, operators, `Enter`, `Clear` |
| **AI Assistant** | Offline Ollama (`lfm2.5-1.2b-instruct`) Q&A with bilingual guardrails and clean spoken normalization | Question input + `Enter` |
| **Scanner Mode** | Offline classical dot-level Grade 1 Braille camera scanner with live presence detection and letter -> word readout | Position sheet, `Enter` to scan/repeat, any key to stop, `Clear` to reset |

Desktop controls during simulation: `A–Z`, `0–9`, `+ - * /`, `Space`, `Enter`, `Backspace`, `F1` (Mode), `F2` (Clear), `F3` (Language EN/FIL), `F11`/`F12` (Volume Down/Up), `Esc` (Quit).

---

## Offline Speech & Audio Architecture

Brailce uses **one consistent offline voice per language**, normalized to `22050 Hz` 16-bit mono PCM with `120 ms` pre-silence / `180 ms` post-silence and routed through `AudioDeviceCoordinator.play_wav_file()` (`aplay -q` on Raspberry Pi):

1. **English (`en`) — Piper (`en_US-ryan-medium`):**
   - `PiperEngine` locks VITS synthesis to `length_scale=1.05`, `noise_scale=0.5`, `noise_w=0.6`.
   - `SpeechService.speak_word_clean()` uses cached `brailce/audio/words_en/<word>.wav` files or Option 1 carrier slicing (`"Please say: <word>."` via `brailce/speech/audio_normalize.py`) so short function words (`is`, `his`, `this`, `the`, `when`, `where`, `why`, `who`) are never clipped.
2. **Filipino (`fil`) — `sherpa-onnx` MMS-TTS Tagalog (`tgl`) VITS:**
   - `FilipinoEngine` synthesizes live Tagalog phrases completely offline from `brailce/models/filipino/model.onnx` + `tokens.txt` and plays pre-rendered WAVs from `brailce/audio/filipino/`.
3. **Bilingual Strings (`brailce/i18n/strings.py`):**
   - Single source of truth for `"en"` and `"fil"` prompts so LCD output and TTS output never diverge.

### Pre-building & Auditing Audio Banks
```bash
# Generate / normalize offline English word cache and Filipino audio bank
python tools/build_audio_bank.py

# Verify sample rate, mono PCM, peak headroom, active RMS consistency (±3 dB), and zero silent WAVs
python tools/audit_audio.py
```

---

## Raspberry Pi 5 Setup & Models (100% Offline Runtime)

Model binaries and audio caches are excluded from git (`*.onnx`, `*.pt`, `brailce/models/`). Place/install the following on the Raspberry Pi 5 before offline operation:

1. **System packages:**
   ```bash
   sudo apt update && sudo apt install -y alsa-utils python3-picamera2
   ```
2. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Piper English model:**
   - Place `piper` binary and `en_US-ryan-medium.onnx` (+ `.json`) under `~/piper/` (or configure `speech.piper_bin` / `speech.piper_model` in `brailce/config/config.json`).
4. **`sherpa-onnx` Filipino MMS-TTS (`tgl`) model:**
   - Download `vits-mms-tgl` (`model.onnx` and `tokens.txt`) from sherpa-onnx releases and place inside `brailce/models/filipino/`.
5. **Local Ollama LLM:**
   ```bash
   ollama pull lfm2.5-1.2b-instruct
   ```
6. *(Optional)* Secondary YOLO weights:
   - Not required; `OBREngine` uses the offline classical dot-level recognizer by default (`vision.use_yolo_secondary: false`).

---

## Braille Scanner Lighting & Focus Guide (Pi Camera v2 / IMX219)

Reliable OBR requires crisp dot shadows and highlands:
- **Camera color order:** Configured in `brailce/vision/camera.py` as `format="RGB888"` (matching OpenCV BGR memory layout without cyan tint), with AE/AWB settle and colour-gain lock.
- **Close focus adjustment:** Adjust the IMX219 stock manual focus lens ring carefully for a **12–15 cm** working distance over a printed/embossed page until dot diameters measure `>= 6 px`.
- **Raking LED lighting:** Mount a warm/neutral LED strip at a shallow **15–30° side angle** (raking light across the paper surface) rather than overhead flash, so each raised Braille dot casts a sharp highlight crest + dark crescent shadow.
- **Diagnostic inspection:**
  ```bash
  python tools/scan_debug.py --camera --out scan_debug.png
  python tools/scan_debug.py --image path/to/sample.jpg --out scan_debug.png
  ```

---

## Environment Configuration & Deployment

Copy `.env.example` to `.env` (never commit `.env` or personal IPs/passwords) when deploying from a development PC:

```bash
cp .env.example .env
# Edit PI_IP, PI_USER, PI_PASS, PI_REMOTE_DIR
python tools/deploy_to_pi.py
```

---

## Testing

Run the unit and integration test suite locally (laptop/headless, without Pi hardware):

```bash
python -m pytest brailce/tests -q --tb=short --maxfail=3
```
