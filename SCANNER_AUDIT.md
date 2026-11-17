# Brailce Scanner Mode — Software Audit

**Date:** 2026-11-17  
**Scope:** `brailce/vision/obr_engine.py`, `brailce/modes/scanner_mode.py`, and related config

---

## 1. What actually runs at runtime (factual inventory)

| Component | What the code actually uses |
|---|---|
| **Dot detection** | `OBREngine.detect_dots()` — multi-scale Difference-of-Gaussians (DoG) on a resized frame; no YOLO, no ML |
| **Braille presence** | `OBREngine.braille_presence_confidence()` → float [0,1]; `detect_braille_presence()` returns `True` when confidence >= threshold |
| **Character decode** | `OBREngine.decode_dots_to_text()` — classical 6-bit dot-grid cell assignment |
| **Camera** | USB webcam via OpenCV `VideoCapture(0)` |
| **TTS** | `fil-PH-BlessicaNeural` (pre-rendered WAV bank) + Piper Ryan (English) |

NOTE: YOLOv8 (`brailce/vision/yolov8_braille.pt`) is an **optional secondary path** (`use_yolo_secondary=False` by default). It is never downloaded at runtime. The default and only active runtime path is the classical OpenCV dot-grid recognizer.

---

## 2. Camera settings negotiated at startup

The USB webcam is opened with `cv2.VideoCapture(0)`. Brailce requests:

- `CAP_PROP_AUTOFOCUS = 1` (enabled)
- `CAP_PROP_FRAME_WIDTH = 1280`, `CAP_PROP_FRAME_HEIGHT = 720` (may be downscaled by driver)
- `CAP_PROP_BUFFERSIZE = 1` (minimize latency)

Actual negotiated values depend on the webcam hardware and are logged at startup.

---

## 3. Datasets / test images

| File | Purpose |
|---|---|
| `tools/enhanced_braille.jpg` | High-contrast scan of a real Grade 1 Braille page (~1294 dots) |
| `tools/test_real_cam.jpg` | Low-angle webcam shot (~35 dots, partial view) |

---

## 4. Configuration audit

| Key | Location | Value | Effect |
|---|---|---|---|
| `vision.confidence_threshold` | `config.json`, `defaults.py` | **0.60** | Minimum confidence for "Braille detected" |
| `vision.use_yolo_secondary` | `defaults.py` | `false` | YOLOv8 disabled |
| `vision.model_path` | `config.json` | `brailce/vision/yolov8_braille.pt` | Unused (secondary path off) |
| `speech.filipino_char_gap_ms` | both | 200 | Scanner spell-out pacing in Filipino |
| `speech.filipino_word_gap_ms` | both | 30 | Word gap in concatenated Filipino TTS |

---

## 5. Presence Detection Flow

```
Frame captured (preview loop, ~15 fps)
     |
     v
check_image_quality()  -- brightness p90 > 55, Laplacian var > 4
     | fail -> 0.0
     v
detect_dots()          -- multi-scale DoG, 640px resize
     | < 4 dots -> 0.0
     v
dot_coverage gate      -- total dot area / frame area > 10% -> noise -> 0.0
     |
     v
diameter consistency   -- score from std/median ratio
     |
     v
nearest-neighbor pitch -- must be in [1.2x, 7.0x] dot diameter
     |
     v
grid regularity        -- fraction within +/-40% of median pitch
     |
     v
polka-dot penalty      -- 4-way cross-neighbor ratio penalizes uniform grids
     |
     v
confidence = 0.25*diam + 0.50*regularity + 0.25*(1 - polka)
     |
     v
ScannerMode._detection_worker compares conf >= 0.60
Hysteresis: 3 consecutive positives -> "Braille detected" announcement
            4 consecutive negatives -> reset
```

---

## 6. Known false-positive cases and results

| Input | Conf | Passes 0.60? | Why |
|---|---|---|---|
| Blank white frame | 0.0 | No | Quality gate fails |
| Random pixel noise (seed 123, 320x320) | 0.0 | No | Dot coverage >10% |
| Polka-dot grid (15px spacing) | ~0.3 | No | Polka-dot cross-neighbor penalty |
| Real Braille page (`enhanced_braille.jpg`) | 0.64 | Yes | Real lattice |
| Synthetic 4-dot Braille grid | ~0.7 | Yes | Regular spacing |

---

## 7. Documentation corrections

Previous versions of `HARDWARE_AND_FUNCTIONS.md` and `README.md` incorrectly claimed:
- YOLOv8 is the main detection engine -> **Corrected**: classical OpenCV dot-grid is the only runtime engine; YOLOv8 is disabled by default.
- eSpeak NG is the TTS engine -> **Corrected**: Piper Neural TTS (Ryan, en_US) for English; pre-rendered `fil-PH-BlessicaNeural` WAV bank for Filipino.
