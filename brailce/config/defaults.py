"""
Brailce Default Configuration
===============================

Defines the complete default configuration dictionary.

All keys and their default values are declared here.
The config loader merges user-supplied config.json on top of these defaults.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- System ----
    "system": {
        "simulation_mode": True,
        "debug": False,
        "log_level": "INFO",
    },

    # ---- Input ----
    "input": {
        # "keyboard" for laptop simulation, "gpio" for Raspberry Pi hardware
        "device": "keyboard",
        # GPIO pin assignments — only used when device == "gpio"
        "gpio_pins": {
            "mode": None,
            "volume_up": None,
            "volume_down": None,
            "backspace": None,
            "clear": None,
            "enter": None,
            "space": None,
        },
        # Key-matrix dimensions — only used when a matrix keypad is configured
        "key_matrix": {
            "rows": None,
            "columns": None,
            "row_pins": [],
            "column_pins": [],
        },
    },

    # ---- Braille ----
    "braille": {
        # Paths to JSON mapping files (relative to project root)
        "mapping_files": [
            "brailce/braille/mappings/letters.json",
            "brailce/braille/mappings/numbers.json",
            "brailce/braille/mappings/punctuation.json",
            "brailce/braille/mappings/math.json",
        ],
        # Whether to load optional Nemeth math mappings
        "load_nemeth": False,
        "nemeth_file": "brailce/braille/mappings/nemeth.json",
    },

    # ---- Speech ----
    "speech": {
        "enabled": True,
        "engine": "piper",
        "character_engine": "piper",
        "text_engine": "piper",
        "behavior": "character_and_word",
        "voice": "en",
        "rate": 165,
        "pitch": 52,
        "volume": 100,
        "word_gap_ms": 5,
        "speak_each_character": True,
        "speak_words": True,
        "speak_control_events": False,
        # Voice for single letters in Filipino mode: "filipino" (default) or "english"
        "filipino_letter_voice": "filipino",
        # Letter naming style in Filipino mode: "english_names" (ey, bi, si, ...) or "abakada" (ah, ba, ka, ...)
        "filipino_letter_style": "english_names",
        # Filipino voice architecture: "friendly_prerendered_plus_offline" (default) or "offline_only"
        "filipino_voice": "friendly_prerendered_plus_offline",
        # Filipino speech rate multiplier
        "filipino_speed": 1.0,
        # Maximum speech queue size before dropping events
        "max_queue_size": 50,
    },

    # ---- Sound Effects ----
    "sound": {
        "enabled": False,
        "volume": 0,
    },

    # ---- Display ----
    "display": {
        # "console" for laptop, "ili9488" for Raspberry Pi TFT
        "driver": "console",
        "width": 480,
        "height": 320,
        "show_braille_unicode": True,
        "show_braille_dots": True,
        "show_braille_visual": True,
    },

    # ---- AI ----
    "ai": {
        "enabled": True,
        "host": "http://127.0.0.1:11434",
        "model": "MODEL_NAME_HERE",
        "timeout_seconds": 30,
        "max_response_tokens": 80,
        "temperature": 0.5,
        "system_prompt": (
            "You are Brailce, a friendly voice assistant for Grade 1 Braille learners. "
            "Answer ONLY the user's actual question in 1 to 2 short, complete sentences. "
            "Never mention math unless the user asks a math question. "
            "Always finish your sentence completely."
        ),
    },

    # ---- Logging / Analytics ----
    "logging": {
        "enabled": True,
        "log_directory": "logs",
        "csv_filename_prefix": "brailce_session",
        "log_characters": True,
        "log_mode_changes": True,
        "log_ai_interactions": True,
        "log_errors": True,
        "record_timing": True,
    },

    # ---- Power ----
    "power": {
        # "simulated" for laptop, "rpi" for Raspberry Pi with battery HAT
        "driver": "simulated",
        "low_battery_threshold": 15,
        "poll_interval_seconds": 60,
    },

    # ---- Vision ----
    "vision": {
        # "usb" for USB webcam (default), "picamera" for Pi Cam v2, "simulated" for file/test, "auto" to probe
        "driver": "usb",
        "usb_device": 0,
        "autofocus": True,
        "focus_value": -1,
        "resolution": [1280, 720],
        "still_resolution": [1920, 1080],
        "model_path": "brailce/vision/yolov8_braille.pt",
        "confidence_threshold": 0.22,
        "iou_threshold": 0.45,
        "use_yolo_secondary": False,
        "announce_capital_sign": False,
        "test_image_path": "",
    },
}
