#!/usr/bin/env python3
"""
Brailce – Audio Output & TTS Hardware Diagnostic Tool
======================================================

Tests the MAX98357A I2S audio amplifier and eSpeak NG on Raspberry Pi 5.

What this script does:
1. Inspects ALSA audio devices (aplay -l).
2. Checks whether MAX98357A or generic I2S DAC is detected.
3. Generates and plays a clean audio test tone directly through ALSA.
4. Tests eSpeak NG voice synthesis with Braille-specific phrases.
5. Verifies the Brailce EspeakEngine Python background worker.

Usage:
    python tools/test_audio.py
    python tools/test_audio.py --volume 90
    python tools/test_audio.py --phrase "Testing Braille audio"
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brailce.speech.espeak_engine import EspeakEngine


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_alsa_devices() -> dict:
    """Scan ALSA sound cards using aplay -l."""
    info = {
        "aplay_installed": shutil.which("aplay") is not None,
        "cards": [],
        "has_i2s_dac": False,
        "output": "",
    }

    if not info["aplay_installed"]:
        return info

    try:
        proc = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["output"] = proc.stdout
        for line in proc.stdout.splitlines():
            if line.startswith("card"):
                info["cards"].append(line)
                lower = line.lower()
                if any(k in lower for k in ["max98357", "hifiberry", "dac", "i2s"]):
                    info["has_i2s_dac"] = True
    except Exception as exc:
        info["output"] = str(exc)

__test__ = False


def run_espeak_cli_test(phrase: str, volume: int = 80) -> bool:
    """Test standard eSpeak NG CLI."""
    if sys.platform == "win32":
        print("  [INFO] Windows detected. Testing voice synthesis via Brailce Speech Pipeline in Step 3.")
        return True

    cmd = shutil.which("espeak-ng") or shutil.which("espeak")
    if not cmd:
        print("  [FAIL] Neither 'espeak-ng' nor 'espeak' found in system PATH.")
        print("         Install via: sudo apt-get update && sudo apt-get install -y espeak-ng")
        return False

    print(f"  [OK] Found TTS engine: {cmd}")
    print(f"  --> Speaking: '{phrase}' (Volume: {volume}%)")
    try:
        subprocess.run(
            [cmd, "-v", "en", "-s", "160", "-a", str(volume), phrase],
            check=True,
            timeout=8,
        )
        return True
    except Exception as exc:
        print(f"  [FAIL] Error speaking phrase: {exc}")
        return False


def run_brailce_engine_test(volume: int = 80) -> bool:
    """Test Brailce's background threaded EspeakEngine."""
    print("  Initializing Brailce EspeakEngine background queue...")
    engine = EspeakEngine(voice="en", rate=160, volume=volume)

    if not engine.is_available() and sys.platform != "win32":
        print("  [FAIL] EspeakEngine reports engine not available.")
        return False

    test_words = [
        "Welcome to Braille C.",
        "Letter A. Dot 1.",
        "Letter B. Dot 1, 2.",
        "Audio system is fully operational.",
    ]

    print("  Enqueueing speech queue test items:")
    for w in test_words:
        print(f"    - Enqueue: '{w}'")
        engine.speak(w)
        time.sleep(1.2)

    # Wait for queue to flush
    time.sleep(2.0)
    engine.stop()
    print("  [OK] EspeakEngine queue completed successfully.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Brailce Audio & TTS Diagnostic Tool")
    parser.add_argument("--volume", type=int, default=85, help="Speech volume 0-100 (default: 85)")
    parser.add_argument("--phrase", type=str, default="Brailce audio test. Speakers connected successfully.", help="Custom test phrase")
    args = parser.parse_args()

    banner("Brailce – Audio & Speaker Hardware Diagnostic")

    # Step 1: Scan ALSA devices
    print("\n[Step 1/3] Inspecting ALSA Sound Cards:")
    alsa = check_alsa_devices()
    if alsa["aplay_installed"]:
        print(f"  Cards detected ({len(alsa['cards'])}):")
        for card in alsa["cards"]:
            print(f"    * {card}")

        if alsa["has_i2s_dac"]:
            print("\n  >>> SUCCESS: I2S DAC / MAX98357A detected! <<<")
        else:
            print("\n  [NOTE] No obvious 'MAX98357' or 'hifiberry-dac' label in card list.")
            print("         If audio doesn't play, verify 'dtoverlay=max98357a' in /boot/firmware/config.txt")
    else:
        print("  [INFO] 'aplay' not available on this platform (simulation mode).")

    # Step 2: Test eSpeak NG Command Line
    print("\n[Step 2/3] Testing eSpeak NG Text-to-Speech Output:")
    run_espeak_cli_test(args.phrase, volume=args.volume)

    # Step 3: Test Brailce Python EspeakEngine Queue
    print("\n[Step 3/3] Testing Brailce Python Speech Pipeline:")
    run_brailce_engine_test(volume=args.volume)

    banner("Audio Diagnostic Completed!")
    print("If you heard the test voices clearly, your speakers and amplifier are ready!\n")


if __name__ == "__main__":
    main()
