#!/bin/bash
# ========================================================
# Brailce Raspberry Pi 5 Update & Verification Script
# Run this inside ~/brailce on the Raspberry Pi
# ========================================================

set -e

echo "=== Updating Brailce from GitHub ==="
git fetch origin master
git reset --hard origin/master

echo "=== Verifying Python Dependencies ==="
if [ -f requirements-pi.txt ]; then
    pip install -q -r requirements-pi.txt || echo "Warning: Some packages may require system apt install."
fi

echo "=== Verifying Audio Banks ==="
python3 tools/audit_audio.py

echo "=== Running Offline Self-Tests ==="
python3 -m pytest brailce/tests/test_english_frozen.py -q
python3 -m pytest brailce/tests/test_camera_crash.py -q
python3 -m pytest brailce/tests/test_filipino_keypress_crash.py -q

echo "========================================================"
echo " Update complete! All tests passed."
echo " To start Brailce, run:"
echo "     python3 main.py"
echo "========================================================"

