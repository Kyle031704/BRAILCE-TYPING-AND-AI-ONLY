"""
Automated Deployment Script to Raspberry Pi 5
==============================================

Uploads updated Brailce code, speech services, and configs to the Raspberry Pi 5,
cleans cache, and runs remote verification.
"""

import os
import sys
import time

try:
    import paramiko
except ImportError:
    print("Paramiko not found. Installing paramiko...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko

def get_pi_host():
    if os.getenv("PI_IP"):
        return os.getenv("PI_IP")
    import socket
    candidates = ["192.168.137.156", "192.168.0.107", "10.161.29.70"]
    for ip in candidates:
        try:
            s = socket.socket()
            s.settimeout(0.4)
            if s.connect_ex((ip, 22)) == 0:
                s.close()
                return ip
            s.close()
        except Exception:
            pass
    return "192.168.137.156"

PI_HOST = get_pi_host()
PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")
REMOTE_BASE = "/home/raspberrypi5/brailce"
LOCAL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FILES_TO_DEPLOY = [
    # Speech system
    "brailce/speech/service.py",
    "brailce/speech/espeak_engine.py",
    "brailce/speech/piper_engine.py",
    "brailce/speech/normalizer.py",
    "brailce/speech/audio_device.py",
    "brailce/speech/__init__.py",
    "brailce/speech/base.py",
    # Core & Modes
    "brailce/core/app.py",
    "brailce/modes/free_input.py",
    "brailce/modes/math_mode.py",
    "brailce/modes/ai_assistant.py",
    "brailce/modes/scanner_mode.py",
    "brailce/vision/obr_engine.py",
    "brailce/braille/arithmetic.py",
    "brailce/ai/ollama_client.py",
    "brailce/audio/sound_effects.py",
    # Config
    "brailce/config/defaults.py",
    "brailce/config/config.json",
    # Mappings
    "brailce/braille/mappings/letters.json",
    "brailce/braille/mappings/numbers.json",
    "brailce/braille/mappings/math.json",
    "brailce/braille/mappings/punctuation.json",
    # Tests
    "brailce/tests/test_speech_service.py",
    "brailce/tests/test_normalizer.py",
    "brailce/tests/test_obr_engine.py",
]


def deploy():
    print(f"Connecting to Raspberry Pi 5 at {PI_HOST} (user: {PI_USER})...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
    except Exception as exc:
        print(f"\n[ERROR] Could not connect to {PI_HOST}: {exc}")
        print("Tip: If the Raspberry Pi IP address changed, set PI_IP, for example:")
        print("    set PI_IP=192.168.0.xxx; python tools/deploy_to_pi.py")
        sys.exit(1)

    print("Connected successfully!")
    sftp = ssh.open_sftp()

    print("\n--- 1. Uploading Updated Files ---")
    for rel in FILES_TO_DEPLOY:
        loc = os.path.join(LOCAL_BASE, rel.replace("/", os.sep))
        rem = f"{REMOTE_BASE}/{rel}"
        if not os.path.exists(loc):
            print(f"Skipping missing local file: {rel}")
            continue

        rem_dir = os.path.dirname(rem)
        try:
            ssh.exec_command(f"mkdir -p '{rem_dir}'")
        except Exception:
            pass

        print(f"  Uploading: {rel} -> {rem}")
        sftp.put(loc, rem)

    # Sync pre-cached Piper character WAV audio files
    char_cache_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "characters")
    if os.path.isdir(char_cache_dir):
        rem_char_dir = f"{REMOTE_BASE}/brailce/audio/characters"
        ssh.exec_command(f"mkdir -p '{rem_char_dir}'")
        wav_files = [f for f in os.listdir(char_cache_dir) if f.endswith(".wav")]
        print(f"\n--- Syncing {len(wav_files)} Pre-Cached Piper Character Audio WAV Files ---")
        for f in wav_files:
            loc_wav = os.path.join(char_cache_dir, f)
            rem_wav = f"{rem_char_dir}/{f}"
            sftp.put(loc_wav, rem_wav)
        print("Pre-cached character WAV files synced successfully!")

    sftp.close()
    print("All files synchronized successfully!")

    print("\n--- 2. Clearing Remote __pycache__ ---")
    ssh.exec_command(f"find {REMOTE_BASE} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null")

    print("\n--- 3. Stopping In-Flight Audio / Brailce Processes ---")
    ssh.exec_command("pkill -9 -f 'python3.*brailce' 2>/dev/null; pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null")
    time.sleep(1)

    print("\n--- 4. Running Remote Verification Tests ---")
    cmd_test = f"{REMOTE_BASE}/venv/bin/pytest {REMOTE_BASE}/brailce/tests/test_speech_service.py -v"
    stdin, stdout, stderr = ssh.exec_command(cmd_test)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out)
    if err and "FAILED" in err:
        print("[STDERR]:", err)

    print("\n=======================================================")
    print("DEPLOYMENT COMPLETE!")
    print(f"To run Brailce interactively on the Raspberry Pi, run:")
    print(f"    ssh {PI_USER}@{PI_HOST}")
    print(f"    cd {REMOTE_BASE}")
    print(f"    source venv/bin/activate")
    print(f"    python3 main.py")
    print("=======================================================")
    ssh.close()


if __name__ == "__main__":
    deploy()

