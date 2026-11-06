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

def _probe_ip(ip, port=22, timeout=0.3):
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        if s.connect_ex((ip, port)) == 0:
            s.close()
            return ip
    except Exception:
        pass
    s.close()
    return None

def find_pi_ip():
    if os.getenv("PI_IP"):
        return os.getenv("PI_IP")

    import socket
    import concurrent.futures

    # Direct hostnames and known IPs
    priority = ["192.168.137.156", "raspberrypi5.local", "raspberrypi.local", "10.161.29.70"]
    for host in priority:
        try:
            addr = socket.gethostbyname(host)
            if _probe_ip(addr):
                return addr
        except Exception:
            pass

    # Fast scan of Hotspot subnet (192.168.137.x) and Wi-Fi subnet (192.168.0.x)
    subnets = ["192.168.137", "192.168.0"]
    candidates = [f"{sub}.{i}" for sub in subnets for i in range(2, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
        for res in ex.map(lambda ip: _probe_ip(ip, 22, 0.25), candidates):
            if res:
                return res

    return None

def get_pi_host(max_wait_seconds=25):
    print("Searching for Raspberry Pi 5 on the network...")
    ip = find_pi_ip()
    if ip:
        print(f"Found Raspberry Pi 5 at {ip}!")
        return ip

    print("Pi not found immediately. Polling for connection...")
    print(">> NOTE: Please make sure Windows Mobile Hotspot is turned ON in Settings/Quick Settings! <<")
    start = time.time()
    while time.time() - start < max_wait_seconds:
        time.sleep(2)
        ip = find_pi_ip()
        if ip:
            print(f"\n[CONNECTED] Raspberry Pi 5 detected at {ip}!")
            return ip
        remaining = int(max_wait_seconds - (time.time() - start))
        print(f"  Waiting for Pi to connect... ({remaining}s remaining)", end="\r", flush=True)

    print("\nCould not detect Pi automatically after timeout. Defaulting to 192.168.137.156.")
    return "192.168.137.156"

PI_HOST = None  # Resolved inside deploy()
PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")
REMOTE_BASE = "/home/raspberrypi5/brailce"
LOCAL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FILES_TO_DEPLOY = [
    # Speech system
    "brailce/speech/service.py",
    "brailce/speech/espeak_engine.py",
    "brailce/speech/piper_engine.py",
    "brailce/speech/filipino_engine.py",
    "brailce/speech/normalizer.py",
    "brailce/speech/audio_device.py",
    "brailce/speech/__init__.py",
    "brailce/speech/base.py",
    # Core & Modes
    "brailce/core/app.py",
    "brailce/core/events.py",
    "brailce/core/state.py",
    "brailce/input/keyboard_input.py",
    "brailce/input/gpio_keypad.py",
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
    "brailce/tests/test_filipino.py",
    "brailce/tests/test_input.py",
]


def deploy():
    global PI_HOST
    if not PI_HOST:
        PI_HOST = get_pi_host()
    print(f"\nConnecting to Raspberry Pi 5 at {PI_HOST} (user: {PI_USER})...")
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

    # Sync pre-cached Piper character WAV audio files (only if size differs)
    char_cache_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "characters")
    if os.path.isdir(char_cache_dir):
        rem_char_dir = f"{REMOTE_BASE}/brailce/audio/characters"
        ssh.exec_command(f"mkdir -p '{rem_char_dir}'")
        try:
            remote_char_files = {attr.filename: attr.st_size for attr in sftp.listdir_attr(rem_char_dir)}
        except Exception:
            remote_char_files = {}
        wav_files = [f for f in os.listdir(char_cache_dir) if f.endswith(".wav")]
        uploaded_chars = 0
        for f in wav_files:
            loc_wav = os.path.join(char_cache_dir, f)
            rem_wav = f"{rem_char_dir}/{f}"
            loc_size = os.path.getsize(loc_wav)
            if remote_char_files.get(f) != loc_size:
                sftp.put(loc_wav, rem_wav)
                uploaded_chars += 1
        print(f"Character audio sync: {uploaded_chars} updated / {len(wav_files)} verified.")

    # Sync pre-cached Grade 1 Filipino audio bank WAV files (only if size differs)
    fil_cache_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "filipino")
    if os.path.isdir(fil_cache_dir):
        rem_fil_dir = f"{REMOTE_BASE}/brailce/audio/filipino"
        ssh.exec_command(f"mkdir -p '{rem_fil_dir}'")
        try:
            remote_fil_files = {attr.filename: attr.st_size for attr in sftp.listdir_attr(rem_fil_dir)}
        except Exception:
            remote_fil_files = {}
        wav_files = [f for f in os.listdir(fil_cache_dir) if f.endswith(".wav")]
        uploaded_fil = 0
        for f in wav_files:
            loc_wav = os.path.join(fil_cache_dir, f)
            rem_wav = f"{rem_fil_dir}/{f}"
            loc_size = os.path.getsize(loc_wav)
            if remote_fil_files.get(f) != loc_size:
                sftp.put(loc_wav, rem_wav)
                uploaded_fil += 1
        print(f"Filipino audio sync: {uploaded_fil} updated / {len(wav_files)} verified.")

    sftp.close()
    print("All files synchronized successfully!")

    print("\n--- 2. Clearing Remote __pycache__ ---")
    ssh.exec_command(f"find {REMOTE_BASE} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null")

    print("\n--- 3. Stopping In-Flight Audio / Brailce Processes ---")
    ssh.exec_command("pkill -9 -f 'python3.*brailce' 2>/dev/null; pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null")
    time.sleep(1)

    print("\n--- 4. Running Remote Verification Tests ---")
    cmd_test = f"{REMOTE_BASE}/venv/bin/pytest {REMOTE_BASE}/brailce/tests/test_speech_service.py {REMOTE_BASE}/brailce/tests/test_filipino.py {REMOTE_BASE}/brailce/tests/test_input.py -v"
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

