"""
Automated Deployment Script to Raspberry Pi 5
==============================================

Uploads updated Brailce code, speech services, and configs to the Raspberry Pi 5,
cleans cache, and runs remote verification.
Supports PuTTY's pscp/plink natively on Windows as well as paramiko.
"""

import os
import subprocess
import sys
import time

PUTTY_DIR = r"C:\Program Files\PuTTY"
PLINK_EXE = os.path.join(PUTTY_DIR, "plink.exe") if os.path.isdir(PUTTY_DIR) else "plink"
PSCP_EXE = os.path.join(PUTTY_DIR, "pscp.exe") if os.path.isdir(PUTTY_DIR) else "pscp"

HAVE_PUTTY = os.path.isfile(PLINK_EXE) and os.path.isfile(PSCP_EXE)

paramiko = None
if not HAVE_PUTTY:
    try:
        import paramiko
    except ImportError:
        pass

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
    "brailce/input/base.py",
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
    # Tests & Utils
    "brailce/utils/system_checks.py",
    "brailce/tests/test_speech_service.py",
    "brailce/tests/test_normalizer.py",
    "brailce/tests/test_obr_engine.py",
    "brailce/tests/test_filipino.py",
    "brailce/tests/test_input.py",
    "brailce/tests/test_speech.py",
]


def run_remote_plink(host: str, user: str, password: str, cmd: str) -> str:
    """Execute command on Pi using PuTTY plink."""
    p = subprocess.run(
        [PLINK_EXE, "-batch", "-pw", password, f"{user}@{host}", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if p.stdout:
        print(p.stdout)
    if p.stderr and ("error" in p.stderr.lower() or "fatal" in p.stderr.lower()):
        print("[REMOTE STDERR]:", p.stderr)
    return p.stdout


def deploy_via_putty(host: str, user: str, password: str) -> None:
    print(f"\n--- 1. Uploading Files via PuTTY PSCP to {user}@{host} ---")
    for rel in FILES_TO_DEPLOY:
        loc = os.path.join(LOCAL_BASE, rel.replace("/", os.sep))
        rem = f"{REMOTE_BASE}/{rel}"
        if not os.path.exists(loc):
            continue
        rem_dir = os.path.dirname(rem)
        run_remote_plink(host, user, password, f"mkdir -p '{rem_dir}'")
        print(f"  Uploading: {rel} -> {rem}")
        subprocess.run(
            [PSCP_EXE, "-batch", "-pw", password, loc, f"{user}@{host}:{rem}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )

    print("\n--- 2. Clearing Remote __pycache__ ---")
    run_remote_plink(host, user, password, f"find {REMOTE_BASE} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null")

    print("\n--- 3. Stopping In-Flight Audio / Brailce Processes ---")
    run_remote_plink(host, user, password, "pkill -9 -f 'python3.*brailce' 2>/dev/null; pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null")
    time.sleep(1)

    print("\n--- 4. Running Remote Verification Tests ---")
    cmd_test = f"{REMOTE_BASE}/venv/bin/pytest {REMOTE_BASE}/brailce/tests/test_speech_service.py {REMOTE_BASE}/brailce/tests/test_filipino.py {REMOTE_BASE}/brailce/tests/test_input.py {REMOTE_BASE}/brailce/tests/test_speech.py -v"
    run_remote_plink(host, user, password, cmd_test)


def deploy():
    global PI_HOST
    if not PI_HOST:
        PI_HOST = get_pi_host()

    if HAVE_PUTTY:
        print(f"Using PuTTY PSCP/PLINK for deployment to {PI_HOST}...")
        deploy_via_putty(PI_HOST, PI_USER, PI_PASS)
    elif paramiko:
        print(f"Connecting to Raspberry Pi 5 at {PI_HOST} via Paramiko...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
        except Exception as exc:
            print(f"\n[ERROR] Could not connect to {PI_HOST}: {exc}")
            sys.exit(1)
        sftp = ssh.open_sftp()
        for rel in FILES_TO_DEPLOY:
            loc = os.path.join(LOCAL_BASE, rel.replace("/", os.sep))
            rem = f"{REMOTE_BASE}/{rel}"
            if os.path.exists(loc):
                ssh.exec_command(f"mkdir -p '{os.path.dirname(rem)}'")
                sftp.put(loc, rem)
        sftp.close()
        ssh.exec_command(f"find {REMOTE_BASE} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null")
        ssh.exec_command("pkill -9 -f 'python3.*brailce' 2>/dev/null; pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null")
        stdin, stdout, stderr = ssh.exec_command(f"{REMOTE_BASE}/venv/bin/pytest {REMOTE_BASE}/brailce/tests/test_speech_service.py {REMOTE_BASE}/brailce/tests/test_filipino.py {REMOTE_BASE}/brailce/tests/test_input.py {REMOTE_BASE}/brailce/tests/test_speech.py -v")
        print(stdout.read().decode("utf-8", errors="replace"))
        ssh.close()
    else:
        print("[ERROR] Neither PuTTY (pscp/plink) nor paramiko found!")
        sys.exit(1)

    print("\n=======================================================")
    print("DEPLOYMENT COMPLETE!")
    print(f"To run Brailce interactively on the Raspberry Pi, run in PuTTY:")
    print(f"    cd {REMOTE_BASE}")
    print(f"    source venv/bin/activate")
    print(f"    python3 main.py")
    print("=======================================================")


if __name__ == "__main__":
    deploy()

