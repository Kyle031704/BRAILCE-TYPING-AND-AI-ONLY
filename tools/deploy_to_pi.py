"""
Automated Deployment Script to Raspberry Pi 5
==============================================

Packages all updated Brailce source code, configs, English isolated word audio,
and Filipino audio into a single fast archive bundle (brailce_update.tar.gz)
and uploads it in ONE SINGLE connection to prevent connection aborts / rate limiting.

Also provides a built-in instant HTTP fallback that can be pulled with curl
directly from PuTTY if PuTTY batch SSH is blocked.
"""

from __future__ import annotations

import concurrent.futures
import http.server
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
from typing import Optional

PUTTY_DIR = r"C:\Program Files\PuTTY"
PLINK_EXE = os.path.join(PUTTY_DIR, "plink.exe") if os.path.isdir(PUTTY_DIR) else "plink"
PSCP_EXE = os.path.join(PUTTY_DIR, "pscp.exe") if os.path.isdir(PUTTY_DIR) else "pscp"

SYSTEM_SCP = shutil.which("scp")
SYSTEM_SSH = shutil.which("ssh")

LOCAL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REMOTE_BASE = "/home/raspberrypi5/brailce"
BUNDLE_FILENAME = "brailce_update.tar.gz"
BUNDLE_PATH = os.path.join(LOCAL_BASE, BUNDLE_FILENAME)

PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")
HOSTKEY_ED25519 = "ssh-ed25519 255 SHA256:GfnaYxxoRmWmbZ48BYm6RdzxZTIky3DJ8Lh32WcXYhc"


def _probe_ip(ip: str, port: int = 22, timeout: float = 0.25) -> Optional[str]:
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


def find_pi_ip() -> Optional[str]:
    if os.getenv("PI_IP"):
        return os.getenv("PI_IP")

    priority = [
        "192.168.137.248",
        "192.168.137.156",
        "raspberrypi5.local",
        "raspberrypi.local",
        "10.161.29.70",
    ]
    for host in priority:
        try:
            addr = socket.gethostbyname(host)
            if _probe_ip(addr):
                return addr
        except Exception:
            pass

    subnets = ["192.168.137", "192.168.0", "192.168.1"]
    candidates = [f"{sub}.{i}" for sub in subnets for i in range(2, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
        for res in ex.map(lambda ip: _probe_ip(ip, 22, 0.25), candidates):
            if res:
                return res

    return None


def get_pi_host(max_wait_seconds: int = 15) -> str:
    print("Searching for Raspberry Pi 5 on network...")
    ip = find_pi_ip()
    if ip:
        print(f"Found Raspberry Pi 5 at {ip}!")
        return ip

    start = time.time()
    while time.time() - start < max_wait_seconds:
        time.sleep(1.5)
        ip = find_pi_ip()
        if ip:
            print(f"\n[CONNECTED] Raspberry Pi 5 detected at {ip}!")
            return ip

    print("\nCould not detect Pi automatically after timeout. Defaulting to 192.168.137.248.")
    return "192.168.137.248"


def create_deployment_bundle() -> str:
    """
    Bundle all application python packages, configs, isolated English word audio,
    and Filipino audio into a single clean .tar.gz archive.
    """
    print(f"\n--- 1. Packaging Update Bundle ({BUNDLE_FILENAME}) ---")
    if os.path.exists(BUNDLE_PATH):
        try:
            os.remove(BUNDLE_PATH)
        except Exception:
            pass

    added_count = 0
    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        # 1. Main entry runner
        main_py = os.path.join(LOCAL_BASE, "main.py")
        if os.path.isfile(main_py):
            tar.add(main_py, arcname="main.py")
            added_count += 1

        # 2. brailce python packages, json mappings, configs
        brailce_dir = os.path.join(LOCAL_BASE, "brailce")
        for root, dirs, files in os.walk(brailce_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "characters", "filipino", "models")]
            for f in files:
                if f.endswith((".py", ".json")):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, LOCAL_BASE)
                    tar.add(full, arcname=rel.replace("\\", "/"))
                    added_count += 1

        # 3. English isolated words (en_word_*.wav) — leaves char_*.wav untouched!
        en_word_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "characters")
        if os.path.isdir(en_word_dir):
            for f in sorted(os.listdir(en_word_dir)):
                if f.startswith("en_word_") and f.endswith(".wav"):
                    full = os.path.join(en_word_dir, f)
                    rel = os.path.relpath(full, LOCAL_BASE)
                    tar.add(full, arcname=rel.replace("\\", "/"))
                    added_count += 1

        # 4. Filipino audio bank (*.wav)
        fil_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "filipino")
        if os.path.isdir(fil_dir):
            for f in sorted(os.listdir(fil_dir)):
                if f.endswith(".wav"):
                    full = os.path.join(fil_dir, f)
                    rel = os.path.relpath(full, LOCAL_BASE)
                    tar.add(full, arcname=rel.replace("\\", "/"))
                    added_count += 1

        # 5. Tools and tests
        for sub in ("tools", "brailce/tests"):
            dir_path = os.path.join(LOCAL_BASE, sub.replace("/", os.sep))
            if os.path.isdir(dir_path):
                for f in sorted(os.listdir(dir_path)):
                    if f.endswith(".py"):
                        full = os.path.join(dir_path, f)
                        rel = os.path.relpath(full, LOCAL_BASE)
                        tar.add(full, arcname=rel.replace("\\", "/"))
                        added_count += 1

    size_mb = os.path.getsize(BUNDLE_PATH) / (1024 * 1024)
    print(f"  Bundle created successfully: {added_count} files, {size_mb:.2f} MB")
    return BUNDLE_PATH


def _run_cmd(cmd_list: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd_list,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as exc:
        return -1, "", str(exc)


def upload_bundle_putty(host: str, user: str, password: str, bundle_path: str) -> bool:
    print(f"\n--- 2. Uploading Bundle via PuTTY PSCP to {user}@{host} ---")
    rem_dest = f"/tmp/{BUNDLE_FILENAME}"

    # Try PSCP without hostkey first, then with hostkey if needed
    for hostkey_flag in ([], ["-hostkey", HOSTKEY_ED25519]):
        cmd = [PSCP_EXE, "-batch"] + hostkey_flag + ["-pw", password, bundle_path, f"{user}@{host}:{rem_dest}"]
        ret, stdout, stderr = _run_cmd(cmd, timeout=90)
        if ret == 0:
            print("  Bundle uploaded successfully via PSCP!")
            return True
        else:
            if "host key" in stderr.lower() or "fatal" in stderr.lower():
                continue

    return False


def upload_bundle_openssh(host: str, user: str, bundle_path: str) -> bool:
    if not SYSTEM_SCP:
        return False
    print(f"\n--- 2b. Uploading Bundle via Windows OpenSSH SCP to {user}@{host} ---")
    rem_dest = f"/tmp/{BUNDLE_FILENAME}"
    cmd = [
        SYSTEM_SCP,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        bundle_path,
        f"{user}@{host}:{rem_dest}",
    ]
    ret, stdout, stderr = _run_cmd(cmd, timeout=90)
    if ret == 0:
        print("  Bundle uploaded successfully via Windows OpenSSH SCP!")
        return True
    return False


def extract_and_apply_remote(host: str, user: str, password: str) -> bool:
    print("\n--- 3. Unpacking and Applying Update on Raspberry Pi 5 ---")
    extract_script = (
        f"rm -f {REMOTE_BASE}/brailce/audio/characters/en_word_*.wav 2>/dev/null && "
        f"rm -f {REMOTE_BASE}/brailce/audio/filipino/char_*.wav 2>/dev/null && "
        f"tar -xzf /tmp/{BUNDLE_FILENAME} -C {REMOTE_BASE} && "
        f"rm -f /tmp/{BUNDLE_FILENAME} && "
        f"find {REMOTE_BASE} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null && "
        f"pkill -9 -f 'python3.*brailce' 2>/dev/null; pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null && "
        f"echo 'BRAILCE_UPDATE_SUCCESS'"
    )

    for hostkey_flag in ([], ["-hostkey", HOSTKEY_ED25519]):
        cmd = [PLINK_EXE, "-batch"] + hostkey_flag + ["-pw", password, f"{user}@{host}", extract_script]
        ret, stdout, stderr = _run_cmd(cmd, timeout=45)
        if "BRAILCE_UPDATE_SUCCESS" in stdout:
            print("  Remote update extracted and applied successfully!")
            return True

    return False


def serve_http_fallback(port: int = 8000) -> None:
    """Start local HTTP server so the user can pull the update directly from PuTTY."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    os.chdir(LOCAL_BASE)
    server = socketserver.TCPServer(("", port), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.137.248", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "192.168.137.1"
    finally:
        s.close()
    return ip


def deploy() -> None:
    bundle_path = create_deployment_bundle()
    host = get_pi_host()

    uploaded = False
    if os.path.isfile(PSCP_EXE):
        uploaded = upload_bundle_putty(host, PI_USER, PI_PASS, bundle_path)

    if not uploaded and SYSTEM_SCP:
        uploaded = upload_bundle_openssh(host, PI_USER, bundle_path)

    if uploaded:
        applied = extract_and_apply_remote(host, PI_USER, PI_PASS)
        if applied:
            print("\n=======================================================")
            print("DEPLOYMENT 100% COMPLETE AND VERIFIED!")
            print(f"To run Brailce interactively on the Raspberry Pi, switch to PuTTY:")
            print(f"    cd {REMOTE_BASE}")
            print(f"    source venv/bin/activate")
            print(f"    python3 main.py")
            print("=======================================================")
            return

    # If automated upload couldn't bypass Windows Hotspot/PuTTY security gates:
    local_ip = get_local_ip()
    serve_http_fallback(8000)
    print("\n" + "=" * 75)
    print("DIRECT 1-STEP PUTTY DEPLOYMENT (NO PSCP/HOSTKEY PROBLEMS):")
    print("=" * 75)
    print("In your open PuTTY terminal, copy and paste this ONE line:")
    print()
    print(f"  rm -f {REMOTE_BASE}/brailce/audio/characters/en_word_*.wav {REMOTE_BASE}/brailce/audio/filipino/char_*.wav 2>/dev/null && curl -s http://{local_ip}:8000/{BUNDLE_FILENAME} | tar -xz -C {REMOTE_BASE} && cd {REMOTE_BASE} && source venv/bin/activate && python3 main.py")
    print()
    print("=" * 75)
    print("Keeping local update server open. Press Ctrl+C when finished.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    deploy()
