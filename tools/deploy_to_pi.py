"""
Automated Deployment Script to Raspberry Pi 5
==============================================

Packages updated Brailce source code, configs, and audio banks into a single
archive bundle (`brailce_update.tar.gz`) and deploys it over SSH/SCP or a local
HTTP fallback. Connection details are loaded exclusively from environment
variables or a git-ignored `.env` file in the repository root (see `.env.example`).
"""

from __future__ import annotations

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

LOCAL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_dotenv(env_path: str) -> None:
    """Load simple KEY=VALUE pairs from `.env` if present without overriding env."""
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv(os.path.join(LOCAL_BASE, ".env"))

PUTTY_DIR = os.getenv("PUTTY_DIR", r"C:\Program Files\PuTTY")
PLINK_EXE = os.path.join(PUTTY_DIR, "plink.exe") if os.path.isdir(PUTTY_DIR) else "plink"
PSCP_EXE = os.path.join(PUTTY_DIR, "pscp.exe") if os.path.isdir(PUTTY_DIR) else "pscp"

SYSTEM_SCP = shutil.which("scp")

BUNDLE_FILENAME = "brailce_update.tar.gz"
BUNDLE_PATH = os.path.join(LOCAL_BASE, BUNDLE_FILENAME)

PI_IP = os.getenv("PI_IP", "").strip()
PI_USER = os.getenv("PI_USER", "").strip()
PI_PASS = os.getenv("PI_PASS", "").strip()
PI_REMOTE_DIR = os.getenv("PI_REMOTE_DIR", "/home/pi/brailce").strip()
PI_HOSTKEY = os.getenv("PI_HOSTKEY", "").strip()


def _probe_ip(ip: str, port: int = 22, timeout: float = 0.4) -> Optional[str]:
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


def get_pi_host() -> Optional[str]:
    """Resolve Raspberry Pi IP from PI_IP env var or local mDNS hostname."""
    if PI_IP:
        return PI_IP
    candidates = [os.getenv("PI_MDNS_HOST", "").strip(), "raspberrypi.local"]
    for mdns_host in [h for h in candidates if h]:
        try:
            addr = socket.gethostbyname(mdns_host)
            if _probe_ip(addr):
                return addr
        except Exception:
            pass
    return None


def create_deployment_bundle() -> str:
    """
    Bundle application source packages, configs, and audio banks into `.tar.gz`.
    """
    print(f"\n--- 1. Packaging Update Bundle ({BUNDLE_FILENAME}) ---")
    if os.path.exists(BUNDLE_PATH):
        try:
            os.remove(BUNDLE_PATH)
        except Exception:
            pass

    added_count = 0
    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        main_py = os.path.join(LOCAL_BASE, "main.py")
        if os.path.isfile(main_py):
            tar.add(main_py, arcname="main.py")
            added_count += 1

        brailce_dir = os.path.join(LOCAL_BASE, "brailce")
        for root, dirs, files in os.walk(brailce_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "characters", "words_en", "filipino", "models")]
            for f in files:
                if f.endswith((".py", ".json")):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, LOCAL_BASE)
                    tar.add(full, arcname=rel.replace("\\", "/"))
                    added_count += 1

        for audio_sub in ("characters", "words_en", "filipino"):
            audio_dir = os.path.join(LOCAL_BASE, "brailce", "audio", audio_sub)
            if os.path.isdir(audio_dir):
                for f in sorted(os.listdir(audio_dir)):
                    if f.endswith(".wav"):
                        full = os.path.join(audio_dir, f)
                        rel = os.path.relpath(full, LOCAL_BASE)
                        tar.add(full, arcname=rel.replace("\\", "/"))
                        added_count += 1

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
    print(f"  Bundle created: {added_count} files, {size_mb:.2f} MB")
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
    hostkey_options = [[], ["-hostkey", PI_HOSTKEY]] if PI_HOSTKEY else [[]]
    for hk in hostkey_options:
        cmd = [PSCP_EXE, "-batch"] + hk + ["-pw", password, bundle_path, f"{user}@{host}:{rem_dest}"]
        ret, _stdout, stderr = _run_cmd(cmd, timeout=90)
        if ret == 0:
            print("  Bundle uploaded successfully via PSCP!")
            return True
        if stderr:
            print(f"  PSCP attempt note: {stderr.strip()}")
    return False


def upload_bundle_openssh(host: str, user: str, bundle_path: str) -> bool:
    if not SYSTEM_SCP:
        return False
    print(f"\n--- 2b. Uploading Bundle via OpenSSH SCP to {user}@{host} ---")
    rem_dest = f"/tmp/{BUNDLE_FILENAME}"
    cmd = [SYSTEM_SCP, bundle_path, f"{user}@{host}:{rem_dest}"]
    ret, _stdout, _stderr = _run_cmd(cmd, timeout=90)
    if ret == 0:
        print("  Bundle uploaded successfully via OpenSSH SCP!")
        return True
    return False


def extract_and_apply_remote(host: str, user: str, password: str) -> bool:
    print("\n--- 3. Unpacking Update on Raspberry Pi 5 ---")
    extract_script = (
        f"mkdir -p {PI_REMOTE_DIR} && "
        f"tar -xzf /tmp/{BUNDLE_FILENAME} -C {PI_REMOTE_DIR} && "
        f"rm -f /tmp/{BUNDLE_FILENAME} && "
        f"find {PI_REMOTE_DIR} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null; "
        f"echo 'BRAILCE_UPDATE_SUCCESS'"
    )
    hostkey_options = [[], ["-hostkey", PI_HOSTKEY]] if PI_HOSTKEY else [[]]
    for hk in hostkey_options:
        cmd = [PLINK_EXE, "-batch"] + hk + (["-pw", password] if password else []) + [f"{user}@{host}", extract_script]
        ret, stdout, _stderr = _run_cmd(cmd, timeout=45)
        if ret == 0 and "BRAILCE_UPDATE_SUCCESS" in stdout:
            print("  Remote update extracted and applied successfully!")
            return True
    return False


def serve_http_fallback(port: int = 8000) -> None:
    """Start local HTTP server so the user can pull the bundle directly from the Pi."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    os.chdir(LOCAL_BASE)
    server = socketserver.TCPServer(("", port), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()


def get_local_ip(target_host: Optional[str]) -> str:
    probe = target_host or "8.8.8.8"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((probe, 80))
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    finally:
        s.close()
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "<PC_LOCAL_IP>"


def deploy() -> None:
    bundle_path = create_deployment_bundle()
    host = get_pi_host()
    remote_dir = os.getenv("PI_REMOTE_DIR", "~/brailce").strip() or "~/brailce"

    if not host or not PI_USER:
        print("\n[INFO] PI_IP / PI_USER not set (configure `.env` from `.env.example` for automated SSH deploy).")
    else:
        uploaded = False
        if PI_PASS and shutil.which(PSCP_EXE):
            uploaded = upload_bundle_putty(host, PI_USER, PI_PASS, bundle_path)
        if not uploaded and SYSTEM_SCP:
            uploaded = upload_bundle_openssh(host, PI_USER, bundle_path)
        if uploaded and extract_and_apply_remote(host, PI_USER, PI_PASS):
            print("\nDEPLOYMENT COMPLETE!")
            print(f"On Raspberry Pi: cd {remote_dir} && source venv/bin/activate && python3 main.py")
            return

    local_ip = get_local_ip(host)
    serve_http_fallback(8000)
    print("\nDIRECT 1-STEP PULL ON RASPBERRY PI (paste into PuTTY):")
    print(f"  mkdir -p {remote_dir} && curl -s http://{local_ip}:8000/{BUNDLE_FILENAME} | tar -xz -C {remote_dir} && cd {remote_dir} && [ -f venv/bin/activate ] && source venv/bin/activate; python3 main.py")
    print("Press Ctrl+C to stop HTTP server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    deploy()
