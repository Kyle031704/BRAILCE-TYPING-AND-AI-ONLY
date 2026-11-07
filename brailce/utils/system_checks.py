"""
Brailce Utility – System Checks
==================================

Runtime checks for system dependencies (eSpeak NG, Ollama, etc.).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SystemCheck:
    """Result of a single system dependency check."""
    name: str
    available: bool
    version: str = ""
    message: str = ""


def check_espeak() -> SystemCheck:
    """Check if eSpeak NG is installed and accessible."""
    for cmd in ["espeak-ng", "espeak"]:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip().split("\n")[0]
                return SystemCheck(
                    name="eSpeak NG",
                    available=True,
                    version=version,
                    message=f"Found: {cmd}",
                )
        except FileNotFoundError:
            continue
        except Exception as exc:
            return SystemCheck(
                name="eSpeak NG",
                available=False,
                message=f"Error checking {cmd}: {exc}",
            )

    try:
        import pyttsx3  # noqa: F401
        return SystemCheck(
            name="Speech (TTS)",
            available=True,
            version="pyttsx3 / Windows SAPI",
            message="Using pyttsx3 fallback for speech audio.",
        )
    except Exception:
        pass

    return SystemCheck(
        name="eSpeak NG",
        available=False,
        message=(
            "eSpeak NG not found. Install from: "
            "https://github.com/espeak-ng/espeak-ng or pip install pyttsx3"
        ),
    )


def check_ollama(host: str = "http://127.0.0.1:11434") -> SystemCheck:
    """Check if Ollama is running and accessible."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return SystemCheck(
                    name="Ollama",
                    available=True,
                    message=f"Connected to {host}",
                )
    except Exception as exc:
        return SystemCheck(
            name="Ollama",
            available=False,
            message=f"Cannot reach Ollama at {host}: {exc}",
        )

    return SystemCheck(
        name="Ollama",
        available=False,
        message=f"Ollama not responding at {host}",
    )


def run_all_checks(ollama_host: str = "http://127.0.0.1:11434") -> List[SystemCheck]:
    """Run all system dependency checks and return results."""
    checks = [
        check_espeak(),
        check_ollama(ollama_host),
    ]

    for check in checks:
        level = logging.INFO if check.available else logging.WARNING
        logger.log(
            level,
            "System check [%s]: %s — %s",
            check.name,
            "OK" if check.available else "MISSING",
            check.message,
        )

    return checks
