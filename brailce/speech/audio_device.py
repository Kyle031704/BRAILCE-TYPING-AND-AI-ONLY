"""
Brailce Audio Device Coordinator
================================

Coordinates exclusive access to the I2S DAC (MAX98357A) and ALSA sound card.
Ensures seamless, clean handoff between eSpeak NG (instant keystrokes)
and Piper Neural TTS (words and sentences) without ALSA device collisions,
busy errors, or audio truncation.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class AudioDeviceCoordinator:
    """
    Singleton audio coordinator that manages DAC access and active playback
    processes across both eSpeak NG and Piper TTS engines.
    """

    _lock = threading.RLock()
    _active_process: Optional[subprocess.Popen] = None
    _active_owner: Optional[str] = None

    @classmethod
    def register_process(cls, proc: subprocess.Popen, owner: str) -> None:
        """Register the currently playing audio process."""
        with cls._lock:
            cls._active_process = proc
            cls._active_owner = owner

    @classmethod
    def unregister_process(cls, proc: Optional[subprocess.Popen] = None) -> None:
        """Unregister the active audio process upon completion."""
        with cls._lock:
            if proc is None or cls._active_process is proc:
                cls._active_process = None
                cls._active_owner = None

    @classmethod
    def interrupt_active(cls, target_owner: Optional[str] = None) -> None:
        """
        Immediately terminate the currently playing audio process.
        If target_owner is specified, only interrupts if the process belongs to that owner.
        """
        with cls._lock:
            proc = cls._active_process
            owner = cls._active_owner
            if proc is not None:
                if target_owner is None or owner == target_owner:
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=0.2)
                            except Exception:
                                proc.kill()
                    except Exception as exc:
                        logger.debug("Error interrupting active audio process (%s): %s", owner, exc)
                    finally:
                        cls._active_process = None
                        cls._active_owner = None

    @classmethod
    def get_lock(cls) -> threading.RLock:
        """Get the shared re-entrant audio device lock."""
        return cls._lock

