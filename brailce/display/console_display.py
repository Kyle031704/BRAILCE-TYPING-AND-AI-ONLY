"""
Brailce Console Display – Rich Aesthetic Terminal Interface
=============================================================

A sleek, high-contrast, modern terminal dashboard for the Brailce device.
Renders cohesive Unicode card layouts, giant high-contrast Braille dots,
real-time sentence buffer with active cursor, and live status bars.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Optional

from brailce.core.events import AppMode
from brailce.display.base import DisplayDevice
from brailce.i18n.strings import get_string, get_mode_name

logger = logging.getLogger(__name__)


# Modern ANSI 256-color palette
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Theme colors
    CYAN = "\033[38;5;51m"
    DEEP_CYAN = "\033[38;5;38m"
    EMERALD = "\033[38;5;48m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GOLD = "\033[38;5;220m"
    AMBER = "\033[38;5;214m"
    VIOLET = "\033[38;5;141m"
    WHITE = "\033[38;5;255m"
    SLATE = "\033[38;5;244m"
    DARK_SLATE = "\033[38;5;238m"
    RED = "\033[38;5;196m"


class ConsoleDisplay(DisplayDevice):
    """
    High-contrast aesthetic console display with cohesive card rendering.
    """

    def __init__(self, use_colors: bool = True) -> None:
        self._use_colors = use_colors and self._supports_color()
        self._current_mode: Optional[AppMode] = None
        self._last_cam_print: float = 0.0
        self._language: str = "en"

    def set_language(self, language: str) -> None:
        self._language = language

    @property
    def driver_name(self) -> str:
        return "Console Display (Terminal)"

    def initialize(self) -> None:
        if sys.platform == "win32":
            try:
                if hasattr(sys.stdout, "reconfigure"):
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        logger.info("ConsoleDisplay initialized.")
        c = self._c()
        print(f"\n{c.CYAN}{c.BOLD}╔════════════════════════════════════════════════════════════╗")
        print(f"║  BRAILCE ASSISTIVE SYSTEM — Console Dashboard Active       ║")
        print(f"╚════════════════════════════════════════════════════════════╝{c.RESET}\n")

    def shutdown(self) -> None:
        c = self._c()
        print(f"\n{c.VIOLET}{c.BOLD}╭────────────────────────────────────────────────────────────╮")
        print(f"│  BRAILCE ASSISTIVE DEVICE — System Offline                 │")
        print(f"╰────────────────────────────────────────────────────────────╯{c.RESET}\n")

    def _c(self):
        if self._use_colors:
            return _Colors
        return type("NoColor", (), {k: "" for k in dir(_Colors) if not k.startswith("_")})

    def show_mode(self, mode: AppMode) -> None:
        self._current_mode = mode
        mode_name = get_mode_name(mode, language=self._language)
        c = self._c()
        
        instruction_keys = {
            AppMode.FREE_INPUT: "FREE_INPUT_CONSOLE_INSTRUCTION",
            AppMode.MATH: "MATH_CONSOLE_INSTRUCTION",
            AppMode.AI_ASSISTANT: "AI_CONSOLE_INSTRUCTION",
            AppMode.SCANNER: "SCANNER_CONSOLE_INSTRUCTION",
        }
        inst_key = instruction_keys.get(mode, "FREE_INPUT_CONSOLE_INSTRUCTION")
        instructions = get_string(inst_key, language=self._language)

        print(f"{c.CYAN}{c.BOLD}[MODE]{c.RESET} {c.GOLD}{mode_name}{c.RESET}")
        print(f"{c.CYAN}╭────────────────────────────────────────────────────────────╮")
        print(f"│  {c.WHITE}{c.BOLD}ACTIVE MODE:{c.RESET} {c.EMERALD}{mode_name:<44}{c.CYAN}│")
        if instructions:
            print(f"│  {c.SLATE}{instructions[:56]:<56}{c.CYAN}│")
        print(f"╰────────────────────────────────────────────────────────────╯{c.RESET}\n")

    def show_character(self, character: str) -> None:
        c = self._c()
        print(f"{c.GREEN}{c.BOLD}[CHARACTER]{c.RESET} {c.EMERALD}{c.BOLD}[ {character} ]{c.RESET}")

    def show_braille(
        self,
        unicode_char: str,
        dots_string: str,
        dots_visual: str,
    ) -> None:
        c = self._c()
        print(f"{c.GREEN}{c.BOLD}[BRAILLE]{c.RESET}")
        print(f"  Unicode : {c.GOLD}{unicode_char}{c.RESET}")
        print(f"  Dots    : {c.CYAN}{dots_string}{c.RESET}")
        print(f"  Visual  :")
        for row in dots_visual.split("\n"):
            r_str = row.strip()
            if r_str:
                colored_row = r_str.replace("●", f"{c.GOLD}●{c.RESET}").replace("○", f"{c.DARK_SLATE}○{c.RESET}") if self._use_colors else r_str
                print(f"    {colored_row}")

    def show_text_buffer(self, text: str) -> None:
        c = self._c()
        display_text = text if text else "(empty)"
        cursor = f"{c.GOLD}█{c.RESET}" if text else ""
        print(f"{c.GREEN}{c.BOLD}[TEXT BUFFER]{c.RESET} {c.WHITE}{display_text}{cursor}{c.RESET}")

    def show_status(self, message: str) -> None:
        c = self._c()
        print(f"{c.GREEN}{c.BOLD}[STATUS]{c.RESET} {c.AMBER}{message}{c.RESET}")

    def show_error(self, message: str) -> None:
        c = self._c()
        print(f"{c.RED}{c.BOLD}[ERROR] {message}{c.RESET}", file=sys.stderr)

    def show_ai_response(self, response: str, query: Optional[str] = None) -> None:
        c = self._c()
        if query:
            print(f"{c.GREEN}{c.BOLD}[AI QUERY]{c.RESET} {query}")
        print(f"{c.GREEN}{c.BOLD}[AI RESPONSE]{c.RESET} {c.WHITE}{response}{c.RESET}")
        print(f"\n{c.VIOLET}{c.BOLD}╭── [ AI ASSISTANT ANSWER ] ─────────────────────────────────╮{c.RESET}")
        print(f"│  {c.WHITE}{response}{c.RESET}")
        print(f"{c.VIOLET}{c.BOLD}╰────────────────────────────────────────────────────────────╯{c.RESET}\n")

    def show_ai_processing(self, query: Optional[str] = None) -> None:
        c = self._c()
        msg = f"Processing query: {query}" if query else "Processing... please wait."
        print(f"{c.GREEN}{c.BOLD}[AI]{c.RESET} {c.GOLD}{msg}{c.RESET}")

    def show_math_result(
        self,
        equation: str,
        result: str,
        explanation: str,
        short_step: Optional[str] = None,
        language: str = "en",
    ) -> None:
        from brailce.i18n.strings import get_string
        c = self._c()
        title = get_string("MATH_CARD_TITLE", language)
        lbl_eq = get_string("MATH_LABEL_EQUATION", language)
        lbl_st = get_string("MATH_LABEL_STEPS", language)
        lbl_ex = get_string("MATH_LABEL_EXPLANATION", language)
        print(f"\n{c.CYAN}{c.BOLD}══════════ {title} ══════════{c.RESET}")
        print(f"{c.GREEN}{c.BOLD}{lbl_eq:<11}:{c.RESET} {c.YELLOW}{equation} = {result}{c.RESET}")
        if short_step:
            print(f"{c.GREEN}{c.BOLD}{lbl_st:<11}:{c.RESET} {c.YELLOW}{short_step}{c.RESET}")
        print(f"{c.GREEN}{c.BOLD}{lbl_ex:<11}:{c.RESET} {c.YELLOW}{explanation}{c.RESET}")
        print(f"{c.CYAN}{c.BOLD}══════════════════════════════════════════════{c.RESET}\n")

    def show_camera_preview(
        self,
        frame: Any,
        status: Optional[str] = None,
        overlays: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Render live camera viewfinder, draw character overlays, and save snapshot."""
        if frame is None:
            return

        now = time.time()

        # 1. Save frame snapshot periodically for viewing on dev PC (throttled to 1 Hz)
        if now - getattr(self, "_last_cam_save", 0.0) > 1.0:
            self._last_cam_save = now
            try:
                import cv2
                if hasattr(frame, "save"):
                    frame.save("camera_preview.jpg")
                else:
                    cv2.imwrite("camera_preview.jpg", frame)
            except Exception:
                pass

        # 2. Render live terminal viewfinder card every 1.5 seconds or when overlays update
        has_overlays = bool(overlays)
        if (now - self._last_cam_print > 1.5) or has_overlays:
            self._last_cam_print = now
            c = self._c()
            lang = getattr(self, "_language", "en") or "en"
            st = status if status else get_string("SCANNER_STATUS_POSITION", language=lang)
            frame_saved_str = get_string("SCANNER_FRAME_SAVED", language=lang)
            target_str = get_string("SCANNER_TARGET_CENTERED", language=lang)
            print(f"\n{c.CYAN}{c.BOLD}╭── [ CAMERA VIEWFINDER: SCANNER MODE ] ─────────────────────╮{c.RESET}")
            print(f"│  {c.EMERALD}● {frame_saved_str[:54]:<54}{c.CYAN} │")
            print(f"│  {target_str[:56]:<56}│")
            if overlays:
                overlay_chars = []
                for ov in overlays:
                    ch = ov.get("char", "")
                    if ov.get("active"):
                        overlay_chars.append(f"{c.YELLOW}{c.BOLD}[{ch}]{c.RESET}{c.CYAN}")
                    else:
                        overlay_chars.append(ch)
                overlay_line = f"Decoded: {' '.join(overlay_chars)}"
                print(f"│  {overlay_line[:56]:<56}│")
            print(f"│  {c.GOLD}{st:<56}{c.CYAN}  │")
            print(f"{c.CYAN}{c.BOLD}╰────────────────────────────────────────────────────────────╯{c.RESET}")

        # 3. If local desktop window is supported and GUI available, display OpenCV window
        try:
            import cv2
            if hasattr(frame, "shape"):
                cv2.imshow("Brailce - Camera Viewfinder", frame)
                cv2.waitKey(1)
        except Exception:
            pass

    def close_camera_preview(self) -> None:
        try:
            import cv2
            cv2.destroyWindow("Brailce - Camera Viewfinder")
        except Exception:
            pass

    def clear(self) -> None:
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def _supports_color() -> bool:
        if os.name == "nt":
            return True
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
