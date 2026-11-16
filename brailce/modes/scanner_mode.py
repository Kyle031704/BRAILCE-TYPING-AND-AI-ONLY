"""
Brailce Mode — Scanner Mode
===========================

Live viewfinder with cheap classical Braille presence detection (~3-5 Hz,
hysteresis + cooldown), sharpest still-frame capture on Enter, classical dot-grid
Grade 1 Braille decoding, and systematic letter-by-letter -> word-by-word readout
that cancels cleanly on any key.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from brailce.core.events import AppMode, EventType, InputEvent
from brailce.i18n.strings import get_mode_name, get_string
from brailce.modes.base_mode import BaseMode
from brailce.vision.camera import CameraNotConnectedError, compute_sharpness

logger = logging.getLogger(__name__)


class ScannerMode(BaseMode):
    """Optical Braille Recognition (OBR) camera scanner mode."""

    # Presence detection tuning (~3-4 Hz)
    PRESENCE_POLL_S: float = 0.28
    PRESENCE_POSITIVE_FRAMES: int = 3
    PRESENCE_NEGATIVE_FRAMES: int = 4
    ANNOUNCEMENT_COOLDOWN_S: float = 6.0

    # Readout pacing (seconds; overridable in tests)
    CHAR_PACE_S: float = 0.42
    WORD_PAUSE_S: float = 0.35
    LINE_PAUSE_S: float = 0.60

    _REASON_TO_STRING_KEY = {
        "blurry": "SCANNER_BLURRY",
        "too_small": "SCANNER_MOVE_CLOSER",
        "too_large": "SCANNER_MOVE_FARTHER",
        "no_braille": "SCANNER_NO_BRAILLE",
    }

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.scanned_text: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._is_scanning: bool = False
        self._is_reading: bool = False
        self._preview_running: bool = False
        self._preview_thread: Optional[threading.Thread] = None
        self._detector_thread: Optional[threading.Thread] = None
        self._readout_thread: Optional[threading.Thread] = None
        self._stop_readout_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._sharp_frames: Deque[Tuple[float, Any]] = deque(maxlen=5)
        self._latest_frame: Any = None
        self._braille_detected: Optional[bool] = None
        self._camera_error_reported: bool = False
        self._last_announce_time: float = 0.0

    @property
    def mode_type(self) -> AppMode:
        return AppMode.SCANNER

    @property
    def display_name(self) -> str:
        lang = self.state.language if self.state else "en"
        return get_mode_name(self.mode_type, lang)

    def _lang(self) -> str:
        return self.state.language if self.state else "en"

    def _announce_capital_sign(self) -> bool:
        cfg = getattr(self.app, "config", None)
        if cfg and hasattr(cfg, "get"):
            return bool(cfg.get("vision", "announce_capital_sign", default=False))
        return False

    def _stop_readout(self, announce_stopped: bool = False) -> None:
        """Stop active letter-by-letter/word-by-word readout immediately."""
        was_reading = self._is_reading
        self._stop_readout_event.set()
        self._is_reading = False
        if getattr(self.app, "speech", None):
            self.app.speech.stop()
        if announce_stopped and was_reading and getattr(self.app, "display", None):
            self.app.display.show_status(get_string("SCANNER_STOPPED", self._lang()))

    def handle_event(self, event: InputEvent) -> None:
        """Handle user input in Scanner Mode."""
        if self._is_reading:
            # Any key cleanly interrupts active speech readout
            self._stop_readout(announce_stopped=True)
            if event.event_type == EventType.CLEAR_PRESSED:
                self._reset_scan_state()
            return

        if self._is_scanning or self.state.ai_processing:
            return

        if event.event_type == EventType.ENTER_PRESSED:
            if self._last_result and self.scanned_text:
                self._start_readout_thread(self._last_result)
            else:
                self._start_scan()
        elif event.event_type == EventType.CLEAR_PRESSED:
            self._reset_scan_state()

    def _reset_scan_state(self) -> None:
        """Clear previously scanned text and return to viewfinder ready state."""
        lang = self._lang()
        self.scanned_text = None
        self._last_result = None
        self._braille_detected = None
        if getattr(self.app, "display", None):
            ready = get_string("SCANNER_READY", lang)
            if hasattr(self.app.display, "show_scanner_viewfinder"):
                self.app.display.show_scanner_viewfinder(status=ready)
            else:
                self.app.display.show_status(ready)

    def _report_camera_missing(self) -> None:
        """Surface missing camera clearly via LCD + Voice once (no fake Braille)."""
        if self._camera_error_reported:
            return
        self._camera_error_reported = True
        msg = get_string("SCANNER_CAMERA_NOT_CONNECTED", self._lang())
        if getattr(self.app, "display", None):
            if hasattr(self.app.display, "show_error"):
                self.app.display.show_error(msg)
            else:
                self.app.display.show_status(msg)
        if getattr(self.app, "speech", None):
            self.app.speech.speak_text(msg, interrupt=True)

    def _detection_worker(self) -> None:
        """
        Cheap classical Braille presence monitor (~3-4 Hz) with hysteresis + cooldown.
        Speaks SCANNER_BRAILLE_DETECTED once per sheet placement without nagging.
        """
        pos_streak = 0
        neg_streak = 0

        while self._preview_running:
            try:
                if self._is_scanning or self._is_reading or not getattr(self.app, "obr", None):
                    time.sleep(self.PRESENCE_POLL_S)
                    continue

                with self._frame_lock:
                    frame = self._latest_frame

                if frame is None:
                    time.sleep(self.PRESENCE_POLL_S)
                    continue

                detected = bool(self.app.obr.detect_braille_presence(frame))
                if detected:
                    pos_streak += 1
                    neg_streak = 0
                    if pos_streak >= self.PRESENCE_POSITIVE_FRAMES:
                        became_detected = self._braille_detected is not True
                        self._braille_detected = True
                        now = time.monotonic()
                        if (
                            became_detected
                            and not self.scanned_text
                            and (now - self._last_announce_time) >= self.ANNOUNCEMENT_COOLDOWN_S
                        ):
                            self._last_announce_time = now
                            if getattr(self.app, "speech", None):
                                self.app.speech.speak_text(
                                    get_string("SCANNER_BRAILLE_DETECTED", self._lang()),
                                    interrupt=False,
                                )
                else:
                    neg_streak += 1
                    pos_streak = 0
                    if neg_streak >= self.PRESENCE_NEGATIVE_FRAMES:
                        self._braille_detected = False
            except Exception as exc:
                logger.debug("Presence detection error: %s", exc)

            time.sleep(self.PRESENCE_POLL_S)

    def _preview_loop(self) -> None:
        """Live camera viewfinder loop maintaining a rolling buffer of sharpest frames."""
        while self._preview_running:
            try:
                cam = getattr(self.app, "camera", None)
                if not self._is_scanning and cam is not None:
                    if hasattr(cam, "is_available") and not cam.is_available():
                        self._report_camera_missing()
                        time.sleep(0.5)
                        continue

                    try:
                        frame = cam.capture_image()
                    except CameraNotConnectedError:
                        self._report_camera_missing()
                        time.sleep(0.5)
                        continue
                    except Exception as exc:
                        logger.debug("Preview capture error: %s", exc)
                        frame = None

                    if frame is not None:
                        self._camera_error_reported = False
                        sharpness = compute_sharpness(frame)
                        with self._frame_lock:
                            self._latest_frame = frame
                            self._sharp_frames.append((sharpness, frame))

                        if getattr(self.app, "display", None) and not self._is_reading:
                            lang = self._lang()
                            if self.scanned_text:
                                status = self.scanned_text
                            elif self._braille_detected is True:
                                status = get_string("SCANNER_BRAILLE_DETECTED", lang)
                            else:
                                status = get_string("SCANNER_READY", lang)
                            try:
                                self.app.display.show_camera_preview(frame, status=status)
                            except Exception as disp_exc:
                                logger.debug("Display show_camera_preview error: %s", disp_exc)
            except Exception as loop_exc:
                logger.warning("Preview loop iteration warning: %s", loop_exc)

            time.sleep(0.06)

    def _capture_best_still(self) -> Optional[Any]:
        """Acquire full-res still (sharpest of N) or sharpest buffered preview frame."""
        cam = getattr(self.app, "camera", None)
        if cam is None or (hasattr(cam, "is_available") and not cam.is_available()):
            self._report_camera_missing()
            return None

        try:
            if hasattr(cam, "capture_sharpest"):
                frame = cam.capture_sharpest(n=5)
                if frame is not None:
                    return frame
            if hasattr(cam, "capture_still"):
                frame = cam.capture_still()
                if frame is not None:
                    return frame
            return cam.capture_image()
        except CameraNotConnectedError:
            self._report_camera_missing()
            return None
        except Exception as exc:
            logger.warning("Still capture failed, falling back to sharpest preview: %s", exc)

        with self._frame_lock:
            if self._sharp_frames:
                return max(self._sharp_frames, key=lambda item: item[0])[1]
            return self._latest_frame

    def _start_scan(self) -> None:
        """Start full capture + classical OBR decode in background thread."""
        self._is_scanning = True
        self.scanned_text = None
        self._last_result = None
        lang = self._lang()

        if getattr(self.app, "display", None):
            self.app.display.show_status(get_string("SCANNER_SCANNING", lang))
        if getattr(self.app, "speech", None):
            self.app.speech.speak_text(get_string("SCANNER_SCANNING", lang), interrupt=False)

        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self) -> None:
        """Capture sharpest image, run OBR engine, report actionable error or start readout."""
        lang = self._lang()
        try:
            if not getattr(self.app, "obr", None):
                self._report_camera_missing()
                return

            image = self._capture_best_still()
            if image is None:
                if not self._camera_error_reported:
                    msg = get_string("SCANNER_NO_BRAILLE", lang)
                    if getattr(self.app, "display", None):
                        self.app.display.show_status(msg)
                    if getattr(self.app, "speech", None):
                        self.app.speech.speak_text(msg, interrupt=True)
                return

            result = self.app.obr.process_image(image)
            if not result.get("readable") or not str(result.get("text", "")).strip():
                reason = str(result.get("reason", "no_braille"))
                key = self._REASON_TO_STRING_KEY.get(reason, "SCANNER_NO_BRAILLE")
                msg = get_string(key, lang)
                self._braille_detected = False
                if getattr(self.app, "display", None):
                    self.app.display.show_status(msg)
                if getattr(self.app, "speech", None):
                    self.app.speech.speak_text(msg, interrupt=True)
                return

            self.scanned_text = str(result["text"]).strip()
            self._last_result = result
            if getattr(self.app, "display", None):
                self.app.display.show_text_buffer(self.scanned_text)
        finally:
            self._is_scanning = False

        if self._last_result is not None:
            self._start_readout_thread(self._last_result)

    def _start_readout_thread(self, result: Dict[str, Any]) -> None:
        """Launch interruptible letter-by-letter + word-by-word readout."""
        self._stop_readout(announce_stopped=False)
        self._stop_readout_event.clear()
        self._is_reading = True
        self._readout_thread = threading.Thread(
            target=self._readout_worker,
            args=(result,),
            daemon=True,
        )
        self._readout_thread.start()

    def _speak_completed_word(self, word_chars: List[str], has_unknown: bool) -> None:
        """Speak an assembled word/number once all of its cells have been spelled out."""
        speech = getattr(self.app, "speech", None)
        if not speech or not word_chars or has_unknown:
            return
        word = "".join(word_chars)
        if word.isdigit():
            speech.speak_text(word, interrupt=False)
        elif word.isalpha():
            if hasattr(speech, "speak_word_clean"):
                speech.speak_word_clean(word, interrupt=False)
            else:
                speech.speak_word(word)
        else:
            speech.speak_text(word, interrupt=False)

    def _readout_worker(self, result: Dict[str, Any]) -> None:
        """
        Systematically read decoded cells:
          1. Each cell letter/digit spoken & displayed (show_braille + show_character)
          2. Completed word/number spoken cleanly at word end
          3. Unknown cell spoken as SCANNER_UNKNOWN_CELL
        Interrupts cleanly whenever _stop_readout_event is set.
        """
        lang = self._lang()
        speech = getattr(self.app, "speech", None)
        display = getattr(self.app, "display", None)
        cells: List[Dict[str, Any]] = list(result.get("cells") or [])

        try:
            if not cells:
                text = str(result.get("text", "")).strip()
                if text and speech:
                    speech.speak_text(text, interrupt=False)
                return

            announce_cap = self._announce_capital_sign()
            cur_word: List[str] = []
            cur_has_unknown = False
            prev_word_idx: Optional[int] = None
            prev_line_idx: Optional[int] = None

            for cell in cells:
                if self._stop_readout_event.is_set():
                    return

                word_idx = cell.get("word_index", 0)
                line_idx = cell.get("line_index", 0)

                if prev_word_idx is not None and (word_idx != prev_word_idx or line_idx != prev_line_idx):
                    self._speak_completed_word(cur_word, cur_has_unknown)
                    cur_word = []
                    cur_has_unknown = False
                    pause = self.LINE_PAUSE_S if (prev_line_idx is not None and line_idx != prev_line_idx) else self.WORD_PAUSE_S
                    if self._stop_readout_event.wait(pause):
                        return

                prev_word_idx = word_idx
                prev_line_idx = line_idx

                kind = cell.get("kind", "char")
                ch = str(cell.get("char", ""))
                dots_str = str(cell.get("dots_str", ""))

                if kind == "indicator":
                    if ch == "^" and announce_cap and speech:
                        speech.speak_text(get_string("SCANNER_CAPITAL", lang), interrupt=False)
                        if self._stop_readout_event.wait(self.CHAR_PACE_S):
                            return
                    continue

                if display:
                    if hasattr(display, "show_character"):
                        display.show_character(ch)
                    if dots_str and hasattr(display, "show_braille"):
                        display.show_braille(ch, dots_str, "")

                if kind == "unknown" or ch == "?":
                    cur_has_unknown = True
                    cur_word.append("?")
                    if speech:
                        speech.speak_text(get_string("SCANNER_UNKNOWN_CELL", lang), interrupt=False)
                elif ch:
                    cur_word.append(ch)
                    if speech:
                        speech.speak_character(ch)

                if self._stop_readout_event.wait(self.CHAR_PACE_S):
                    return

            if cur_word and not self._stop_readout_event.is_set():
                self._speak_completed_word(cur_word, cur_has_unknown)
        finally:
            self._is_reading = False
            if display and self.scanned_text and not self._stop_readout_event.is_set():
                display.show_text_buffer(self.scanned_text)

    def on_enter(self) -> None:
        logger.info("Entered Scanner Mode.")
        self.scanned_text = None
        self._last_result = None
        self._is_scanning = False
        self._is_reading = False
        self._braille_detected = None
        self._latest_frame = None
        self._sharp_frames.clear()
        self._camera_error_reported = False
        self._stop_readout_event.clear()

        if getattr(self.app, "display", None):
            self.app.display.clear()
            self.app.display.show_mode(self.mode_type)
            ready = get_string("SCANNER_READY", self._lang())
            if hasattr(self.app.display, "show_scanner_viewfinder"):
                self.app.display.show_scanner_viewfinder(status=ready)
            else:
                self.app.display.show_status(ready)

        self._preview_running = True
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True, name="scanner-preview")
            self._preview_thread.start()
        if self._detector_thread is None or not self._detector_thread.is_alive():
            self._detector_thread = threading.Thread(target=self._detection_worker, daemon=True, name="scanner-detector")
            self._detector_thread.start()

    def on_exit(self) -> None:
        logger.info("Exited Scanner Mode.")
        self._preview_running = False
        self._is_scanning = False
        self._stop_readout(announce_stopped=False)
        self._braille_detected = None
        self._latest_frame = None
        self._sharp_frames.clear()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=0.2)
        if self._detector_thread and self._detector_thread.is_alive():
            self._detector_thread.join(timeout=0.2)
        if getattr(self.app, "display", None):
            self.app.display.close_camera_preview()
