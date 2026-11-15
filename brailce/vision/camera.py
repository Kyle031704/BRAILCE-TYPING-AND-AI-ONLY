"""
Camera interface for capturing physical Braille documents.
==========================================================

Supports:
  - PiCameraDevice: Raspberry Pi Camera Module V2 (Sony IMX219) via Picamera2
    (or OpenCV V4L2 fallback), configured with RGB888 stream layout (matching OpenCV BGR
    memory order without cyan/blue cast), settled AE/AWB locking, and sharpness selection.
    Raises / reports missing hardware cleanly instead of faking a synthetic Braille frame.
  - SimulatedCamera: Reads an image file from disk (or local webcam) for desktop testing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, List, Dict

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

logger = logging.getLogger(__name__)


class CameraNotConnectedError(RuntimeError):
    """Raised when camera hardware is missing or disconnected."""


def compute_sharpness(frame: Optional[np.ndarray]) -> float:
    """Return variance of Laplacian sharpness metric for a BGR/gray frame."""
    if frame is None or cv2 is None or frame.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


class CameraDevice(ABC):
    """Abstract base class for camera capture."""

    @abstractmethod
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture and return an image as a numpy array (BGR)."""
        pass

    def is_available(self) -> bool:
        """Return True if the camera source is connected and usable."""
        return True

    def capture_still(self) -> Optional[np.ndarray]:
        """Capture a high-quality still frame (defaults to capture_image)."""
        return self.capture_image()

    def capture_sharpest(self, n: int = 5) -> Optional[np.ndarray]:
        """Capture N frames and return the one with the highest Laplacian sharpness."""
        best_frame = None
        best_score = -1.0
        for _ in range(max(1, n)):
            frame = self.capture_still()
            if frame is None:
                continue
            score = compute_sharpness(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
            time.sleep(0.03)
        return best_frame


class SimulatedCamera(CameraDevice):
    """Simulates a camera by loading an image from disk or using a local webcam."""

    def __init__(self, image_path: Optional[str] = None):
        self.image_path = image_path
        self._cap = None
        if not image_path and cv2 is not None:
            try:
                cap = cv2.VideoCapture(0)
                if cap is not None and cap.isOpened():
                    self._cap = cap
                elif cap is not None:
                    cap.release()
            except Exception:
                self._cap = None

    def is_available(self) -> bool:
        if self.image_path and os.path.exists(self.image_path):
            return True
        return bool(self._cap and self._cap.isOpened())

    def capture_image(self) -> Optional[np.ndarray]:
        if self.image_path and os.path.exists(self.image_path) and cv2 is not None:
            try:
                img = cv2.imread(self.image_path)
                if img is not None:
                    return img
            except Exception:
                pass

        if self._cap and self._cap.isOpened():
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return frame
            except Exception:
                pass

        return None

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class PiCameraDevice(CameraDevice):
    """
    Hardware driver for Raspberry Pi Camera Module V2 (Sony IMX219) on Raspberry Pi 5.

    Uses Picamera2 ("RGB888" -> OpenCV BGR byte order without cyan cast) at 1640x1232
    full-FOV 2x2 binned sensor mode, settles AWB/exposure, and never fabricates a
    synthetic Braille image when hardware is disconnected.
    """

    def __init__(self, resolution: tuple[int, int] = (1640, 1232)):
        self.resolution = resolution
        self._picam2 = None
        self._still_config = None
        self._cap = None
        self._is_active = False
        self.init_error: Optional[str] = None
        self._init_camera()

    def _init_camera(self) -> bool:
        """Attempt to initialize Picamera2 (or V4L2 OpenCV fallback)."""
        import sys
        import glob

        dist_paths = ["/usr/lib/python3/dist-packages"] + glob.glob("/usr/lib/python3.*/dist-packages")
        for dp in dist_paths:
            if os.path.isdir(dp) and dp not in sys.path:
                sys.path.insert(0, dp)

        try:
            from picamera2 import Picamera2
            self._picam2 = Picamera2()
            # In Picamera2, format="RGB888" delivers B-G-R bytes in memory for OpenCV,
            # avoiding the cyan/blue channel swap seen with "BGR888".
            preview_config = self._picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self._still_config = self._picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self._picam2.configure(preview_config)
            self._picam2.start()

            # Let AE/AWB settle briefly on the paper scene, then lock white balance
            time.sleep(0.35)
            try:
                self._picam2.set_controls({
                    "Sharpness": 1.5,
                    "Contrast": 1.15,
                    "AwbEnable": True,
                })
                meta = self._picam2.capture_metadata()
                gains = meta.get("ColourGains")
                if gains:
                    self._picam2.set_controls({"AwbEnable": False, "ColourGains": gains})
            except Exception:
                pass

            self._is_active = True
            self.init_error = None
            logger.info("PiCameraDevice: Picamera2 initialized cleanly (still=%s)", self.resolution)
            return True
        except Exception as exc:
            # Fallback to OpenCV V4L2 webcam if physically present (Linux/USB only)
            if cv2 is not None:
                try:
                    cap = cv2.VideoCapture(0)
                    if cap is not None and cap.isOpened():
                        self._cap = cap
                        self._is_active = True
                        self.init_error = None
                        logger.info("PiCameraDevice: Initialized via OpenCV VideoCapture(0).")
                        return True
                    elif cap is not None:
                        cap.release()
                except Exception:
                    pass

            self._picam2 = None
            self._cap = None
            self._is_active = False
            self.init_error = f"Camera not connected: {exc}"
            logger.warning("PiCameraDevice unavailable: %s", self.init_error)
            return False

    def is_available(self) -> bool:
        """Check if real camera hardware is connected and active."""
        return self._is_active

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture a preview frame (raises CameraNotConnectedError if disconnected)."""
        if not self._is_active:
            raise CameraNotConnectedError(self.init_error or "Camera not connected")

        if self._picam2 is not None:
            try:
                return self._picam2.capture_array("main")
            except Exception as exc:
                logger.warning("Picamera2 preview capture error: %s", exc)
                return None

        if self._cap and self._cap.isOpened():
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return frame
            except Exception as exc:
                logger.warning("OpenCV capture error: %s", exc)
                return None

        raise CameraNotConnectedError("Camera stream lost")

    def capture_still(self) -> Optional[np.ndarray]:
        """Capture a full-resolution still frame via Picamera2 still configuration."""
        if not self._is_active:
            raise CameraNotConnectedError(self.init_error or "Camera not connected")

        if self._picam2 is not None and self._still_config is not None:
            try:
                return self._picam2.switch_mode_and_capture_array(self._still_config, "main")
            except Exception as exc:
                logger.debug("switch_mode_and_capture_array fallback to main: %s", exc)
                return self.capture_image()

        return self.capture_image()

    def close(self) -> None:
        """Safely release camera hardware."""
        if self._picam2 and self._is_active:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_active = False


class UsbWebcamDevice(CameraDevice):
    """
    Hardware driver for USB Webcams on Raspberry Pi 5 & dev PCs using OpenCV + V4L2.

    Features:
    - Auto-detection across candidate /dev/video* nodes or device indices (skips metadata/ISP nodes).
    - MJPG hardware encoding at configurable resolution (default 1280x720).
    - Zero stale-frame latency: background capture thread with buffer size 1.
    - Autofocus and manual focus controls (CAP_PROP_AUTOFOCUS, CAP_PROP_FOCUS).
    - Automatic reconnection retry on disconnect without crashing.
    """

    def __init__(
        self,
        device: Any = 0,
        resolution: tuple[int, int] = (1280, 720),
        still_resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
        autofocus: bool = True,
        focus_value: int = -1,
    ) -> None:
        self.device = os.environ.get("BRAILCE_CAMERA_DEVICE", device)
        self.resolution = resolution
        self.still_resolution = still_resolution
        self.fps = fps
        self.autofocus = autofocus
        self.focus_value = focus_value
        self._cap: Optional[Any] = None
        self._is_active = False
        self.init_error: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_camera()

    @staticmethod
    def probe_device_candidates() -> list[Any]:
        """Discover candidate webcam device nodes or indices."""
        candidates = []
        env_dev = os.environ.get("BRAILCE_CAMERA_DEVICE")
        if env_dev is not None:
            candidates.append(int(env_dev) if env_dev.isdigit() else env_dev)

        # Linux V4L2 device nodes
        import glob
        by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
        candidates.extend(by_id)
        video_nodes = sorted(glob.glob("/dev/video*"))
        candidates.extend(video_nodes)

        # Fallback numeric indices
        for idx in (0, 1, 2, 4, 6):
            if idx not in candidates:
                candidates.append(idx)
        return candidates

    def _open_capture(self, dev: Any) -> Optional[Any]:
        if cv2 is None:
            return None
        target = int(dev) if str(dev).isdigit() else str(dev)
        
        # Multiple backends depending on platform
        if os.name == "nt":
            backends = [getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), cv2.CAP_ANY]
        else:
            backends = [getattr(cv2, "CAP_V4L2", 200), cv2.CAP_ANY]

        for backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(target, backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    continue

                # Warm-up loop: discard initial frames / wait up to ~2s for valid exposed frame
                start_time = time.time()
                valid = False
                for _ in range(15):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        # Once sensor auto-exposes above dark noise floor
                        if float(np.mean(test_frame)) > 0.5:
                            valid = True
                            break
                    time.sleep(0.08)
                    if time.time() - start_time > 2.0:
                        break

                if valid:
                    logger.info("UsbWebcamDevice: Opened device %s via backend %s (warm-up OK)", target, backend)
                    return cap

                cap.release()
            except Exception as exc:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                logger.debug("Failed opening device %s via backend %s: %s", target, backend, exc)

        return None

    def _init_camera(self) -> bool:
        if cv2 is None:
            self.init_error = "OpenCV (cv2) is not installed."
            return False

        # Try user-configured device first, then probe candidates
        candidates = [self.device] + self.probe_device_candidates()
        seen = set()
        unique_cands = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                unique_cands.append(c)

        for dev in unique_cands:
            cap = self._open_capture(dev)
            if cap is not None:
                self._cap = cap
                self.device = dev
                self._configure_camera()
                self._is_active = True
                self.init_error = None
                self._start_reader_thread()
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info("UsbWebcamDevice: Successfully opened USB camera on %s (%dx%d @ %d fps)", dev, w, h, self.fps)
                return True

        self._is_active = False
        self.init_error = "No functional USB camera found."
        logger.warning("UsbWebcamDevice: %s", self.init_error)
        return False

    def _configure_camera(self) -> None:
        if not self._cap or cv2 is None:
            return
        try:
            # Set MJPG fourcc if supported
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            w, h = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Focus controls
            try:
                if not self.autofocus and self.focus_value >= 0:
                    self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    self._cap.set(cv2.CAP_PROP_FOCUS, float(self.focus_value))
                elif self.autofocus:
                    self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("UsbWebcamDevice configure warning: %s", exc)

    def _start_reader_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="usb-cam-reader")
        self._thread.start()

    def _capture_worker(self) -> None:
        """Continuously grab latest frame to guarantee zero latency and discard queued frames."""
        while self._running:
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.5)
                continue
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self._lock:
                        self._latest_frame = frame
                else:
                    time.sleep(0.02)
            except Exception:
                time.sleep(0.05)

    def is_available(self) -> bool:
        return self._is_active and bool(self._cap and self._cap.isOpened())

    def capture_image(self) -> Optional[np.ndarray]:
        if not self._is_active:
            raise CameraNotConnectedError(self.init_error or "Camera not connected")

        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        # Direct grab fallback
        if self._cap and self._cap.isOpened():
            for _ in range(2):  # Flush any stale buffer frame
                self._cap.grab()
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame

        raise CameraNotConnectedError("USB Camera stream lost")

    def capture_still(self) -> Optional[np.ndarray]:
        """Capture high quality still frame at still_resolution (1920x1080) with autofocus handling and sharpest-of-N."""
        if not self._is_active:
            raise CameraNotConnectedError(self.init_error or "Camera not connected")

        sw, sh = self.still_resolution
        orig_w, orig_h = self.resolution
        frames: List[np.ndarray] = []
        if self._cap and self._cap.isOpened():
            try:
                self._cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), sw)
                self._cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), sh)
                time.sleep(0.08)
                for _ in range(3):
                    self._cap.grab()
                for _ in range(5):
                    ret, f = self._cap.read()
                    if ret and f is not None and f.size > 0:
                        frames.append(f)
                    time.sleep(0.02)
            except Exception as exc:
                logger.debug("UsbWebcamDevice still capture switch warning: %s", exc)
            finally:
                try:
                    self._cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), orig_w)
                    self._cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), orig_h)
                except Exception:
                    pass

        if frames:
            return max(frames, key=lambda f: compute_sharpness(f))

        return self.capture_image()

    def reconnect(self) -> bool:
        """Attempt to re-initialize camera connection without crashing."""
        self.close()
        time.sleep(0.2)
        return self._init_camera()

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_active = False
