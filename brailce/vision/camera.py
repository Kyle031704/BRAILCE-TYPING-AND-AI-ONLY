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
import time
from abc import ABC, abstractmethod
from typing import Optional

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
