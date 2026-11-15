"""
Regression test for Crash #1: UsbWebcamDevice construction and missing imports.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from brailce.vision.camera import UsbWebcamDevice, CameraDevice


def test_usb_webcam_device_construction_no_name_error():
    """Verify UsbWebcamDevice can be instantiated without NameError on threading or Any."""
    with patch.object(UsbWebcamDevice, "_init_camera", return_value=True):
        cam = UsbWebcamDevice(device=0)
        assert isinstance(cam, CameraDevice)
        assert cam.resolution == (1280, 720)
        assert cam.autofocus is True
        assert hasattr(cam, "_lock")

