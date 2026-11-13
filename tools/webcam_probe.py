#!/usr/bin/env python3
"""
tools/webcam_probe.py - Probe USB webcams on Raspberry Pi / PC.
==============================================================
Discovers V4L2 video devices, checks supported resolutions/formats,
reports autofocus/manual focus controls, and measures live sharpness.
"""

import glob
import os
import sys
import time

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python (cv2) is not installed.")
    sys.exit(1)

import numpy as np


def compute_sharpness(frame: np.ndarray) -> float:
    """Return variance of Laplacian sharpness metric."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def list_device_nodes() -> list[str]:
    """Find /dev/video* nodes on Linux or numeric indices on Windows."""
    if os.name != "nt":
        by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
        if by_id:
            print("\nFound /dev/v4l/by-id devices:")
            for p in by_id:
                real = os.path.realpath(p)
                print(f"  {p} -> {real}")
        nodes = sorted(glob.glob("/dev/video*"))
        return nodes if nodes else ["0", "1", "2"]
    return ["0", "1", "2"]


def probe_device(dev_identifier: str | int):
    target = int(dev_identifier) if str(dev_identifier).isdigit() else str(dev_identifier)
    api = cv2.CAP_V4L2 if isinstance(target, str) or os.name != "nt" else cv2.CAP_ANY

    cap = cv2.VideoCapture(target, api)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        cap.release()
        return None

    # Check properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    sharpness = compute_sharpness(frame)

    info = {
        "device": dev_identifier,
        "width": width,
        "height": height,
        "fps": fps,
        "fourcc": fourcc,
        "autofocus": autofocus,
        "focus": focus,
        "sample_sharpness": sharpness,
    }
    cap.release()
    return info


def main():
    print("=" * 60)
    print(" Brailce USB Webcam Probe")
    print("=" * 60)

    devices = list_device_nodes()
    found = []

    for dev in devices:
        print(f"Probing {dev}...", end="", flush=True)
        info = probe_device(dev)
        if info:
            print(f" [OK] - {info['width']}x{info['height']} @ {info['fps']:.1f} fps (sharpness: {info['sample_sharpness']:.1f})")
            found.append(info)
        else:
            print(" [Skip: not a video capture stream or inactive]")

    print("\nSummary of working capture webcams:")
    if not found:
        print("  No working webcam capture streams detected.")
        print("  Check USB cable connection or try: lsusb / v4l2-ctl --list-devices")
        return

    for idx, c in enumerate(found):
        print(f"\n  Webcam #{idx+1}: {c['device']}")
        print(f"    Resolution: {c['width']}x{c['height']}")
        print(f"    FPS:        {c['fps']}")
        print(f"    FourCC:     {c['fourcc']}")
        print(f"    Autofocus:  {c['autofocus']} (focus val: {c['focus']})")
        print(f"    Sharpness:  {c['sample_sharpness']:.2f}")

    print("\nRecommendation for brailce/config/config.json (or .env):")
    primary = found[0]['device']
    print(f'  Set vision.usb_device to "{primary}" (or export BRAILCE_CAMERA_DEVICE="{primary}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
