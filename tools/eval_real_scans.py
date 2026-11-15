#!/usr/bin/env python3
"""
Real-Photo Evaluation & Tuning Harness for Brailce OBR
=====================================================
Evaluates OBREngine against real-world Braille photos (e.g. tools/enhanced_braille.jpg)
and reports detection statistics, pitch/skew estimates, lattice alignment, and decoded text.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except ImportError:
    cv2 = None

from brailce.vision.obr_engine import OBREngine


def evaluate_image(image_path: str, obr_engine: OBREngine) -> dict:
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return {}

    if cv2 is None:
        print("Error: OpenCV is required to run eval_real_scans.")
        return {}

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image: {image_path}")
        return {}

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    res = obr_engine.process_image(img)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    dots, dot_d = obr_engine.detect_dots(img)
    skew_deg, p_dot = obr_engine._estimate_skew_and_pitch(dots, dot_d) if dots else (0.0, 0.0)

    print("=" * 60)
    print(f"File: {os.path.basename(image_path)} ({w}x{h})")
    print(f"Processing time: {dt_ms:.1f} ms")
    print(f"Readable: {res.get('readable')} | Reason: {res.get('reason')}")
    print(f"Detected dots: {len(dots)} | Est dot diameter: {dot_d:.1f} px")
    print(f"Estimated skew: {skew_deg:.2f}° | Estimated pitch: {p_dot:.1f} px")
    print(f"Decoded cell events: {len(res.get('cells', []))}")
    print("-" * 60)
    print("Decoded Text:")
    print(res.get("text", "(None)"))
    print("=" * 60)

    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate real Braille scans.")
    parser.add_argument("images", nargs="*", default=["tools/enhanced_braille.jpg"], help="Path(s) to scan images")
    args = parser.parse_args()

    obr = OBREngine()
    for img_path in args.images:
        evaluate_image(img_path, obr)


if __name__ == "__main__":
    main()
