"""
Brailce Tests — Classical Dot-Level Grade 1 Braille Recognizer
================================================================

Tests OBREngine classical pipeline against synthetic shaded embossed Braille
pages (lighting gradients, blur, rotation, noise, low contrast, unknown cells)
and checks existing repository image fixtures safely.
"""

import os
from typing import List

import cv2
import numpy as np
import pytest

from brailce.vision.obr_engine import OBREngine, UNKNOWN_CELL_CHAR


def render_shaded_braille(
    text: str,
    pitch: int = 20,
    dot_r: float = 4.5,
    blur_sigma: float = 0.0,
    angle_deg: float = 0.0,
    noise_std: float = 0.0,
    contrast_scale: float = 1.0,
    width: int = 640,
    height: int = 320,
) -> np.ndarray:
    """Render an embossed-style Grade 1 Braille page with shading & uneven lighting."""
    rev = {v: k for k, v in OBREngine.BINARY_DICT.items() if len(v) == 1 and v.isalpha()}
    rev["#"] = "001111"
    rev["^"] = "000001"
    dmap = {v: k for k, v in OBREngine.NUM_MAP.items()}

    img = np.full((height, width, 3), 215, dtype=np.float32)
    gx = np.linspace(-20.0, 20.0, width)[None, :, None]
    gy = np.linspace(-15.0, 15.0, height)[:, None, None]
    img = np.clip(img + gx + gy, 0, 255).astype(np.uint8)

    tokens: List[str] = []
    in_num = False
    for ch in text:
        if ch == " ":
            tokens.append("000000")
            in_num = False
        elif ch.isdigit():
            if not in_num:
                tokens.append("001111")
                in_num = True
            tokens.append(rev[dmap[ch]])
        elif ch.isupper():
            in_num = False
            tokens.append("000001")
            tokens.append(rev[ch])
        else:
            in_num = False
            tokens.append(rev[ch.upper()])

    x0, y0 = 55.0, 105.0
    hi_delta = int(round(27 * contrast_scale))
    lo_delta = int(round(75 * contrast_scale))
    hi_col = (min(255, 215 + hi_delta),) * 3
    lo_col = (max(0, 215 - lo_delta),) * 3

    for i, bits in enumerate(tokens):
        cx = x0 + i * (pitch * 2.5)
        for bi, b in enumerate(bits):
            if b == "1":
                col = bi // 3
                row = bi % 3
                dx = int(round(cx + col * pitch))
                dy = int(round(y0 + row * pitch))
                r_px = max(2, int(round(dot_r)))
                cv2.circle(img, (dx - 1, dy - 1), r_px, hi_col, -1, cv2.LINE_AA)
                cv2.circle(img, (dx + 2, dy + 2), r_px, lo_col, -1, cv2.LINE_AA)

    if angle_deg:
        M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    if blur_sigma > 0.0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, noise_std, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


class TestClassicalOBRWordsAndResilience:
    """Verify exact Grade 1 decoding on shaded synthetic Braille pages."""

    @pytest.mark.parametrize("phrase,expected", [
        ("need", "NEED"),
        ("braille", "BRAILLE"),
        ("hello", "HELLO"),
        ("1234", "1234"),
        ("Cat", "CAT"),
    ])
    def test_synthetic_shaded_grade1_words(self, phrase: str, expected: str) -> None:
        engine = OBREngine()
        img = render_shaded_braille(phrase)
        res = engine.process_image(img)
        assert res["readable"] is True, f"Failed {phrase}: {res}"
        assert res["text"] == expected

    @pytest.mark.parametrize("angle_deg", [-5.0, 4.0, 6.0])
    def test_rotation_resilience(self, angle_deg: float) -> None:
        engine = OBREngine()
        img = render_shaded_braille("need", angle_deg=angle_deg)
        res = engine.process_image(img)
        assert res["readable"] is True
        assert res["text"] == "NEED"

    def test_noise_and_low_contrast_resilience(self) -> None:
        engine = OBREngine()
        img = render_shaded_braille("hello", noise_std=5.0, contrast_scale=0.55)
        res = engine.process_image(img)
        assert res["readable"] is True
        assert res["text"] == "HELLO"


class TestClassicalOBRQualityGatesAndUnknowns:
    """Verify failure diagnostics and zero fabricated guesses on invalid cells."""

    def test_flat_dark_image_returns_no_braille(self) -> None:
        engine = OBREngine()
        dark = np.zeros((320, 640, 3), dtype=np.uint8)
        res = engine.process_image(dark)
        assert res["readable"] is False
        assert res["reason"] == "no_braille"

    def test_blurry_image_returns_blurry(self) -> None:
        engine = OBREngine()
        blurry = render_shaded_braille("need", blur_sigma=3.2)
        res = engine.process_image(blurry)
        assert res["readable"] is False
        assert res["reason"] == "blurry"

    def test_tiny_dots_returns_too_small_or_no_braille(self) -> None:
        engine = OBREngine()
        # Sub-threshold ~3px dots across a 640x480 frame
        img = np.full((480, 640, 3), 215, dtype=np.uint8)
        for i in range(6):
            cv2.circle(img, (100 + i * 15, 200), 1, (110, 110, 110), -1)
        res = engine.process_image(img)
        assert res["readable"] is False
        assert res["reason"] in {"too_small", "no_braille"}

    def test_unknown_six_dot_cell_returns_question_mark(self) -> None:
        # "000110" (dots 4-5) has no Grade 1 letter mapping -> must decode as '?'
        boxes = [
            {"x": 100.0, "y": 100.0, "w": 30.0, "h": 45.0, "name": "100000", "conf": 0.9},
            {"x": 135.0, "y": 100.0, "w": 30.0, "h": 45.0, "name": "000110", "conf": 0.9},
        ]
        res = OBREngine.decode_boxes(boxes)
        assert res["readable"] is True
        assert res["text"] == f"A{UNKNOWN_CELL_CHAR}"

    def test_repository_image_fixtures_safe_handling(self) -> None:
        engine = OBREngine()
        if os.path.exists("tools/test_real_cam.jpg"):
            real_cam = cv2.imread("tools/test_real_cam.jpg")
            res = engine.process_image(real_cam)
            # Monitor terminal photo contains zero Braille — must NOT hallucinate Braille text
            assert res["readable"] is False
            assert res["text"] == ""
        if os.path.exists("tools/enhanced_braille.jpg"):
            enhanced = cv2.imread("tools/enhanced_braille.jpg")
            dots, dd = engine.detect_dots(enhanced)
            assert len(dots) > 100
            assert dd > 3.0
