#!/usr/bin/env python3
"""
tools/synth_braille.py - Ground-truth synthetic Braille page generator.
======================================================================
Generates synthetic Grade 1 Braille images with realistic physical parameters:
- Dot diameter: 5 to 16 px
- Standard geometry: dot pitch ~2.5 * d, cell pitch ~2.4 * p_dot, line pitch ~4.0 * p_dot
- Styles: "ink" (printed dark dots) and "embossed" (3D raised bumps with shadow/highlight)
- Degradations: rotation (+/-10 deg), perspective (+/-5 deg), blur, noise, uneven lighting
- Character support: letters A-Z, capital sign (^), digits (number sign # + A-J), punctuation.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Exact Grade 1 Braille cell patterns (bit 0..5 = dots 1..6)
CHAR_TO_BITS: Dict[str, str] = {
    # Alphabet
    "A": "100000", "B": "110000", "C": "100100", "D": "100110", "E": "100010",
    "F": "110100", "G": "110110", "H": "110010", "I": "010100", "J": "010110",
    "K": "101000", "L": "111000", "M": "101100", "N": "101110", "O": "101010",
    "P": "111100", "Q": "111110", "R": "111010", "S": "011100", "T": "011110",
    "U": "101001", "V": "111001", "W": "010111", "X": "101101", "Y": "101111",
    "Z": "101011",
    # Punctuation
    ",": "010000", ";": "011000", ":": "010010", ".": "010011",
    "!": "011010", "?": "011001", "'": "001000", "-": "001001", "\"": "001011",
    # Indicators
    "^": "000001",  # Capital sign (Dot 6)
    "#": "001111",  # Number sign (Dots 3-4-5-6)
    " ": "000000",  # Blank space
}

DIGIT_TO_LETTER = {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
    "6": "F", "7": "G", "8": "H", "9": "I", "0": "J",
}


def text_to_braille_cells(text: str) -> List[List[str]]:
    """
    Convert text (lines separated by newline) into a grid of 6-bit cell strings.
    Inserts capital signs for uppercase letters and number signs for digit sequences.
    """
    lines_cells: List[List[str]] = []
    for raw_line in text.split("\n"):
        cells: List[str] = []
        in_number = False
        for ch in raw_line:
            if ch.isdigit():
                if not in_number:
                    cells.append(CHAR_TO_BITS["#"])
                    in_number = True
                cells.append(CHAR_TO_BITS[DIGIT_TO_LETTER[ch]])
            else:
                in_number = False
                if ch == " ":
                    cells.append(CHAR_TO_BITS[" "])
                elif ch.isupper():
                    cells.append(CHAR_TO_BITS["^"])
                    cells.append(CHAR_TO_BITS.get(ch, "000000"))
                elif ch.upper() in CHAR_TO_BITS:
                    cells.append(CHAR_TO_BITS[ch.upper()])
                elif ch in CHAR_TO_BITS:
                    cells.append(CHAR_TO_BITS[ch])
        lines_cells.append(cells)
    return lines_cells


def generate_synthetic_braille_image(
    text: str,
    dot_diameter: float = 9.0,
    style: str = "embossed",  # "embossed" or "ink"
    rotation_deg: float = 0.0,
    perspective_deg: float = 0.0,
    blur_sigma: float = 0.6,
    noise_sigma: float = 5.0,
    uneven_lighting: bool = True,
    margin_x: float = 60.0,
    margin_y: float = 60.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, str, List[Dict[str, float]]]:
    """
    Render synthetic Braille document image.
    Returns:
      (image_bgr, expected_text, ground_truth_dots)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Standard Braille proportions relative to dot diameter d:
    # dot_pitch (center-to-center between dots 1 and 2): ~2.5 * d
    # cell_pitch (center-to-center between cells): ~2.4 * dot_pitch
    # line_pitch (baseline-to-baseline): ~4.0 * dot_pitch
    d = float(dot_diameter)
    p_dot = d * 2.5
    p_cell = p_dot * 2.4
    p_line = p_dot * 4.0

    lines_cells = text_to_braille_cells(text)
    max_cols = max(len(ln) for ln in lines_cells) if lines_cells else 1
    num_lines = max(1, len(lines_cells))

    # Base canvas size
    width = int(margin_x * 2 + max_cols * p_cell + p_dot * 2)
    height = int(margin_y * 2 + num_lines * p_line + p_dot * 3)
    width = max(320, width)
    height = max(240, height)

    # Base paper background (light cream / off-white paper: ~235-245)
    canvas = np.full((height, width), 240.0, dtype=np.float32)

    # Add subtle paper texture
    paper_noise = np.random.normal(0, 2.5, (height, width)).astype(np.float32)
    canvas = np.clip(canvas + paper_noise, 0, 255)

    gt_dots: List[Dict[str, float]] = []

    # Dot radius
    radius = d / 2.0

    # Draw each cell
    for line_idx, line in enumerate(lines_cells):
        y0 = margin_y + line_idx * p_line
        for cell_idx, bit_str in enumerate(line):
            if bit_str == "000000":
                continue
            x0 = margin_x + cell_idx * p_cell

            # 6 dots:
            # col 0: dots 1, 2, 3 -> (x0, y0), (x0, y0 + p_dot), (x0, y0 + 2*p_dot)
            # col 1: dots 4, 5, 6 -> (x0 + p_dot, y0), (x0 + p_dot, y0 + p_dot), (x0 + p_dot, y0 + 2*p_dot)
            coords = [
                (x0, y0),                     # Dot 1
                (x0, y0 + p_dot),             # Dot 2
                (x0, y0 + 2.0 * p_dot),       # Dot 3
                (x0 + p_dot, y0),             # Dot 4
                (x0 + p_dot, y0 + p_dot),     # Dot 5
                (x0 + p_dot, y0 + 2.0 * p_dot)# Dot 6
            ]

            for dot_num, (cx, cy) in enumerate(coords):
                if bit_str[dot_num] == "1":
                    gt_dots.append({"x": cx, "y": cy, "d": d, "line": line_idx, "cell": cell_idx})

    # Render dots onto canvas
    # Create coordinate grid for vectorization
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    if style == "ink":
        # Solid dark ink dots (e.g. intensity ~45-60)
        for dot in gt_dots:
            cx, cy = dot["x"], dot["y"]
            r2 = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            # Anti-aliased circle profile
            mask = np.clip(1.0 - (np.sqrt(r2) - (radius - 0.7)), 0.0, 1.0)
            canvas = canvas * (1.0 - mask * 0.82) + (50.0 * mask * 0.82)

    else:  # "embossed" style: raised spherical dome lit by raking light from top-left (angle -135 deg)
        # Highlight is top-left, shadow is bottom-right
        light_dx = -0.55
        light_dy = -0.55
        for dot in gt_dots:
            cx, cy = dot["x"], dot["y"]
            dx = x_coords - cx
            dy = y_coords - cy
            dist2 = dx ** 2 + dy ** 2
            in_dome = dist2 <= (radius * 1.25) ** 2

            if not np.any(in_dome):
                continue

            # Spherical dome surface normal: (nx, ny, nz)
            # where dome height z = sqrt(max(0, R^2 - r^2))
            r = np.sqrt(dist2)
            dome_mask = np.clip(1.0 - (r - radius), 0.0, 1.0)

            # Dot shading: directional derivative along lighting vector
            shading = (dx * (-light_dx) + dy * (-light_dy)) / max(1.0, radius)
            # Highlight on top-left (shading < 0), shadow on bottom-right (shading > 0)
            # Add highlight (brighter than paper) and shadow (darker than paper)
            effect = -shading * 42.0 * dome_mask
            canvas[in_dome] = np.clip(canvas[in_dome] + effect[in_dome], 0.0, 255.0)

    # Add uneven illumination gradient across page if requested
    if uneven_lighting:
        grad_x = np.linspace(-15.0, 15.0, width, dtype=np.float32)
        grad_y = np.linspace(-20.0, 20.0, height, dtype=np.float32)[:, None]
        canvas = np.clip(canvas + grad_x + grad_y, 0.0, 255.0)

    # Add Gaussian blur (lens defocus / optical resolution)
    if blur_sigma > 0:
        ksize = int(math.ceil(blur_sigma * 4)) | 1
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), blur_sigma)

    # Add camera sensor noise
    if noise_sigma > 0:
        sensor_noise = np.random.normal(0, noise_sigma, (height, width)).astype(np.float32)
        canvas = np.clip(canvas + sensor_noise, 0.0, 255.0)

    # Convert to 3-channel BGR
    img_bgr = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Apply rotation if specified
    if abs(rotation_deg) > 0.01:
        M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation_deg, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Apply perspective skew if specified
    if abs(perspective_deg) > 0.01:
        rad = math.radians(perspective_deg)
        dx = int(width * 0.5 * math.tan(abs(rad)))
        src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        if perspective_deg > 0:
            dst_pts = np.float32([[dx, 0], [width - dx, 0], [width, height], [0, height]])
        else:
            dst_pts = np.float32([[0, 0], [width, 0], [width - dx, height], [dx, height]])
        M_p = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img_bgr = cv2.warpPerspective(img_bgr, M_p, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Compute expected decoded text
    expected_text = text.strip()
    return img_bgr, expected_text, gt_dots


if __name__ == "__main__":
    sample_text = "HELLO WORLD\n1234\nBRAILLE IS FUN"
    img, exp, dots = generate_synthetic_braille_image(sample_text, dot_diameter=9.0, style="embossed", rotation_deg=0.0)
    cv2.imwrite("tools/sample_synth_embossed.png", img)
    print(f"Generated synthetic Braille image: {img.shape}, {len(dots)} dots. Expected text: {exp!r}")

