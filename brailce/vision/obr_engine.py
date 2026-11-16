"""
Optical Braille Recognition (OBR) Engine
========================================

Primary offline recognizer:
  Classical dot-level Grade 1 Braille pipeline (works on any embossed or printed
  Braille page without dataset/internet dependency):
    1. Image quality gates (paper contrast, Laplacian sharpness, dot resolution).
    2. Multi-scale Difference of Gaussians (DoG) for scale-adaptive dot detection
       (captures ink dots and merges embossed shadow/highlight pairs to dome centers).
    3. Grid orientation deskew + nearest-neighbor dot-pitch estimation.
    4. Data-driven text line clustering (prevents multi-line merge).
    5. Global 2x3 cell lattice fitting per text line (row y0 fit + column anchor fit).
    6. Exact Grade 1 Braille lookup with number sign (#) and capital sign (^) handling.

Optional secondary YOLO cell/dot detector:
  Available behind `use_yolo_secondary=True` if local weights exist (never downloads at runtime).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

logger = logging.getLogger(__name__)

UNKNOWN_CELL_CHAR = "?"


class OBREngine:
    """
    Universal engine for translating camera images of physical Braille into text.
    """

    # Exact Grade 1 Braille 6-bit Binary Dot Strings (bit 0 = dot 1 .. bit 5 = dot 6).
    # Zero fabricated/fuzzy guesses: any unrecognized pattern returns UNKNOWN_CELL_CHAR.
    BINARY_DICT: Dict[str, str] = {
        # Standard Alphabet (A-Z)
        "100000": "A", "110000": "B", "100100": "C", "100110": "D", "100010": "E",
        "110100": "F", "110110": "G", "110010": "H", "010100": "I", "010110": "J",
        "101000": "K", "111000": "L", "101100": "M", "101110": "N", "101010": "O",
        "111100": "P", "111110": "Q", "111010": "R", "011100": "S", "011110": "T",
        "101001": "U", "111001": "V", "010111": "W", "101101": "X", "101111": "Y",
        "101011": "Z",
        # Standard Grade 1 Punctuation
        "010000": ",", "011000": ";", "010010": ":", "010011": ".",
        "011010": "!", "011001": "?", "001000": "'", "001001": "-",
        "001011": "\"",
        # Indicators
        "000001": "^",  # Capital sign (Dot 6)
        "000101": "",   # Letter sign (Dots 4-6)
        "001111": "#",  # Number sign (Dots 3-4-5-6)
        "000000": " ",  # Blank cell
    }

    NAMED_DICT: Dict[str, str] = {
        "capital": "^", "cap": "^",
        "number_sign": "#", "number": "#", "num": "#", "hash": "#",
        "letter_sign": "",
        "period": ".", "comma": ",",
        "question_mark": "?", "question": "?",
        "exclamation_mark": "!", "exclamation": "!",
        "colon": ":", "semicolon": ";",
        "hyphen": "-", "dash": "-", "apostrophe": "'",
        "openquotation": "\"", "closequotation": "\"", "quote": "\"",
        "parenthesis": "(", "open_parenthesis": "(", "close_parenthesis": ")",
        "asterisk": "*", "ampersand": "&",
        "plus": "+", "minus": "-", "equals": "=", "slash": "/",
    }

    DIRECT_SYMBOLS: Dict[str, str] = {
        "#": "#", "^": "^", ".": ".", ",": ",", "?": "?", "!": "!",
        "-": "-", "'": "'", "\"": "\"", ":": ":", ";": ";",
        "+": "+", "*": "*", "/": "=", "(": "(", ")": ")", "=": "=",
    }

    NUM_MAP: Dict[str, str] = {
        "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
        "F": "6", "G": "7", "H": "8", "I": "9", "J": "0",
    }

    def __init__(
        self,
        model_path: str = "brailce/vision/yolov8_braille.pt",
        conf: float = 0.22,
        iou: float = 0.45,
        use_yolo_secondary: bool = False,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.use_yolo_secondary = use_yolo_secondary
        self._model = None
        self.detected_schema = "classical_dot_grid"

    def _load_model(self) -> bool:
        """Load local YOLO weights only if use_yolo_secondary=True and weights exist locally."""
        if not self.use_yolo_secondary:
            return False
        if self._model is not None:
            return True
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("Optional secondary YOLO weights missing at %s (no runtime download).", self.model_path)
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self.detected_schema = self._detect_model_schema()
            return True
        except Exception as exc:
            logger.debug("Optional YOLO model load skipped: %s", exc)
            return False

    def _detect_model_schema(self) -> str:
        if not self._model or not hasattr(self._model, "names") or not self._model.names:
            return "universal"
        str_names = [str(n).strip() for n in self._model.names.values()]
        if sum(1 for n in str_names if len(n) == 6 and set(n).issubset({"0", "1"})) >= min(5, len(str_names)):
            return "binary_dots"
        if any(n.lower() in {"number_sign", "capital", "period"} for n in str_names):
            return "named_classes"
        return "direct_characters"

    @classmethod
    def decode_cell_label(cls, raw_name: str) -> str:
        """Decode a single Braille cell label across 6-bit binary, named, and direct schemas."""
        clean_name = str(raw_name).strip()
        if not clean_name:
            return ""

        if clean_name in cls.DIRECT_SYMBOLS:
            return cls.DIRECT_SYMBOLS[clean_name]

        if clean_name in cls.BINARY_DICT:
            return cls.BINARY_DICT[clean_name]

        lowered = clean_name.lower().replace(" ", "_")
        if lowered in cls.NAMED_DICT:
            return cls.NAMED_DICT[lowered]

        for prefix in ("cell_", "braille_", "char_", "letter_"):
            if lowered.startswith(prefix):
                remainder = lowered[len(prefix):]
                if remainder in cls.NAMED_DICT:
                    return cls.NAMED_DICT[remainder]
                if remainder in cls.BINARY_DICT:
                    return cls.BINARY_DICT[remainder]
                if len(remainder) == 1 and remainder.isalpha():
                    return remainder.upper()
                if remainder.isdigit():
                    return remainder

        if len(clean_name) == 6 and set(clean_name).issubset({"0", "1"}):
            return ""

        if len(clean_name) == 1 and clean_name.isalpha():
            return clean_name.upper()
        if clean_name.isdigit():
            return clean_name

        return ""

    @classmethod
    def filter_valid_detections(cls, boxes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boxes_data or len(boxes_data) < 2:
            return []

        valid_shape = []
        for b in boxes_data:
            w, h = float(b["w"]), float(b["h"])
            aspect = min(w, h) / max(1e-6, max(w, h))
            if aspect >= 0.35:
                valid_shape.append(b)

        if len(valid_shape) < 2:
            return []

        sizes = [(float(b["w"]) + float(b["h"])) / 2.0 for b in valid_shape]
        median_s = float(np.median(sizes))
        if median_s <= 0:
            return []

        consistent = [
            b for b in valid_shape
            if abs(((float(b["w"]) + float(b["h"])) / 2.0) - median_s) / median_s <= 0.55
        ]
        return consistent if len(consistent) >= 2 else []

    @classmethod
    def decode_boxes(cls, boxes_data: List[Dict[str, Any]], space_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Decode cell boxes (from YOLO or synthetic boxes) into Grade 1 text."""
        valid_boxes = cls.filter_valid_detections(boxes_data)
        if not valid_boxes:
            return {"readable": False, "reason": "no_braille", "text": "", "cells": []}

        widths = [float(b["w"]) for b in valid_boxes]
        median_w = float(np.median(widths)) if widths else 30.0
        if space_threshold is None:
            space_threshold = max(20.0, median_w * 1.30)

        sorted_by_y = sorted(valid_boxes, key=lambda b: b["y"])
        lines: List[List[Dict[str, Any]]] = []
        for b in sorted_by_y:
            b_top = b["y"] - b["h"] / 2.0
            b_bot = b["y"] + b["h"] / 2.0
            best_line = None
            max_overlap = 0.0
            for line in lines:
                line_top = min(item["y"] - item["h"] / 2.0 for item in line)
                line_bot = max(item["y"] + item["h"] / 2.0 for item in line)
                line_h = max(item["h"] for item in line)
                overlap = max(0.0, min(b_bot, line_bot) - max(b_top, line_top))
                if overlap > (line_h * 0.35) and overlap > max_overlap:
                    max_overlap = overlap
                    best_line = line
            if best_line is not None:
                best_line.append(b)
            else:
                lines.append([b])

        lines.sort(key=lambda ln: float(np.median([item["y"] for item in ln])))

        cleaned_lines: List[List[Dict[str, Any]]] = []
        for line in lines:
            line.sort(key=lambda b: b["x"])
            dedup: List[Dict[str, Any]] = []
            for b in line:
                if not dedup:
                    dedup.append(b)
                elif abs(b["x"] - dedup[-1]["x"]) < (median_w * 0.65):
                    if b.get("conf", 0) > dedup[-1].get("conf", 0):
                        dedup[-1] = b
                else:
                    dedup.append(b)
            if dedup:
                cleaned_lines.append(dedup)

        line_strings: List[str] = []
        cell_events: List[Dict[str, Any]] = []
        word_idx = 0

        for line_idx, line in enumerate(cleaned_lines):
            line_str = ""
            is_number = False
            is_capital = False
            last_x = None

            for b in line:
                if last_x is not None and (b["x"] - last_x) > space_threshold:
                    line_str += " "
                    is_number = False
                    is_capital = False
                    word_idx += 1
                last_x = b["x"]

                raw = str(b.get("name", ""))
                char = cls.decode_cell_label(raw)
                if not char and raw != "000101" and raw != "letter_sign":
                    char = UNKNOWN_CELL_CHAR

                if char == "#":
                    is_number = True
                    cell_events.append({"char": "#", "kind": "indicator", "word_index": word_idx, "line_index": line_idx})
                    continue
                if char == "^":
                    is_capital = True
                    cell_events.append({"char": "^", "kind": "indicator", "word_index": word_idx, "line_index": line_idx})
                    continue
                if char == "":
                    continue

                if char == UNKNOWN_CELL_CHAR:
                    line_str += UNKNOWN_CELL_CHAR
                    cell_events.append({"char": UNKNOWN_CELL_CHAR, "kind": "unknown", "word_index": word_idx, "line_index": line_idx})
                elif is_number:
                    out_c = cls.NUM_MAP[char] if char in cls.NUM_MAP else (char if (char.isdigit() or char in {".", ","}) else char)
                    if not (out_c.isdigit() or out_c in {".", ","}):
                        is_number = False
                    line_str += out_c
                    cell_events.append({"char": out_c, "kind": "digit" if out_c.isdigit() else "char", "word_index": word_idx, "line_index": line_idx})
                else:
                    if is_capital and char.isalpha():
                        out_c = char.upper()
                        is_capital = False
                        cap_flag = True
                    else:
                        out_c = char
                        cap_flag = False
                    line_str += out_c
                    cell_events.append({"char": out_c, "kind": "char", "capitalized": cap_flag, "word_index": word_idx, "line_index": line_idx})

            clean_l = " ".join(line_str.split()).strip()
            if clean_l:
                line_strings.append(clean_l)
            word_idx += 1

        final_text = "\n".join(line_strings).strip()
        if not final_text:
            return {"readable": False, "reason": "no_braille", "text": "", "cells": []}

        return {"readable": True, "reason": "ok", "text": final_text, "cells": cell_events}

    # =========================================================================
    # Classical Scale-Adaptive Dot-Level Recognizer (Primary Offline Path)
    # =========================================================================

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Flatten uneven paper lighting and enhance embossed dot contrast via CLAHE."""
        if image is None or cv2 is None or image.size == 0:
            return image
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    @staticmethod
    def has_paper_background(image: np.ndarray) -> bool:
        """Verify frame is not an empty/black dead frame."""
        if image is None or image.size == 0:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if (cv2 is not None and len(image.shape) == 3) else image
        return float(np.percentile(gray, 90)) > 55.0 and float(np.std(gray)) > 3.5

    @staticmethod
    def check_image_quality(image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Check basic brightness and Laplacian sharpness.
        Returns (ok, reason, laplacian_var).
        """
        if image is None or image.size == 0 or cv2 is None:
            return False, "no_braille", 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if float(np.percentile(gray, 90)) <= 55.0 or float(np.std(gray)) <= 3.5:
            return False, "no_braille", 0.0

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 4.0:
            return False, "blurry", lap_var

        return True, "ok", lap_var

    @staticmethod
    def detect_dots(image: np.ndarray) -> Tuple[List[Dict[str, float]], float]:
        """
        Detect embossed or printed Braille dots across a wide scale range (4px to 20px)
        using multi-scale Difference of Gaussians (DoG). Robustly merges paired
        shadow-highlight crescents of embossed dots to the true geometric dome center,
        and retains solid ink dots.
        """
        if image is None or cv2 is None or image.size == 0:
            return [], 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]
        gray_f = gray.astype(np.float32)

        # 3-octave multi-scale Gaussian smoothing
        g0 = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
        g1 = cv2.GaussianBlur(gray_f, (0, 0), 2.0)
        g2 = cv2.GaussianBlur(gray_f, (0, 0), 4.0)
        g3 = cv2.GaussianBlur(gray_f, (0, 0), 8.0)

        # Dark response (center darker than surround: ink dots, shadow crescents)
        dog0_d = np.clip(g1 - g0, 0.0, 255.0)
        dog1_d = np.clip(g2 - g1, 0.0, 255.0)
        dog2_d = np.clip(g3 - g2, 0.0, 255.0)
        dark_resp = np.maximum(np.maximum(dog0_d, dog1_d), dog2_d)

        # Light response (center brighter than surround: embossed highlights)
        dog0_l = np.clip(g0 - g1, 0.0, 255.0)
        dog1_l = np.clip(g1 - g2, 0.0, 255.0)
        dog2_l = np.clip(g2 - g3, 0.0, 255.0)
        light_resp = np.maximum(np.maximum(dog0_l, dog1_l), dog2_l)

        # Robust thresholding: Braille dots typically occupy < 2% of the page
        p98_d = float(np.percentile(dark_resp, 98.0))
        p98_l = float(np.percentile(light_resp, 98.0))
        thresh_d = max(4.0, p98_d * 0.42)
        thresh_l = max(4.0, p98_l * 0.42)

        mask_d = (dark_resp >= thresh_d).astype(np.uint8)
        mask_l = (light_resp >= thresh_l).astype(np.uint8)

        # Dark blobs (shadows / ink dots)
        num_d, _, stats_d, cents_d = cv2.connectedComponentsWithStats(mask_d, connectivity=8)
        dark_blobs = []
        for lbl in range(1, num_d):
            area = float(stats_d[lbl, cv2.CC_STAT_AREA])
            bw = float(stats_d[lbl, cv2.CC_STAT_WIDTH])
            bh = float(stats_d[lbl, cv2.CC_STAT_HEIGHT])
            if area < 3.0 or bw < 2.0 or bh < 2.0 or bw > (w * 0.12) or bh > (h * 0.12):
                continue
            aspect = min(bw, bh) / max(1e-6, max(bw, bh))
            if aspect < 0.32:
                continue
            cx, cy = float(cents_d[lbl][0]), float(cents_d[lbl][1])
            diam = math.sqrt(4.0 * area / math.pi)
            score = float(dark_resp[int(cy), int(cx)]) if 0 <= int(cy) < h and 0 <= int(cx) < w else 1.0
            dark_blobs.append({"x": cx, "y": cy, "d": diam, "score": score, "area": area})

        # Light blobs (highlights)
        num_l, _, stats_l, cents_l = cv2.connectedComponentsWithStats(mask_l, connectivity=8)
        light_blobs = []
        for lbl in range(1, num_l):
            area = float(stats_l[lbl, cv2.CC_STAT_AREA])
            bw = float(stats_l[lbl, cv2.CC_STAT_WIDTH])
            bh = float(stats_l[lbl, cv2.CC_STAT_HEIGHT])
            if area < 3.0 or bw < 2.0 or bh < 2.0 or bw > (w * 0.12) or bh > (h * 0.12):
                continue
            aspect = min(bw, bh) / max(1e-6, max(bw, bh))
            if aspect < 0.32:
                continue
            cx, cy = float(cents_l[lbl][0]), float(cents_l[lbl][1])
            diam = math.sqrt(4.0 * area / math.pi)
            score = float(light_resp[int(cy), int(cx)]) if 0 <= int(cy) < h and 0 <= int(cx) < w else 1.0
            light_blobs.append({"x": cx, "y": cy, "d": diam, "score": score, "area": area})

        if not dark_blobs:
            return [], 0.0

        # Filter out fringe artifacts / anti-aliasing edge crescents (< 30% of median dot area)
        if len(dark_blobs) >= 4:
            med_a_d = float(np.median([b["area"] for b in dark_blobs]))
            dark_blobs = [b for b in dark_blobs if b["area"] >= (0.30 * med_a_d)]

        if len(light_blobs) >= 4:
            med_a_l = float(np.median([b["area"] for b in light_blobs]))
            light_blobs = [b for b in light_blobs if b["area"] >= (0.30 * med_a_l)]

        if not dark_blobs:
            return [], 0.0

        est_d = float(np.median([b["d"] for b in dark_blobs]))
        max_pair_dist = max(4.0, est_d * 1.15)

        # Match paired shadow and highlight for embossed Braille
        paired_dots: List[Dict[str, float]] = []
        used_l = set()
        for d_blob in dark_blobs:
            best_l = None
            best_dist = float("inf")
            for l_idx, l_blob in enumerate(light_blobs):
                if l_idx in used_l:
                    continue
                dist = math.hypot(d_blob["x"] - l_blob["x"], d_blob["y"] - l_blob["y"])
                if dist <= max_pair_dist and dist < best_dist:
                    best_dist = dist
                    best_l = l_idx

            if best_l is not None and best_dist >= (est_d * 0.25):
                l_match = light_blobs[best_l]
                diam_ratio = max(d_blob["d"], l_match["d"]) / max(1e-3, min(d_blob["d"], l_match["d"]))
                if diam_ratio <= 2.2:
                    used_l.add(best_l)
                    # Center of embossed dot is midpoint between shadow and highlight
                    cx = (d_blob["x"] + l_match["x"]) / 2.0
                    cy = (d_blob["y"] + l_match["y"]) / 2.0
                    dot_d = (d_blob["d"] + l_match["d"]) / 2.0
                    paired_dots.append({"x": cx, "y": cy, "d": dot_d, "score": d_blob["score"] + l_match["score"]})
                else:
                    paired_dots.append({"x": d_blob["x"], "y": d_blob["y"], "d": d_blob["d"], "score": d_blob["score"]})
            else:
                # Solid ink dot or top-lit spot
                paired_dots.append({"x": d_blob["x"], "y": d_blob["y"], "d": d_blob["d"], "score": d_blob["score"]})

        # Non-maximum suppression / deduplicate any dots closer than 0.5 * est_d
        paired_dots.sort(key=lambda d: d["score"], reverse=True)
        final_dots: List[Dict[str, float]] = []
        suppress_r = max(3.0, est_d * 0.52)
        for cand in paired_dots:
            if any(math.hypot(cand["x"] - ex["x"], cand["y"] - ex["y"]) <= suppress_r for ex in final_dots):
                continue
            final_dots.append(cand)

        final_d = float(np.median([d["d"] for d in final_dots])) if final_dots else 0.0
        return final_dots, final_d

    @staticmethod
    def _estimate_skew_and_pitch(dots: List[Dict[str, float]], dot_diam: float) -> Tuple[float, float]:
        """
        Estimate grid rotation angle (degrees, in [-25, 25]) and within-cell
        dot-to-dot pitch p_dot directly from nearest-neighbor vectors.
        """
        if len(dots) < 4:
            return 0.0, max(12.0, dot_diam * 2.5)

        pts = np.array([[d["x"], d["y"]] for d in dots], dtype=np.float64)
        n = len(pts)

        # 1. Direct Nearest-Neighbor p_dot measurement
        nn_dists: List[float] = []
        for i in range(n):
            diffs = pts - pts[i]
            d = np.hypot(diffs[:, 0], diffs[:, 1])
            d = d[d > 3.0]
            if len(d):
                nn_dists.append(float(np.min(d)))

        p_dot = float(np.median(nn_dists)) if nn_dists else max(12.0, dot_diam * 2.5)

        # 2. Skew estimation from orthogonal neighbor vectors around 1.0 * p_dot
        angles_deg: List[float] = []
        min_r = p_dot * 0.75
        max_r = p_dot * 1.35
        for i in range(n):
            diffs = pts - pts[i]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            mask = (dists >= min_r) & (dists <= max_r)
            if not np.any(mask):
                continue
            for j in np.where(mask)[0]:
                dx, dy = diffs[j]
                ang = math.degrees(math.atan2(dy, dx))
                # Fold angle modulo 90 degrees into [-45, 45]
                ang_folded = ((ang + 45.0) % 90.0) - 45.0
                if abs(ang_folded) <= 22.0:
                    angles_deg.append(ang_folded)

        skew_deg = float(np.median(angles_deg)) if len(angles_deg) >= 3 else 0.0
        return skew_deg, p_dot

    @classmethod
    def decode_dots_to_text(
        cls,
        dots: List[Dict[str, float]],
        dot_diam: float,
        image_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        Group detected dots into Braille lines, fit 2x3 cell grids, and translate
        exact 6-bit patterns using Grade 1 rules.
        """
        if len(dots) < 2:
            return {"readable": False, "reason": "no_braille", "text": "", "cells": [], "boxes": []}

        h, w = image_shape[:2]
        skew_deg, p_dot = cls._estimate_skew_and_pitch(dots, dot_diam)

        cx0, cy0 = w / 2.0, h / 2.0
        rad = math.radians(-skew_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        rot_dots: List[Dict[str, float]] = []
        for d in dots:
            dx = d["x"] - cx0
            dy = d["y"] - cy0
            rx = dx * cos_a - dy * sin_a + cx0
            ry = dx * sin_a + dy * cos_a + cy0
            rot_dots.append({**d, "rx": rx, "ry": ry})

        # 1. Cluster into text lines by Y separation:
        # A single Braille line has height <= 2.0 * p_dot.
        # Line grouping: dots with ry within 2.3 * p_dot of the line's top belong to the same line.
        rot_dots.sort(key=lambda d: d["ry"])
        lines: List[List[Dict[str, float]]] = []
        for d in rot_dots:
            if not lines:
                lines.append([d])
            else:
                line_min_y = min(item["ry"] for item in lines[-1])
                if (d["ry"] - line_min_y) <= (p_dot * 2.30):
                    lines[-1].append(d)
                else:
                    lines.append([d])

        cell_pitch_default = p_dot * 2.4
        line_strings: List[str] = []
        cell_events: List[Dict[str, Any]] = []
        debug_boxes: List[Dict[str, Any]] = []
        word_idx = 0

        for line_idx, line_dots in enumerate(lines):
            if len(line_dots) < 1:
                continue

            # 2. Fit best top-row baseline y0 such that y0, y0+p_dot, y0+2*p_dot maximize dot fit
            ys = np.array([d["ry"] for d in line_dots], dtype=np.float64)
            best_y0 = float(np.min(ys))
            best_err = float("inf")
            for y_anchor in ys:
                for offset in (0, 1, 2):
                    cand_y0 = y_anchor - offset * p_dot
                    row_dists = np.min(
                        np.abs(ys[:, None] - (cand_y0 + np.array([0.0, p_dot, 2.0 * p_dot])[None, :])),
                        axis=1,
                    )
                    err = float(np.sum(row_dists))
                    if err < best_err:
                        best_err = err
                        best_y0 = float(cand_y0)

            # 3. Cluster horizontal dot positions into vertical sub-columns
            line_dots.sort(key=lambda d: d["rx"])
            sub_cols: List[List[Dict[str, float]]] = []
            for d in line_dots:
                if not sub_cols:
                    sub_cols.append([d])
                else:
                    mean_x = float(np.mean([item["rx"] for item in sub_cols[-1]]))
                    if abs(d["rx"] - mean_x) <= p_dot * 0.45:
                        sub_cols[-1].append(d)
                    else:
                        sub_cols.append([d])

            col_xs = [float(np.mean([d["rx"] for d in sc])) for sc in sub_cols]

            # Find paired columns (dx in [0.82 * p_dot, 1.18 * p_dot]) to anchor cell grid.
            # Intra-cell column gap is strictly 1.0 * p_dot, whereas inter-cell gap is >= 1.4 * p_dot.
            pairs: List[Tuple[float, float]] = []
            i = 0
            while i < len(col_xs) - 1:
                dx = col_xs[i + 1] - col_xs[i]
                if 0.82 * p_dot <= dx <= 1.18 * p_dot:
                    pairs.append((col_xs[i], col_xs[i + 1]))
                    i += 2
                else:
                    i += 1

            p_cell = p_dot * 2.4
            if len(pairs) >= 2:
                diffs = [pairs[j + 1][0] - pairs[j][0] for j in range(len(pairs) - 1)]
                cands = []
                for df in diffs:
                    for k in range(1, 10):
                        pitch = df / k
                        if 2.1 * p_dot <= pitch <= 2.7 * p_dot:
                            cands.append(pitch)
                if cands:
                    p_cell = float(np.median(cands))

            x0 = pairs[0][0] if pairs else col_xs[0]

            # Assign each dot to (cell_k, col in {0, 1}, row in {0, 1, 2})
            # Decision boundary between cell k and k+1 is midway between col 1 and next col 0:
            # col 1 is at 1.0 * p_dot, next col 0 is at p_cell (~2.4 * p_dot). Midpoint = (p_cell + p_dot) / 2
            half_gap = 0.5 * (p_cell - p_dot)
            cells_bits: Dict[int, List[int]] = {}
            for d in line_dots:
                dx = d["rx"] - x0
                k = int(math.floor((dx + half_gap) / p_cell))
                x_cell = x0 + k * p_cell
                rem = d["rx"] - x_cell
                col = 0 if rem < (0.5 * p_dot) else 1
                row = int(round((d["ry"] - best_y0) / p_dot))
                row = max(0, min(2, row))
                bits = cells_bits.setdefault(k, [0, 0, 0, 0, 0, 0])
                bits[col * 3 + row] = 1

            line_str = ""
            is_number = False
            is_capital = False
            prev_k: Optional[int] = None

            for k in sorted(cells_bits.keys()):
                if prev_k is not None and (k - prev_k) > 1:
                    line_str += " "
                    is_number = False
                    is_capital = False
                    word_idx += 1
                prev_k = k

                bits = cells_bits[k]
                bit_str = "".join(str(b) for b in bits)
                char = cls.BINARY_DICT.get(bit_str, UNKNOWN_CELL_CHAR)
                active_dots = [idx + 1 for idx, b in enumerate(bits) if b == 1]
                dots_str = "Dots " + "-".join(str(d) for d in active_dots) if active_dots else "No dots"

                # Compute unrotated visual bounding box center
                rx_c = x0 + k * p_cell + 0.5 * p_dot
                ry_c = best_y0 + p_dot
                dx_c = rx_c - cx0
                dy_c = ry_c - cy0
                rad_pos = math.radians(skew_deg)
                ux = dx_c * math.cos(rad_pos) - dy_c * math.sin(rad_pos) + cx0
                uy = dx_c * math.sin(rad_pos) + dy_c * math.cos(rad_pos) + cy0

                if char == "#":
                    is_number = True
                    debug_boxes.append({"x": ux, "y": uy, "w": p_dot * 1.6, "h": p_dot * 2.6, "label": "#"})
                    cell_events.append({"char": "#", "kind": "indicator", "dots": active_dots, "dots_str": dots_str, "word_index": word_idx, "line_index": line_idx})
                    continue
                if char == "^":
                    is_capital = True
                    debug_boxes.append({"x": ux, "y": uy, "w": p_dot * 1.6, "h": p_dot * 2.6, "label": "^"})
                    cell_events.append({"char": "^", "kind": "indicator", "dots": active_dots, "dots_str": dots_str, "word_index": word_idx, "line_index": line_idx})
                    continue
                if char == "" or char == " ":
                    continue

                if char == UNKNOWN_CELL_CHAR:
                    out_c = UNKNOWN_CELL_CHAR
                    kind = "unknown"
                    cap_flag = False
                elif is_number:
                    out_c = cls.NUM_MAP.get(char, char)
                    if not (out_c.isdigit() or out_c in {".", ","}):
                        is_number = False
                        kind = "char"
                    else:
                        kind = "digit" if out_c.isdigit() else "char"
                    cap_flag = False
                else:
                    if is_capital and char.isalpha():
                        out_c = char.upper()
                        is_capital = False
                        cap_flag = True
                    else:
                        out_c = char
                        cap_flag = False
                    kind = "char"

                line_str += out_c
                debug_boxes.append({"x": ux, "y": uy, "w": p_dot * 1.6, "h": p_dot * 2.6, "label": out_c})
                cell_events.append({
                    "char": out_c,
                    "kind": kind,
                    "capitalized": cap_flag,
                    "dots": active_dots,
                    "dots_str": dots_str,
                    "word_index": word_idx,
                    "line_index": line_idx,
                })



            clean_l = " ".join(line_str.split()).strip()
            if clean_l:
                line_strings.append(clean_l)
            word_idx += 1

        final_text = "\n".join(line_strings).strip()
        if not final_text:
            return {"readable": False, "reason": "no_braille", "text": "", "cells": [], "boxes": debug_boxes}

        return {
            "readable": True,
            "reason": "ok",
            "text": final_text,
            "cells": cell_events,
            "boxes": debug_boxes,
            "skew_deg": skew_deg,
            "dot_pitch": p_dot,
            "dot_diameter": dot_diam,
        }

    def detect_braille_presence(self, image: np.ndarray) -> bool:
        """
        Robust presence check (~5-10 Hz on Pi 5) verifying:
        1. Basic lighting & paper background quality.
        2. Minimum dot count (>= 4 dots).
        3. Diameter consistency (variance/std of dot radius <= 0.65 of median).
        4. Regular nearest-neighbor pitch ratio (1.6 <= pitch/diam <= 5.5).
        5. Grid regularity: at least 40% of dots share a neighbor within consistent pitch.
        """
        if image is None or cv2 is None or image.size == 0:
            return False
        try:
            ok, _reason, _lap = self.check_image_quality(image)
            if not ok:
                return False

            h, w = image.shape[:2]
            scale = 1.0
            if max(w, h) > 640:
                scale = 640.0 / float(max(w, h))
                small = cv2.resize(image, (int(w * scale), int(h * scale)))
            else:
                small = image

            dots, dot_diam = self.detect_dots(small)
            if len(dots) < 4:
                return False

            # Reject static / noise frames where thousands of spurious dots are found
            sh, sw = small.shape[:2]
            max_expected_dots = int((sw * sh) / 350.0)
            if len(dots) > max(120, max_expected_dots):
                return False

            # Check dot diameter consistency
            diams = np.array([d["d"] for d in dots], dtype=np.float64)
            med_d = float(np.median(diams))
            if med_d < 2.5:
                return False
            std_d = float(np.std(diams))
            if (std_d / max(1e-3, med_d)) > 0.65:
                return False

            # Check nearest-neighbor grid pitch regularity
            pts = np.array([[d["x"], d["y"]] for d in dots], dtype=np.float64)
            n = len(pts)
            nn_dists = []
            for i in range(n):
                diffs = pts - pts[i]
                dists = np.hypot(diffs[:, 0], diffs[:, 1])
                valid = dists[dists > 3.0]
                if len(valid):
                    nn_dists.append(float(np.min(valid)))

            if len(nn_dists) < 4:
                return False

            med_pitch = float(np.median(nn_dists))
            pitch_ratio = med_pitch / max(1.0, med_d)
            if not (1.2 <= pitch_ratio <= 6.0):
                return False

            # Grid consistency: at least 40% of dots within +/- 40% of median pitch
            regular_count = sum(1 for d in nn_dists if 0.60 * med_pitch <= d <= 1.40 * med_pitch)
            if (regular_count / len(nn_dists)) < 0.40:
                return False

            return True
        except Exception as exc:
            logger.debug("detect_braille_presence error: %s", exc)
            return False

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Decode physical Braille image using quality gates + classical dot-grid recognizer
        (with optional secondary YOLO fallback only if explicitly enabled & weights exist).
        """
        if image is None:
            return {"readable": False, "reason": "no_braille", "text": "", "cells": []}

        quality_ok, reason, lap_var = self.check_image_quality(image)
        if not quality_ok:
            return {"readable": False, "reason": reason, "text": "", "cells": [], "lap_var": lap_var}

        enhanced = self.enhance_contrast(image)
        dots, dot_diam = self.detect_dots(image)
        if len(dots) < 4:
            dots_enh, diam_enh = self.detect_dots(enhanced)
            if len(dots_enh) > len(dots):
                dots, dot_diam = dots_enh, diam_enh
        h, w = image.shape[:2]

        if len(dots) >= 2:
            classical_res = self.decode_dots_to_text(dots, dot_diam, image.shape)
            classical_res["lap_var"] = lap_var
            classical_res["dots"] = dots

            # Quality check based on fitted lattice rather than noisy background blobs
            p_dot = classical_res.get("dot_pitch", dot_diam * 2.5)
            if p_dot < 6.0 and max(w, h) >= 640:
                return {"readable": False, "reason": "too_small", "text": "", "cells": [], "dots": dots, "lap_var": lap_var}
            if p_dot > min(w, h) * 0.40:
                return {"readable": False, "reason": "too_large", "text": "", "cells": [], "dots": dots, "lap_var": lap_var}

            if classical_res.get("readable"):
                cells = classical_res.get("cells", [])
                lines_count = len(set(c.get("line_index", 0) for c in cells)) if cells else 0
                unknown_count = sum(1 for c in cells if c.get("char") == UNKNOWN_CELL_CHAR)
                unknown_ratio = (unknown_count / len(cells)) if cells else 0.0

                # Reject random noise images that don't form coherent Braille documents
                if len(cells) >= 4 and unknown_ratio > 0.20:
                    return {"readable": False, "reason": "no_braille", "text": "", "cells": [], "dots": dots, "lap_var": lap_var}
                if lines_count >= 3 and (len(cells) / max(1, lines_count)) < 2.5:
                    return {"readable": False, "reason": "no_braille", "text": "", "cells": [], "dots": dots, "lap_var": lap_var}

                return classical_res

        # Optional secondary YOLO detector (disabled by default; local weights only)
        if self.use_yolo_secondary and self._load_model():
            try:
                enhanced = self.enhance_contrast(image)
                results = self._model(enhanced, conf=self.conf, iou=self.iou, verbose=False)
                boxes_data = []
                for result in results or []:
                    if not hasattr(result, "boxes") or result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        boxes_data.append({
                            "x": (x1 + x2) / 2.0,
                            "y": (y1 + y2) / 2.0,
                            "w": abs(x2 - x1),
                            "h": abs(y2 - y1),
                            "name": str(result.names[class_id]),
                            "conf": float(box.conf[0]),
                        })
                yolo_res = self.decode_boxes(boxes_data)
                if yolo_res.get("readable"):
                    yolo_res["lap_var"] = lap_var
                    return yolo_res
            except Exception as exc:
                logger.debug("Optional YOLO fallback error: %s", exc)

        return {"readable": False, "reason": "no_braille", "text": "", "cells": [], "lap_var": lap_var, "dots": dots}
