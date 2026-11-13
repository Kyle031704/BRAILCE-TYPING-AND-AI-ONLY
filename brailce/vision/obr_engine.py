"""
Optical Braille Recognition (OBR) Engine
========================================

Primary offline recognizer:
  Classical dot-level Grade 1 Braille pipeline (works on any embossed or printed
  Braille page without dataset/internet dependency):
    1. Image quality gates (paper contrast, Laplacian sharpness, dot resolution).
    2. Illumination flattening + shadow/highlight + top-hat/black-hat dot detection.
    3. Grid orientation deskew + nearest-neighbor dot-pitch estimation.
    4. Global 2x3 cell lattice fitting per text line (row y0 fit + column anchor fit).
    5. Exact Grade 1 Braille lookup (unknown 6-bit cells map to '?' / "unknown", never guessed).

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

        lower_name = clean_name.lower()
        if lower_name in cls.NAMED_DICT:
            return cls.NAMED_DICT[lower_name]

        for prefix in ("letter_", "char_", "digit_"):
            if lower_name.startswith(prefix):
                remainder = clean_name[len(prefix):]
                if len(remainder) == 1 and remainder.isalpha():
                    return remainder.upper()
                if remainder.isdigit():
                    return remainder

        if len(clean_name) == 6 and set(clean_name).issubset({"0", "1"}):
            # Unrecognized 6-bit binary cell must map to unknown ("?"), never fall through to isdigit().
            return ""

        if len(clean_name) == 1 and clean_name.isalpha():
            return clean_name.upper()
        if clean_name.isdigit():
            return clean_name

        return ""

    @classmethod
    def filter_valid_detections(cls, boxes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate candidate Braille cell or dot boxes:
          - Supports tall whole Braille cells (aspect >= 0.35) as well as circular dots.
          - Requires size consistency (within 55% of median area/scale) and >= 2 boxes.
        """
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
    # Classical Dot-Level Grade 1 Braille Recognizer (Primary Offline Path)
    # =========================================================================

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Flatten uneven paper lighting and enhance embossed dot contrast via CLAHE."""
        if image is None or cv2 is None:
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
        if lap_var < 4.5:
            return False, "blurry", lap_var

        return True, "ok", lap_var

    @staticmethod
    def detect_dots(image: np.ndarray) -> Tuple[List[Dict[str, float]], float]:
        """
        Detect embossed or printed Braille dots using illumination flattening
        and Difference-of-Gaussians (captures both dark shadows and bright highlights).
        Returns (dots_list, median_diameter_px) where each dot is {'x', 'y', 'd', 'score'}.
        """
        if image is None or cv2 is None:
            return [], 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]

        # Scale-adaptive smoothing and DoG response
        gray_f = gray.astype(np.float32)
        blur_fine = cv2.GaussianBlur(gray_f, (3, 3), 0.8)
        bg_k = max(21, (min(w, h) // 18) | 1)
        bg = cv2.GaussianBlur(gray_f, (bg_k, bg_k), 0)

        # Positive (bright bump) + negative (dark shadow / ink dot) deviation from local paper plane
        diff = blur_fine - bg
        dark_resp = np.clip(-diff, 0.0, 255.0)
        light_resp = np.clip(diff, 0.0, 255.0)

        # If a dot is shaded (bright highlight beside dark shadow), merge shifted pairs
        # and also keep pure dark spots (printed or top-lit pits).
        # Morphological closing on dark_resp fills thin shadow crescents into solid dot blobs.
        combined = np.maximum(dark_resp, light_resp * 0.85)
        # Local noise floor based on MAD + adaptive shoulder of dot peak response
        med = float(np.median(combined))
        mad = float(np.median(np.abs(combined - med))) + 1e-5
        peak_99 = float(np.percentile(combined, 99.5))
        thresh_val = max(8.0, med + 4.5 * mad, peak_99 * 0.28)

        mask = (combined >= thresh_val).astype(np.uint8) * 255
        # Close tiny shadow/highlight crescent gap into single dot circle
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        # Open to strip 1px sensor specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        raw_dots: List[Dict[str, float]] = []
        min_dim = 3.0
        max_dim = max(40.0, min(w, h) * 0.09)

        for lbl in range(1, num_labels):
            bx = float(stats[lbl, cv2.CC_STAT_LEFT])
            by = float(stats[lbl, cv2.CC_STAT_TOP])
            bw = float(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = float(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if bx <= 2.0 or by <= 2.0 or (bx + bw) >= (w - 2.0) or (by + bh) >= (h - 2.0):
                continue
            if bw < min_dim or bh < min_dim or bw > max_dim or bh > max_dim:
                continue
            aspect = min(bw, bh) / max(1e-6, max(bw, bh))
            if aspect < 0.42:
                continue
            fill_ratio = area / max(1.0, bw * bh)
            if fill_ratio < 0.30:
                continue
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            diam = math.sqrt(4.0 * area / math.pi)
            score = float(np.max(combined[labels == lbl]))
            raw_dots.append({"x": cx, "y": cy, "d": diam, "area": area, "score": score})

        if len(raw_dots) < 3:
            return raw_dots, (float(np.median([d["d"] for d in raw_dots])) if raw_dots else 0.0)

        # Merge duplicate shadow+highlight blobs belonging to the same embossed dot
        # (embossed dot pair distance <= ~1.3 * median diameter)
        med_d = float(np.median([d["d"] for d in raw_dots]))
        raw_dots.sort(key=lambda d: d["score"], reverse=True)
        merged: List[Dict[str, float]] = []
        merge_radius = max(4.0, med_d * 1.25)
        for d in raw_dots:
            area_ok = 0.20 * (med_d ** 2 * math.pi / 4.0) <= d["area"] <= 4.0 * (med_d ** 2 * math.pi / 4.0)
            if not area_ok:
                continue
            near = False
            for m in merged:
                if math.hypot(d["x"] - m["x"], d["y"] - m["y"]) <= merge_radius:
                    near = True
                    break
            if not near:
                merged.append(d)

        final_d = float(np.median([d["d"] for d in merged])) if merged else 0.0
        return merged, final_d

    @staticmethod
    def _estimate_skew_and_pitch(dots: List[Dict[str, float]], dot_diam: float) -> Tuple[float, float]:
        """
        Estimate grid rotation angle (degrees, in [-20, 20]) and within-cell
        dot-to-dot pitch p_dot from nearest-neighbor vectors.
        """
        if len(dots) < 4:
            return 0.0, max(12.0, dot_diam * 2.5)

        pts = np.array([[d["x"], d["y"]] for d in dots], dtype=np.float64)
        n = len(pts)
        angles_deg: List[float] = []
        cand_dists: List[float] = []

        max_search_r = max(25.0, dot_diam * 5.2)
        min_search_r = max(5.0, dot_diam * 1.02)

        for i in range(n):
            diffs = pts - pts[i]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            mask = (dists >= min_search_r) & (dists <= max_search_r)
            if not np.any(mask):
                continue
            valid_idx = np.where(mask)[0]
            # Sort nearest neighbors
            order = valid_idx[np.argsort(dists[valid_idx])][:4]
            for j in order:
                dx, dy = diffs[j]
                dist = dists[j]
                cand_dists.append(float(dist))
                ang = math.degrees(math.atan2(dy, dx))
                # Fold angle modulo 90 degrees into [-45, 45]
                ang_folded = ((ang + 45.0) % 90.0) - 45.0
                if abs(ang_folded) <= 20.0:
                    angles_deg.append(ang_folded)

        skew_deg = float(np.median(angles_deg)) if len(angles_deg) >= 3 else 0.0

        # Rotate points by -skew_deg to measure axial dx/dy for exact dot pitch
        rad = math.radians(-skew_deg)
        c, s = math.cos(rad), math.sin(rad)
        rot = np.column_stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c])

        axial_dists: List[float] = []
        for i in range(n):
            diffs = np.abs(rot - rot[i])
            for j in range(n):
                if i == j:
                    continue
                dx, dy = diffs[j]
                dist = math.hypot(dx, dy)
                if min_search_r <= dist <= max_search_r:
                    # Strictly horizontal or strictly vertical dot neighbors within/across cells
                    if dy <= dot_diam * 0.55 and dx >= min_search_r:
                        axial_dists.append(float(dx))
                    elif dx <= dot_diam * 0.55 and dy >= min_search_r:
                        axial_dists.append(float(dy))

        if axial_dists:
            # Find dense lowest cluster corresponding to 1x dot pitch (p_dot)
            axial_arr = np.sort(np.array(axial_dists))
            p_min = float(np.percentile(axial_arr, 15))
            short_cluster = axial_arr[(axial_arr >= p_min * 0.72) & (axial_arr <= p_min * 1.38)]
            p_dot = float(np.median(short_cluster)) if short_cluster.size >= 2 else float(p_min)
        elif cand_dists:
            p_dot = float(np.percentile(cand_dists, 20))
        else:
            p_dot = max(12.0, dot_diam * 2.5)

        return skew_deg, max(dot_diam * 1.4, p_dot)

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

        # Rotate dots around image center to deskew
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

        # 1. Cluster into text lines by Y separation (> 2.4 * p_dot starts a new line)
        rot_dots.sort(key=lambda d: d["ry"])
        lines: List[List[Dict[str, float]]] = []
        for d in rot_dots:
            if not lines:
                lines.append([d])
            else:
                prev_ys = [item["ry"] for item in lines[-1]]
                if d["ry"] - max(prev_ys) > (p_dot * 2.35):
                    lines.append([d])
                else:
                    lines[-1].append(d)

        cell_pitch = p_dot * 2.5  # Standard Grade 1 horizontal cell-to-cell pitch (~2.4-2.6x p_dot)
        # Refine cell_pitch from global paired horizontal gaps across all lines if visible
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
            best_hits = -1
            for y_anchor in ys:
                for row_offset in (0, 1, 2):
                    cand_y0 = y_anchor - row_offset * p_dot
                    # Compute snap error to nearest row in {0, 1, 2}
                    row_dists = np.min(
                        np.abs(ys[:, None] - (cand_y0 + np.array([0.0, p_dot, 2.0 * p_dot])[None, :])),
                        axis=1,
                    )
                    hits = int(np.sum(row_dists <= p_dot * 0.42))
                    err = float(np.sum(row_dists))
                    if hits > best_hits or (hits == best_hits and err < best_err):
                        best_hits = hits
                        best_err = err
                        best_y0 = float(cand_y0)

            # 3. Cluster horizontal dot positions into vertical sub-columns
            line_dots.sort(key=lambda d: d["rx"])
            sub_cols: List[List[Dict[str, float]]] = []
            for d in line_dots:
                if not sub_cols:
                    sub_cols.append([d])
                else:
                    mean_x = float(np.mean([x["rx"] for x in sub_cols[-1]]))
                    if abs(d["rx"] - mean_x) <= p_dot * 0.42:
                        sub_cols[-1].append(d)
                    else:
                        sub_cols.append([d])

            col_xs = [float(np.mean([d["rx"] for d in sc])) for sc in sub_cols]

            # Find paired intra-cell columns (dx within [0.68*p_dot, 1.38*p_dot]) to anchor grid
            pair_col1_xs: List[float] = []
            i = 0
            while i < len(col_xs) - 1:
                dx = col_xs[i + 1] - col_xs[i]
                if 0.68 * p_dot <= dx <= 1.38 * p_dot:
                    pair_col1_xs.append(col_xs[i])
                    i += 2
                else:
                    i += 1

            # Estimate line's cell_pitch from consecutive anchors or col gaps
            pitch_cands: List[float] = []
            for k in range(len(col_xs) - 1):
                dx = col_xs[k + 1] - col_xs[k]
                if 1.85 * p_dot <= dx <= 3.35 * p_dot:
                    pitch_cands.append(dx)
            for k in range(len(pair_col1_xs) - 1):
                dx = pair_col1_xs[k + 1] - pair_col1_xs[k]
                if 1.85 * p_dot <= dx <= 3.35 * p_dot:
                    pitch_cands.append(dx)
            line_cell_pitch = float(np.median(pitch_cands)) if pitch_cands else cell_pitch

            x_anchor = pair_col1_xs[0] if pair_col1_xs else col_xs[0]

            # Assign every dot in this line to (cell_idx, col in {0,1}, row in {0,1,2})
            cells_bits: Dict[int, List[int]] = {}
            for sc, cx in zip(sub_cols, col_xs):
                k_est = round((cx - x_anchor) / line_cell_pitch)
                rem = cx - (x_anchor + k_est * line_cell_pitch)
                if abs(rem) <= p_dot * 0.46:
                    c_idx, col = k_est, 0
                elif abs(rem - p_dot) <= p_dot * 0.46:
                    c_idx, col = k_est, 1
                elif abs(rem + line_cell_pitch - p_dot) <= p_dot * 0.46:
                    c_idx, col = k_est - 1, 1
                else:
                    c_idx, col = k_est, (0 if abs(rem) < abs(rem - p_dot) else 1)

                bits = cells_bits.setdefault(c_idx, [0, 0, 0, 0, 0, 0])
                for d in sc:
                    row = int(round((d["ry"] - best_y0) / p_dot))
                    row = max(0, min(2, row))
                    bit_idx = col * 3 + row
                    bits[bit_idx] = 1

            # Decode sorted cells in this line
            sorted_c_idxs = sorted(cells_bits.keys())
            line_str = ""
            is_number = False
            is_capital = False
            prev_c_idx: Optional[int] = None

            for c_idx in sorted_c_idxs:
                if prev_c_idx is not None and (c_idx - prev_c_idx) > 1:
                    line_str += " "
                    is_number = False
                    is_capital = False
                    word_idx += 1
                prev_c_idx = c_idx

                bits = cells_bits[c_idx]
                bit_str = "".join(str(b) for b in bits)
                active_dots = [idx + 1 for idx, b in enumerate(bits) if b == 1]
                dots_str = "Dots " + "-".join(str(d) for d in active_dots) if active_dots else "No dots"

                char = cls.BINARY_DICT.get(bit_str)
                if char is None:
                    char = UNKNOWN_CELL_CHAR

                # Compute unrotated visual bounding box center for debug visualization
                rx_c = x_anchor + c_idx * line_cell_pitch + 0.5 * p_dot
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
                    out_c = cls.NUM_MAP[char] if char in cls.NUM_MAP else char
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
        Lightweight presence check (~3-5 Hz on Pi 5) using the cheap classical dot
        detector at reduced resolution without running YOLO inference.
        """
        if image is None or cv2 is None:
            return False
        try:
            ok, _reason, _lap = self.check_image_quality(image)
            if not ok:
                return False

            # Downsample moderately for fast ~3-5 Hz preview presence check while keeping dots >= 5px
            h, w = image.shape[:2]
            scale = 1.0
            if max(w, h) > 800:
                scale = 640.0 / float(max(w, h))
                small = cv2.resize(image, (int(w * scale), int(h * scale)))
            else:
                small = image

            dots, _ = self.detect_dots(small)
            return len(dots) >= 3
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

        dots, dot_diam = self.detect_dots(image)
        h, w = image.shape[:2]

        if len(dots) >= 2:
            if len(dots) >= 4 and dot_diam < 5.0 and max(w, h) >= 480:
                return {"readable": False, "reason": "too_small", "text": "", "cells": [], "dots": dots, "dot_diameter": dot_diam, "lap_var": lap_var}
            if len(dots) >= 4 and dot_diam > min(w, h) * 0.08:
                return {"readable": False, "reason": "too_large", "text": "", "cells": [], "dots": dots, "dot_diameter": dot_diam, "lap_var": lap_var}

            classical_res = self.decode_dots_to_text(dots, dot_diam, image.shape)
            classical_res["lap_var"] = lap_var
            classical_res["dots"] = dots
            if classical_res.get("readable"):
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
