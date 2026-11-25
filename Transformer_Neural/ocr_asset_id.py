
import os
import re
from typing import Optional, Tuple, List, Dict, Iterator

import cv2
import numpy as np
import easyocr

from config import ASSET_TYPE_MAP

EASYOCR_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "easyocr_models"))

# Kept for backward compatibility (other modules may import it).
# Only the ID itself is matched now; loose patterns like "TRANSFORMER.*01"
# or "BREAKER.*01" were removed because they caused false positives.
ASSET_PATTERNS: Dict[str, List[re.Pattern]] = {
    "TR-1": [re.compile(r"\bTR[- ]?0?1\b", re.IGNORECASE)],
    "TR-2": [re.compile(r"\bTR[- ]?0?2\b", re.IGNORECASE)],
    "CB-1": [re.compile(r"\bCB[- ]?0?1\b", re.IGNORECASE)],
    "CS-02": [re.compile(r"\bCS[- ]?0?2\b", re.IGNORECASE)],
}

# ---------------------------------------------------------------------------
# Tolerant ID matching: handles OCR confusions (O/0, I/l/1, Z/2, C8/CB, C5/CS),
# missing or odd dashes ("TR 1", "TR1", "TR—1", "CS02"), but still requires a
# valid prefix + valid number, so random text can't match.
# ---------------------------------------------------------------------------
_DASHES = str.maketrans({c: "-" for c in "\u2013\u2014\u2212\u2010\u2011\u2012\u2015_~"})
_NUM_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                          "|": "1", "!": "1", "Z": "2"})
_PREFIX_FIX = {"C8": "CB", "C5": "CS"}

_CANDIDATE = re.compile(
    r"(?<![A-Z0-9])(TR|CB|C8|CS|C5)\s*[-.:]?\s*([0-9OQDILZ|!]{1,2})(?![A-Z0-9])"
)

# (prefix, number) -> canonical asset id, built from ASSET_PATTERNS keys
_ID_LOOKUP: Dict[Tuple[str, int], str] = {}
for _aid in ASSET_PATTERNS:
    _m = re.fullmatch(r"([A-Z]+)-?0*(\d+)", _aid)
    if _m:
        _ID_LOOKUP[(_m.group(1), int(_m.group(2)))] = _aid


def _normalize(text: str) -> str:
    return str(text).upper().translate(_DASHES).strip()


def _find_ids(text: str) -> List[str]:
    """Return canonical asset IDs found in a text string."""
    found = []
    for m in _CANDIDATE.finditer(_normalize(text)):
        prefix = _PREFIX_FIX.get(m.group(1), m.group(1))
        digits = m.group(2).translate(_NUM_FIX)
        if not digits.isdigit():
            continue
        aid = _ID_LOOKUP.get((prefix, int(digits)))
        if aid:
            found.append(aid)
    return found


def _is_exact_id(text: str) -> bool:
    """True if the whole text is just the ID (e.g. 'TR-1', not 'ASSET ID: TR-1')."""
    return _CANDIDATE.fullmatch(_normalize(text)) is not None


class AssetIDDetector:
    """Robust OCR engine for reading and verifying substation asset ID plates."""

    MIN_CONF = 0.40          # min per-segment OCR confidence to accept a match
    EARLY_EXIT_CONF = 0.70   # stop trying extra preprocessing variants above this
    ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:. "

    def __init__(self, model_dir: str = EASYOCR_MODEL_DIR, gpu: bool = True):
        print(f"[OCR] Initializing GPU EasyOCR engine (models: {model_dir})...")
        os.makedirs(model_dir, exist_ok=True)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=model_dir,
            user_network_directory=model_dir,
            verbose=False,
        )
        print("[OCR] Engine initialized and ready.")

    # ------------------------------------------------------------------ utils
    def _read(self, img: np.ndarray) -> List:
        """Single OCR call with plate-friendly settings."""
        try:
            return self.reader.readtext(
                img,
                allowlist=self.ALLOWLIST,
                detail=1,
                paragraph=False,
                width_ths=0.8,      # merge split pieces like "TR" "-" "1"
                text_threshold=0.5,
                low_text=0.3,
            )
        except Exception as e:
            print(f"[OCR] readtext failed: {e}")
            return []

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Enhances contrast and sharpness for small or blurry text plates."""
        gray = crop if crop.ndim == 2 else cv2.cvtColor(self._to_bgr(crop), cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(enhanced, -1, kernel)

    def _variants(self, crop: np.ndarray) -> Iterator[np.ndarray]:
        """
        Lazily yields preprocessed versions of a plate crop.
        Tight crops (like a detector box around 'TR-1') are padded and scaled
        to a text-friendly height, which helps EasyOCR a lot.
        """
        bgr = self._to_bgr(crop)
        h, w = bgr.shape[:2]

        target_h = 96
        if h < target_h:
            scale = min(target_h / h, 4.0)
        elif h > 480:
            scale = 480 / h
        else:
            scale = 1.0
        if scale != 1.0:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interp)

        pad = max(8, int(0.15 * bgr.shape[0]))
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # 1) padded / scaled colour
        yield bgr

        # 2) CLAHE grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        enh = clahe.apply(gray)
        yield enh

        # 3) inverted (white text on black plates -> dark on light)
        yield cv2.bitwise_not(enh)

        # 4) Otsu binarization (both polarities)
        _, otsu = cv2.threshold(cv2.GaussianBlur(enh, (3, 3), 0), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu
        yield cv2.bitwise_not(otsu)

        # 5) original sharpening kernel
        yield self.preprocess_crop(bgr)

    # --------------------------------------------------------------- matching
    def match_texts_to_asset(self, raw_results: List) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Matches OCR text segments against valid Asset IDs.
        Returns: (asset_id, asset_type, confidence, raw_texts)
        """
        if not raw_results:
            return None, None, 0.0, []

        raw_texts, confs = [], []
        for item in raw_results:
            if len(item) >= 3:
                raw_texts.append(str(item[1]).strip())
                confs.append(float(item[2]))
            elif len(item) >= 2:
                raw_texts.append(str(item[1]).strip())
                confs.append(0.85)

        # Per-segment hits (confidence = that segment's own confidence,
        # not the average over unrelated noise segments).
        hits: Dict[str, List[float]] = {}
        for text, conf in zip(raw_texts, confs):
            if conf < self.MIN_CONF:
                continue
            for aid in _find_ids(text):
                hits.setdefault(aid, []).append(conf)

        # Fallback: ID split across segments ("TR" + "1")
        if not hits and raw_texts:
            joined = " ".join(raw_texts)
            for aid in _find_ids(joined):
                hits.setdefault(aid, []).append(float(np.mean(confs)))

        if not hits:
            return None, None, 0.0, raw_texts

        # Best asset: highest peak confidence, small bonus for repeated reads
        # (the plate shows the ID twice: big "TR-1" and "ASSET ID: TR-1").
        best_id = max(hits, key=lambda a: max(hits[a]) + 0.05 * (len(hits[a]) - 1))
        conf = min(1.0, max(hits[best_id]) + 0.05 * (len(hits[best_id]) - 1))
        return best_id, ASSET_TYPE_MAP.get(best_id, "Unknown"), conf, raw_texts

    def verify_crop(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Verifies whether a candidate plate crop contains a valid asset ID.
        Returns: (asset_id, asset_type, confidence, raw_texts) or (None, None, 0.0, texts)
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 20:
            return None, None, 0.0, []

        best = (None, None, 0.0)
        all_texts: List[str] = []

        for variant in self._variants(crop):
            results = self._read(variant)
            aid, atype, conf, texts = self.match_texts_to_asset(results)
            all_texts.extend(t for t in texts if t not in all_texts)
            if aid is not None and conf > best[2]:
                best = (aid, atype, conf)
                if conf >= self.EARLY_EXIT_CONF:
                    break

        if best[0] is not None:
            return best[0], best[1], best[2], all_texts
        return None, None, 0.0, all_texts

    # -------------------------------------------------------------- full frame
    def _scan(self, img: np.ndarray, scale: float, x_off: int, y_off: int) -> List[Dict]:
        hits = []
        for bbox, text, conf in self._read(img):
            conf = float(conf)
            if conf < self.MIN_CONF:
                continue
            ids = _find_ids(text)
            if not ids:
                continue
            pts = np.array(bbox, dtype=float) / scale
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            clean = _normalize(text)
            hits.append({
                "asset_id": ids[0],
                "asset_type": ASSET_TYPE_MAP.get(ids[0], "Unknown"),
                "bbox": [int(x1) + x_off, int(y1) + y_off, int(x2) + x_off, int(y2) + y_off],
                "conf": conf,
                "text": clean,
                # prefer the standalone big "TR-1" over "ASSET ID: TR-1"
                "_score": conf + (0.25 if _is_exact_id(clean) else 0.0),
            })
        return hits

    def detect_in_frame(self, frame: np.ndarray, search_region: str = "full") -> Optional[Dict]:
        """
        Scans a frame or region ('full', 'lower_half', 'center') with OCR.
        Returns dict with asset_id, asset_type, bbox, conf, text or None.
        """
        if frame is None or frame.size == 0:
            return None
        frame = self._to_bgr(frame)
        fh, fw = frame.shape[:2]

        if search_region == "lower_half":
            y0, x0 = int(fh * 0.45), 0
            roi = frame[y0:, :]
        elif search_region == "center":
            y0, x0 = int(fh * 0.2), int(fw * 0.1)
            roi = frame[y0:int(fh * 0.8), x0:int(fw * 0.9)]
        else:
            y0 = x0 = 0
            roi = frame

        # Pass 1: raw ROI
        hits = self._scan(roi, 1.0, x0, y0)

        # Pass 2: CLAHE-enhanced, 1.5x upscale (small / low-contrast plates)
        if not hits:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            up = cv2.resize(enh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            hits = self._scan(up, 1.5, x0, y0)

        # Pass 3: inverted (light text on dark plate)
        if not hits:
            hits = self._scan(cv2.bitwise_not(enh), 1.0, x0, y0)

        if not hits:
            return None

        best = max(hits, key=lambda h: h["_score"])
        best.pop("_score", None)
        return best
