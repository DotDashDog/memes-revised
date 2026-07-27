from __future__ import annotations

import inspect
import os
import tempfile
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from paddleocr import PaddleOCR as PaddleOCRType
else:
    PaddleOCRType = Any

import cv2
import numpy as np

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


@dataclass(frozen=True)
class OcrBox:
    text: str
    confidence: Optional[float]
    points: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: Optional[float] = None
    boxes: Tuple[OcrBox, ...] = ()


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_meme_default(image: np.ndarray) -> np.ndarray:
    """Balanced preprocessing that works well for many meme styles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale_factor = max(1.0, 900.0 / max(height, width))
    if scale_factor > 1.0:
        gray = cv2.resize(
            gray,
            (int(width * scale_factor), int(height * scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    filtered = cv2.bilateralFilter(gray, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # Heuristic: invert if text appears white-on-dark
    black_ratio = np.mean(closed == 0)
    if black_ratio < 0.35:
        closed = cv2.bitwise_not(closed)
    return closed


def preprocess_meme_high_contrast(image: np.ndarray) -> np.ndarray:
    """Aggressive contrast boosting for outlined or low-contrast text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def preprocess_meme_white_text_mask(image: np.ndarray) -> np.ndarray:
    """Target common meme style: white text with black outline."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def preprocess_meme_white_or_black_text_mask(image: np.ndarray) -> np.ndarray:
    """
    Mask for either white or black text (most common meme text colors).
    We combine a white mask in HSV with a black mask from low luminance.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray, 0, 70)

    combined = cv2.bitwise_or(white_mask, black_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined


def preprocess_meme_outline_removed(image: np.ndarray) -> np.ndarray:
    """
    Attempt to suppress black outlines by focusing on the fill and smoothing edges.
    Works best for white text with dark outlines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = np.mean(binary == 0)
    if black_ratio < 0.4:
        binary = cv2.bitwise_not(binary)

    # Erode to shave off thin outlines, then dilate to restore fill thickness.
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    shaved = cv2.erode(binary, erode_kernel, iterations=1)
    restored = cv2.dilate(shaved, dilate_kernel, iterations=2)
    return restored


class PaddleOcrEngine:
    _instances: Dict[Tuple[str, str, bool, str], PaddleOCRType] = {}

    @classmethod
    def get(
        cls,
        det_algorithm: str = "DB",
        rec_algorithm: str = "SVTR_LCNet",
        use_angle_cls: bool = True,
        lang: str = "en",
    ) -> PaddleOCRType:
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Install with: pip install paddlepaddle paddleocr")
        key = (det_algorithm, rec_algorithm, use_angle_cls, lang)
        if key not in cls._instances:
            # PaddleOCR keyword support varies by version, so only pass supported kwargs.
            try:
                signature = inspect.signature(PaddleOCR.__init__)
                supported = set(signature.parameters)
            except (TypeError, ValueError):
                supported = set()

            candidate_kwargs = {
                "use_angle_cls": use_angle_cls,
                "lang": lang,
                "show_log": False,
                "det_algorithm": det_algorithm,
                "rec_algorithm": rec_algorithm,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": use_angle_cls,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            }
            kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

            unsupported = [k for k in candidate_kwargs if k not in supported]
            if unsupported:
                print(
                    "PaddleOCR init fallback: unsupported args stripped:",
                    ", ".join(unsupported),
                )

            cls._instances[key] = PaddleOCR(**kwargs)
        return cls._instances[key]


class EasyOcrEngine:
    _readers: Dict[Tuple[str, ...], "easyocr.Reader"] = {}

    @classmethod
    def get(cls, langs: Tuple[str, ...] = ("en",)) -> "easyocr.Reader":
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR is not installed. Install with: pip install easyocr")
        if langs not in cls._readers:
            cls._readers[langs] = easyocr.Reader(list(langs))
        return cls._readers[langs]


def _paddle_call(ocr: PaddleOCRType, image: np.ndarray, **kwargs):
    if hasattr(ocr, "ocr"):
        fn = ocr.ocr
    else:
        fn = ocr.predict

    def call_with_path(path, call_kwargs):
        try:
            return fn(path, **call_kwargs) if call_kwargs else fn(path)
        except Exception:
            return fn(path)

    try:
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        explicit_params = {
            p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        supported = set(signature.parameters)
    except (TypeError, ValueError):
        has_var_kwargs = False
        explicit_params = set()
        supported = set()

    if has_var_kwargs:
        filtered = {k: v for k, v in kwargs.items() if k in explicit_params}
        stripped = [k for k in kwargs if k not in filtered]
        if stripped:
            print(
                "PaddleOCR call fallback: unsupported args stripped:",
                ", ".join(stripped),
            )
        try:
            return fn(image, **filtered) if filtered else fn(image)
        except Exception:
            if isinstance(image, np.ndarray):
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_file.name
                temp_file.close()
                cv2.imwrite(temp_path, image)
                try:
                    return call_with_path(temp_path, filtered)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            raise

    if kwargs:
        filtered = {k: v for k, v in kwargs.items() if k in supported}
        stripped = [k for k in kwargs if k not in filtered]
        if stripped:
            print(
                "PaddleOCR call fallback: unsupported args stripped:",
                ", ".join(stripped),
            )
        try:
            return fn(image, **filtered) if filtered else fn(image)
        except Exception:
            if isinstance(image, np.ndarray):
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_file.name
                temp_file.close()
                cv2.imwrite(temp_path, image)
                try:
                    return call_with_path(temp_path, filtered)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            raise

    return fn(image)


def _parse_paddle_full_ocr(result) -> Tuple[str, Optional[float]]:
    if not result:
        return "", None

    if isinstance(result, tuple) and len(result) >= 2:
        result = result[1]
    if isinstance(result, list) and len(result) == 2 and isinstance(result[0], str):
        result = result[1]

    lines = []
    confidences = []
    for line in result:
        if isinstance(line, dict):
            if "rec_texts" in line:
                texts = line.get("rec_texts") or []
                scores = line.get("rec_scores") or []
                if isinstance(texts, str):
                    texts = [texts]
                if isinstance(scores, (float, int)):
                    scores = [scores]
                for idx, text in enumerate(texts):
                    if text:
                        lines.append(text)
                        if idx < len(scores):
                            confidences.append(scores[idx])
            elif "text" in line:
                text = line.get("text")
                conf = line.get("score")
                if text:
                    lines.append(text)
                    if isinstance(conf, (float, int)):
                        confidences.append(conf)
            continue

        if isinstance(line, (list, tuple)) and len(line) >= 2:
            candidate = line[1]
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                text = candidate[0]
                conf = candidate[1]
                if text:
                    lines.append(text)
                    if isinstance(conf, (float, int)):
                        confidences.append(conf)
            elif isinstance(candidate, str):
                lines.append(candidate)

    text = _normalize_text(" ".join(lines))
    if confidences:
        return text, float(np.mean(confidences))
    return text, None


def _paddle_supports_det_rec(ocr: PaddleOCRType) -> bool:
    if not hasattr(ocr, "ocr"):
        return False
    try:
        signature = inspect.signature(ocr.ocr)
        params = set(signature.parameters)
    except (TypeError, ValueError):
        return False
    return bool({"det", "rec", "cls"} & params)


def run_paddleocr_ppocrv5(image: np.ndarray) -> OcrResult:
    ocr = PaddleOcrEngine.get(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_angle_cls=False, lang="en")
    result = _paddle_call(ocr, image, cls=True)
    try:
        text, conf = _parse_paddle_full_ocr(result)
    except Exception as exc:
        print("PaddleOCR parse error:", exc)
        print("PaddleOCR raw result type:", type(result))
        try:
            preview = repr(result)
        except Exception:
            preview = "<unprintable>"
        print("PaddleOCR raw result preview:", preview[:1000])
        raise
    return OcrResult(text=text, confidence=conf)


def run_paddleocr_dbnet(image: np.ndarray) -> OcrResult:
    ocr = PaddleOcrEngine.get(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_angle_cls=False, lang="en")
    if not _paddle_supports_det_rec(ocr):
        # Older PaddleOCR versions can't split detection/recognition.
        print("DBNet fallback: PaddleOCR API lacks det/rec; using full OCR path.")
        return run_paddleocr_ppocrv5(image)

    detections = _paddle_call(ocr, image, det=True, rec=False, cls=False)
    if not detections:
        return OcrResult(text="")

    # If the API doesn't support det/rec, we may have a full OCR result already.
    if isinstance(detections, list) and detections and isinstance(detections[0], (list, tuple)) and len(detections[0]) >= 2:
        text, conf = _parse_paddle_full_ocr(detections)
        return OcrResult(text=text, confidence=conf)

    boxes = detections[0] if isinstance(detections, list) else detections
    if not boxes:
        return OcrResult(text="")

    crops = []
    for box in boxes:
        points = np.array(box).astype(np.int32)
        x_min = max(0, np.min(points[:, 0]))
        y_min = max(0, np.min(points[:, 1]))
        x_max = min(image.shape[1], np.max(points[:, 0]))
        y_max = min(image.shape[0], np.max(points[:, 1]))
        if x_max <= x_min or y_max <= y_min:
            continue
        crops.append(image[y_min:y_max, x_min:x_max])

    if not crops:
        return OcrResult(text="")

    rec_results = _paddle_call(ocr, crops, det=False, rec=True, cls=False)
    lines = []
    confidences = []
    for rec in rec_results:
        if len(rec) >= 1:
            text = rec[0][0]
            conf = rec[0][1]
            if text:
                lines.append(text)
                confidences.append(conf)

    text = _normalize_text(" ".join(lines))
    if confidences:
        return OcrResult(text=text, confidence=float(np.mean(confidences)))
    return OcrResult(text=text)


def run_easyocr_craft(image: np.ndarray) -> OcrResult:
    reader = EasyOcrEngine.get(("en",))
    results = reader.readtext(image, detail=1, paragraph=False)
    if not results:
        return OcrResult(text="")

    boxes = _parse_easyocr_boxes(results)
    return _easyocr_boxes_to_result(boxes)


def _parse_easyocr_boxes(results) -> List[OcrBox]:
    boxes = []
    for entry in results:
        if len(entry) >= 3:
            points = tuple((float(x), float(y)) for x, y in entry[0])
            text = entry[1]
            conf = entry[2]
            if text:
                boxes.append(OcrBox(text=text, confidence=float(conf), points=points))
        elif len(entry) >= 2:
            points = tuple((float(x), float(y)) for x, y in entry[0])
            text = entry[1]
            if text:
                boxes.append(OcrBox(text=text, confidence=None, points=points))
    return boxes


def _easyocr_boxes_to_result(boxes: List[OcrBox]) -> OcrResult:
    lines = [box.text for box in boxes if box.text]
    confidences = [box.confidence for box in boxes if box.confidence is not None]
    text = _normalize_text(" ".join(lines))
    if confidences:
        return OcrResult(text=text, confidence=float(np.mean(confidences)), boxes=tuple(boxes))
    return OcrResult(text=text, boxes=tuple(boxes))


def _box_bounds(box: OcrBox) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in box.points]
    ys = [point[1] for point in box.points]
    return min(xs), min(ys), max(xs), max(ys)


def _box_reading_order_key(box: OcrBox) -> Tuple[int, float]:
    x_min, y_min, _x_max, y_max = _box_bounds(box)
    y_center = (y_min + y_max) / 2
    # Bucket by approximate text line before sorting left-to-right.
    return int(y_center // 24), x_min


def _looks_like_ocr_junk(text: str, confidence: Optional[float]) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return True

    alnum_count = sum(ch.isalnum() for ch in normalized)
    alpha_count = sum(ch.isalpha() for ch in normalized)
    alnum_ratio = alnum_count / len(normalized)

    if confidence is not None and confidence < 0.18:
        return True
    if len(normalized) <= 2 and (confidence is None or confidence < 0.55):
        return True
    if alnum_ratio < 0.45:
        return True
    if alpha_count == 0 and len(normalized) <= 4 and (confidence is None or confidence < 0.65):
        return True
    return False


def run_easyocr_boxed_filtered(image: np.ndarray, batch_size: int = 1) -> OcrResult:
    reader = EasyOcrEngine.get(("en",))
    results = reader.readtext(image, detail=1, paragraph=False, batch_size=batch_size)
    if not results:
        return OcrResult(text="")

    raw_boxes = _parse_easyocr_boxes(results)
    filtered = [box for box in raw_boxes if not _looks_like_ocr_junk(box.text, box.confidence)]
    filtered.sort(key=_box_reading_order_key)
    return _easyocr_boxes_to_result(filtered)


def run_tesseract_sparse(image: np.ndarray) -> OcrResult:
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract is not installed. Install with: uv add pytesseract")
    config = "--psm 11 --oem 1"
    text = pytesseract.image_to_string(image, lang="eng", config=config)
    return OcrResult(text=_normalize_text(text), confidence=None)


def run_pipeline(
    image: np.ndarray,
    pipeline_name: str,
    apply_preprocess: bool = False,
    preprocess_mode: str = "white_text",
    batch_size: int = 1,
) -> OcrResult:
    pipeline_name = pipeline_name.lower()
    input_image = image
    if apply_preprocess:
        input_image = preprocess_image(image, preprocess_mode)

    if pipeline_name in {"paddleocr", "ppocrv5", "paddle"}:
        return run_paddleocr_ppocrv5(input_image)
    if pipeline_name in {"dbnet", "paddle_dbnet"}:
        return run_paddleocr_dbnet(input_image)
    if pipeline_name in {"easyocr", "craft"}:
        return run_easyocr_craft(input_image)
    if pipeline_name in {"easyocr_filtered", "easyocr_boxed", "easyocr_boxed_filtered"}:
        return run_easyocr_boxed_filtered(input_image, batch_size=batch_size)
    if pipeline_name in {"tesseract", "tess"}:
        return run_tesseract_sparse(input_image)

    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def preprocess_image(image: np.ndarray, preprocess_mode: str) -> np.ndarray:
    mode = preprocess_mode.lower()
    if mode == "none":
        return image
    if mode == "default":
        return preprocess_meme_default(image)
    if mode == "contrast":
        return preprocess_meme_high_contrast(image)
    if mode == "white_text":
        return preprocess_meme_white_text_mask(image)
    if mode in {"white_or_black", "white_black"}:
        return preprocess_meme_white_or_black_text_mask(image)
    if mode == "outline":
        return preprocess_meme_outline_removed(image)
    raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")


def available_pipelines() -> List[str]:
    pipelines = []
    if PADDLEOCR_AVAILABLE:
        pipelines.extend(["paddleocr", "dbnet"])
    if EASYOCR_AVAILABLE:
        pipelines.extend(["easyocr", "easyocr_filtered"])
    if TESSERACT_AVAILABLE:
        pipelines.append("tesseract")
    return pipelines
