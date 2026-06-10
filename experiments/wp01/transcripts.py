"""Bootstrap and load in-image text transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from experiments.wp01.config import OCR_PIPELINE, OCR_PREPROCESS
from memes_base.ocr_pipelines import run_pipeline

from experiments.wp01.schema import load_json, new_transcripts_manifest, save_json


def bootstrap_transcripts(
    sample: dict[str, Any],
    wp01_root: Path,
    *,
    ocr_pipeline: str = OCR_PIPELINE,
    preprocess_mode: str = OCR_PREPROCESS,
    source_mode: str = "title_proxy",
) -> dict[str, Any]:
    path = wp01_root / "transcripts.json"
    if path.exists():
        return load_json(path)

    manifest = new_transcripts_manifest()
    items: dict[str, Any] = manifest["items"]

    for entry in sample["items"]:
        item_id = entry["id"]
        if source_mode == "title_proxy":
            in_image = entry.get("title", "") or entry.get("post_text", "")
            items[item_id] = {
                "in_image_text": in_image,
                "source": "title_proxy",
                "ocr_confidence": None,
                "annotator": None,
            }
            continue

        image_path = wp01_root / entry["local_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = run_pipeline(
            image,
            ocr_pipeline,
            apply_preprocess=preprocess_mode != "none",
            preprocess_mode=preprocess_mode,
        )
        in_image = result.text or ""
        source = "ocr_bootstrap"
        if not in_image.strip() and entry.get("title"):
            in_image = entry["title"]
            source = "title_fallback"
        items[item_id] = {
            "in_image_text": in_image,
            "source": source,
            "ocr_confidence": result.confidence,
            "annotator": None,
        }

    save_json(path, manifest)
    return manifest
