"""Three WP-01 embedding pipelines: VLM-native, hybrid, OCR-status-quo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from memes_base.ocr_pipelines import run_pipeline

from experiments.wp01.embedders import ClipBackend


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    use_image: bool
    use_post_text: bool
    use_ocr_text: bool


PIPELINES = {
    "p_vlm": PipelineSpec("p_vlm", use_image=True, use_post_text=True, use_ocr_text=False),
    "p_vlm_only": PipelineSpec("p_vlm_only", use_image=True, use_post_text=False, use_ocr_text=False),
    "p_ocr": PipelineSpec("p_ocr", use_image=True, use_post_text=True, use_ocr_text=True),
}


def _combine_text(post_text: str, ocr_text: str) -> str:
    parts = [p.strip() for p in (post_text, ocr_text) if p and p.strip()]
    return " ".join(parts)


def embed_pipeline(
    backend: ClipBackend,
    images_bgr: Sequence[np.ndarray],
    post_texts: Sequence[str],
    *,
    pipeline: PipelineSpec,
    ocr_texts: Sequence[str] | None = None,
    ocr_preprocess: str = "white_text",
) -> np.ndarray:
    parts: list[np.ndarray] = []

    if pipeline.use_image:
        parts.append(backend.encode_images(images_bgr))

    text_inputs: list[str] = []
    if pipeline.use_post_text or pipeline.use_ocr_text:
        for idx, post in enumerate(post_texts):
            ocr = ""
            if pipeline.use_ocr_text:
                if ocr_texts is not None:
                    ocr = ocr_texts[idx]
                else:
                    result = run_pipeline(
                        images_bgr[idx],
                        "easyocr",
                        apply_preprocess=True,
                        preprocess_mode=ocr_preprocess,
                    )
                    ocr = result.text or ""
            combined = _combine_text(post, ocr) if pipeline.use_ocr_text else post
            if not combined.strip():
                combined = "[empty]"
            text_inputs.append(combined)
        parts.append(backend.encode_texts(text_inputs))

    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)
