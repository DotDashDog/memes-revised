"""Embedding compositions.

Current (ADR 0018, 2026-11-28): `embed_channels` — an image channel and a text channel kept
as two separate vectors, the text embedded with a sentence model.

Superseded: `embed_pipeline` and `PIPELINES` — single concatenated vectors with the text
embedded by the CLIP text tower (ADR 0006/0017). Kept unchanged, including the "[empty]"
placeholder for text-free images, only so that recorded runs can be reproduced. Do not use
them for new work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from memes_base.ocr_pipelines import run_pipeline

from experiments.wp01.embedders import ClipBackend

# Sentence model for the text channel (ADR 0018). The revision is pinned so that stored
# vectors stay reproducible; changing either line means re-embedding.
TEXT_MODEL = "BAAI/bge-base-en-v1.5"
TEXT_MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


@dataclass(frozen=True)
class MemeChannels:
    """The two channels of ADR 0018, row-aligned.

    image:    (n, d_img) unit rows, OpenCLIP image embedding. Template identity.
    text:     (n, d_txt) unit rows from the sentence model; all-zero where has_text is False.
    has_text: (n,) bool. False when the image has no extracted text, in which case the text
              row carries no information and must not be compared with other rows.
    """

    image: np.ndarray
    text: np.ndarray
    has_text: np.ndarray


def load_text_encoder(device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL, revision=TEXT_MODEL_REVISION, device=device)


def embed_text_channel(texts: Sequence[str], encoder: Any, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Embed in-image text with the sentence model. Returns (vectors, has_text)."""
    clean = [(t or "").strip() for t in texts]
    has_text = np.array([bool(t) for t in clean], dtype=bool)
    # renamed in recent sentence-transformers; fall back for older versions
    dim = getattr(encoder, "get_embedding_dimension", None) or encoder.get_sentence_embedding_dimension
    dim = dim()
    out = np.zeros((len(clean), dim), dtype=np.float32)
    if has_text.any():
        vecs = encoder.encode([t for t in clean if t], batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=False)
        out[has_text] = np.asarray(vecs, dtype=np.float32)
    return out, has_text


def embed_channels(
    backend: ClipBackend,
    images_bgr: Sequence[np.ndarray],
    ocr_texts: Sequence[str],
    *,
    text_encoder: Any | None = None,
) -> MemeChannels:
    """ADR 0018 composition: image channel and text channel, not concatenated.

    `ocr_texts` is the extracted in-image text, one string per image ("" for none). Post
    title and body do not belong here (ADR 0017).
    """
    if len(images_bgr) != len(ocr_texts):
        raise ValueError(f"{len(images_bgr)} images but {len(ocr_texts)} texts")
    image = backend.encode_images(images_bgr)
    text, has_text = embed_text_channel(ocr_texts, text_encoder or load_text_encoder())
    return MemeChannels(image=image, text=text, has_text=has_text)


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
    # ADR 0017: the clustering/meme-identity composition — image + OCR, no post text
    "p_img_ocr": PipelineSpec("p_img_ocr", use_image=True, use_post_text=False, use_ocr_text=True),
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
            combined = _combine_text(post if pipeline.use_post_text else "", ocr)
            if not combined.strip():
                combined = "[empty]"
            text_inputs.append(combined)
        parts.append(backend.encode_texts(text_inputs))

    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)
