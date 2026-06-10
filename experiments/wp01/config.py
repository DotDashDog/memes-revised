"""Frozen model identifiers and provenance notes for WP-01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WP01_ROOT = Path(__file__).resolve().parents[1] / "WP-01"
DEFAULT_SEED = 42

# Clustering defaults (documented in report)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3


@dataclass(frozen=True)
class ClipModelSpec:
    key: str
    family: str  # "openai_clip" | "openclip"
    model_id: str
    pretrained: str
    release_or_pool: str
    cutoff_note: str
    throughput_note: str


MODELS: dict[str, ClipModelSpec] = {
    "openai_clip_b32": ClipModelSpec(
        key="openai_clip_b32",
        family="openai_clip",
        model_id="openai/clip-vit-base-patch32",
        pretrained="openai",
        release_or_pool="WIT (~400M image-text pairs, sources not fully public)",
        cutoff_note=(
            "Model released Jan 2021. No per-sample crawl cutoff published; "
            "treat eval memes as post-2021 for contamination control."
        ),
        throughput_note="~170–220 img/s on RTX 3090 class GPU (ViT-B/32 class)",
    ),
    "openclip_b16_laion2b": ClipModelSpec(
        key="openclip_b16_laion2b",
        family="openclip",
        model_id="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        release_or_pool="LAION-2B subset of LAION-5B (Common Crawl; pool announced Mar 2022)",
        cutoff_note=(
            "No exact per-URL crawl window in model card; approximate upper bound "
            "early–mid 2022 from LAION-5B provenance."
        ),
        throughput_note="~150 img/s on RTX 3090 class GPU (public OpenCLIP benchmarks)",
    ),
}

# Primary encoder for pipeline comparison (OpenCLIP for quality/speed balance)
PRIMARY_IMAGE_ENCODER = "openclip_b16_laion2b"
# Text-fidelity retrieval runs on both encoders
TEXT_FIDELITY_ENCODERS = ("openai_clip_b32", "openclip_b16_laion2b")

OCR_PIPELINE = "easyocr_filtered"
OCR_PREPROCESS = "none"
