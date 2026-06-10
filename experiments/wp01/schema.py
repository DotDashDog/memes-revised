"""JSON schemas and I/O helpers for WP-01 artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def new_sample_manifest(seed: int) -> dict[str, Any]:
    return {
        "version": 1,
        "created_utc": _utc_now(),
        "seed": seed,
        "description": (
            "WP-01 stratified meme sample. Items include post metadata and "
            "local image paths under images/."
        ),
        "items": [],
    }


def new_transcripts_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "created_utc": _utc_now(),
        "description": (
            "Ground-truth or bootstrap transcripts of in-image text. "
            "source=manual is preferred; ocr_bootstrap is a proxy for development."
        ),
        "items": {},
    }


def new_candidates_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "created_utc": _utc_now(),
        "description": "Per-meme candidate strings for image-to-text retrieval.",
        "items": {},
    }
