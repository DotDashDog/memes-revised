"""Sample selection from Reddit parquet shards."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from experiments.wp01.io_utils import load_image_from_url
from experiments.wp01.dhash import dhash
from experiments.wp01.schema import load_json, new_sample_manifest, save_json


def collect_parquet_paths(dataset_dir: Path) -> list[Path]:
    if dataset_dir.is_file() and dataset_dir.suffix == ".parquet":
        return [dataset_dir]
    if dataset_dir.is_dir():
        return sorted(dataset_dir.glob("*.parquet"))
    return []


def load_candidate_rows(
    parquet_paths: list[Path],
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for path in parquet_paths:
        df = pd.read_parquet(path, columns=["img_url", "title", "text", "link", "score", "created_utc"])
        for _, row in df.iterrows():
            url = row.get("img_url")
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            rows.append(
                {
                    "img_url": url,
                    "title": str(row.get("title") or ""),
                    "post_text": str(row.get("text") or ""),
                    "link": str(row.get("link") or ""),
                    "score": int(row.get("score") or 0),
                    "created_utc": int(row.get("created_utc") or 0),
                }
            )
        if max_rows > 0 and len(rows) >= max_rows * 3:
            break
    rng.shuffle(rows)
    if max_rows > 0:
        rows = rows[: max_rows * 3]
    return rows


def _post_id_from_link(link: str) -> str:
    m = re.search(r"/comments/([^/]+)/", link)
    return m.group(1) if m else re.sub(r"[^a-zA-Z0-9_-]", "_", link)[:32]


def classify_stratum(title: str, post_text: str) -> str:
    """Fast pre-download stratification from post metadata (refined post-OCR in report)."""
    title_len = len(title.strip())
    post_len = len(post_text.strip())
    if title_len >= 20 or post_len >= 40:
        return "text_heavy"
    if title_len <= 4 and post_len == 0:
        return "image_heavy"
    return "mixed"


def build_sample(
    dataset_dir: Path,
    output_dir: Path,
    *,
    target_items: int = 200,
    max_pool_rows: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    parquet_paths = collect_parquet_paths(dataset_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}")

    pool = load_candidate_rows(parquet_paths, max_rows=max_pool_rows, seed=seed)
    rng = random.Random(seed)

    buckets: dict[str, list[dict[str, Any]]] = {"text_heavy": [], "image_heavy": [], "mixed": []}
    per_stratum = max(1, target_items // 3)

    attempts = 0
    max_attempts = target_items * 25
    next_item_idx = 1
    for row in pool:
        total = sum(len(b) for b in buckets.values())
        if total >= target_items:
            break
        attempts += 1
        if attempts > max_attempts:
            break
        image = load_image_from_url(row["img_url"])
        if image is None:
            continue
        stratum = classify_stratum(row["title"], row["post_text"])
        strata_filled = all(len(buckets[k]) >= per_stratum for k in buckets)
        if len(buckets[stratum]) >= per_stratum and not strata_filled:
            continue

        item_id = f"meme-{next_item_idx:05d}"
        next_item_idx += 1
        local_name = f"{item_id}.jpg"
        local_path = images_dir / local_name
        cv2.imwrite(str(local_path), image)

        post_id = _post_id_from_link(row["link"])
        item = {
            "id": item_id,
            "post_id": post_id,
            "img_url": row["img_url"],
            "title": row["title"],
            "post_text": row["post_text"],
            "score": row["score"],
            "created_utc": row["created_utc"],
            "stratum": stratum,
            "local_path": f"images/{local_name}",
            "dhash": dhash(image),
        }
        buckets[stratum].append(item)
        total = sum(len(b) for b in buckets.values())
        if total % 5 == 0:
            print(f"  sample: {total}/{target_items} ({stratum})", flush=True)

    items: list[dict[str, Any]] = []
    for key in ("text_heavy", "image_heavy", "mixed"):
        items.extend(buckets[key])

    rng.shuffle(items)
    manifest = new_sample_manifest(seed)
    manifest["items"] = items
    manifest["n_items"] = len(items)
    manifest["stratum_counts"] = {k: sum(1 for it in items if it["stratum"] == k) for k in buckets}
    save_json(output_dir / "sample.json", manifest)
    return manifest


def load_sample(output_dir: Path) -> dict[str, Any]:
    return load_json(output_dir / "sample.json")
