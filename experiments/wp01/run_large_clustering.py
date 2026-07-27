#!/usr/bin/env python3
"""Run a larger cached-image clustering check for WP-01."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.wp01.clustering_eval import (
    assign_template_ids,
    cluster_embeddings,
    purity_recall,
    silhouette_if_applicable,
)
from experiments.wp01.config import OCR_PIPELINE, OCR_PREPROCESS, PRIMARY_IMAGE_ENCODER, WP01_ROOT
from experiments.wp01.dhash import dhash
from experiments.wp01.embedders import ClipBackend
from experiments.wp01.pipelines import PIPELINES, embed_pipeline
from experiments.wp01.schema import load_json, save_json
from memes_base.ocr_pipelines import run_pipeline


def _sample_lookup(wp01_root: Path) -> dict[str, dict[str, Any]]:
    path = wp01_root / "sample.json"
    if not path.exists():
        return {}
    sample = load_json(path)
    out = {}
    for entry in sample.get("items", []):
        out.setdefault(entry["id"], entry)
    return out


def load_cached_items(wp01_root: Path, target_items: int, seed: int) -> list[dict[str, Any]]:
    """Select items from the sample.json manifest, skipping byte-identical duplicates.

    The item universe is the manifest, not the images/ directory: images/ can hold
    orphan files from older sample generations, which have no metadata (their titles
    would silently embed as empty — the clustering-large-001 defect). Byte-identical
    files (reposts, or the same post drawn twice) are deduplicated so HDBSCAN density
    is not inflated by exact copies; the first drawn member of each group is kept.
    """
    lookup = _sample_lookup(wp01_root)
    image_paths = sorted((wp01_root / "images").glob("meme-*.jpg"))
    rng = np.random.default_rng(seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    items = []
    n_orphans = 0
    n_dups = 0
    seen_digests: set[str] = set()
    for image_path in shuffled:
        item_id = image_path.stem
        if item_id not in lookup:
            n_orphans += 1
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen_digests:
            n_dups += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        seen_digests.add(digest)
        entry = dict(lookup[item_id])
        entry.setdefault("title", "")
        entry.setdefault("post_text", "")
        entry.setdefault("stratum", "cached")
        entry["local_path"] = f"images/{image_path.name}"
        entry["_abs_path"] = image_path
        entry["dhash"] = entry.get("dhash") or dhash(image)
        items.append(entry)
        if len(items) >= target_items:
            break
    print(
        f"Item selection: kept {len(items)}, skipped {n_orphans} orphans "
        f"(not in sample.json) and {n_dups} byte-identical duplicates",
        flush=True,
    )
    return items


def build_ocr_texts(
    items: list[dict[str, Any]],
    *,
    ocr_pipeline: str,
    ocr_preprocess: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    texts = []
    rows = []
    for idx, entry in enumerate(items, start=1):
        image = cv2.imread(str(entry["_abs_path"]))
        start = time.perf_counter()
        result = run_pipeline(
            image,
            pipeline_name=ocr_pipeline,
            apply_preprocess=ocr_preprocess != "none",
            preprocess_mode=ocr_preprocess,
        )
        seconds = time.perf_counter() - start
        text = result.text or ""
        texts.append(text)
        rows.append(
            {
                "id": entry["id"],
                "ocr_text": text,
                "confidence": result.confidence,
                "seconds": seconds,
            }
        )
        if idx % 10 == 0:
            print(f"  OCR {idx}/{len(items)}", flush=True)
    return texts, rows


def _cluster_examples(
    labels: np.ndarray,
    items: list[dict[str, Any]],
    ocr_texts: list[str],
    max_clusters: int = 10,
) -> list[dict[str, Any]]:
    examples = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idxs = [idx for idx, value in enumerate(labels) if value == label]
        examples.append(
            {
                "cluster": int(label),
                "size": len(idxs),
                "items": [
                    {
                        "id": items[idx]["id"],
                        "title": items[idx].get("title", ""),
                        "ocr_text": ocr_texts[idx][:180],
                    }
                    for idx in idxs[:8]
                ],
            }
        )
    examples.sort(key=lambda row: row["size"], reverse=True)
    return examples[:max_clusters]


def run_large_clustering(
    *,
    wp01_root: Path,
    output_dir: Path,
    target_items: int,
    seed: int,
    encoder_key: str,
    ocr_pipeline: str,
    ocr_preprocess: str,
) -> dict[str, Any]:
    items = load_cached_items(wp01_root, target_items, seed)
    if len(items) < 10:
        raise RuntimeError(f"Too few cached images for clustering: {len(items)}")

    print(f"Using {len(items)} cached images", flush=True)
    ocr_texts, ocr_rows = build_ocr_texts(
        items,
        ocr_pipeline=ocr_pipeline,
        ocr_preprocess=ocr_preprocess,
    )

    backend = ClipBackend.load(encoder_key)
    images = [cv2.imread(str(entry["_abs_path"])) for entry in items]
    post_texts = [f"{entry.get('title', '')} {entry.get('post_text', '')}".strip() for entry in items]
    template_ids = assign_template_ids(items)

    results: dict[str, Any] = {
        "description": "Larger cached-image clustering check for WP-01.",
        "n_items": len(items),
        "encoder": encoder_key,
        "ocr_pipeline": ocr_pipeline,
        "ocr_preprocess": ocr_preprocess,
        "pipelines": {},
        "ocr": {
            "mean_seconds": float(np.mean([row["seconds"] for row in ocr_rows])),
            "mean_confidence": float(
                np.mean([row["confidence"] for row in ocr_rows if row["confidence"] is not None])
            ),
        },
    }

    for name, spec in PIPELINES.items():
        print(f"Embedding/clustering {name}", flush=True)
        start = time.perf_counter()
        embeddings = embed_pipeline(
            backend,
            images,
            post_texts,
            pipeline=spec,
            ocr_texts=ocr_texts,
        )
        embed_seconds = time.perf_counter() - start
        labels, info = cluster_embeddings(embeddings, seed=seed)
        pr = purity_recall(labels, template_ids)
        sil = silhouette_if_applicable(embeddings, labels)
        results["pipelines"][name] = {
            **info,
            **pr,
            "silhouette": sil,
            "embed_seconds": embed_seconds,
            "dim": int(embeddings.shape[1]),
            "examples": _cluster_examples(labels, items, ocr_texts),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", results)
    save_json(output_dir / "ocr_outputs.json", {"items": ocr_rows})
    write_markdown(output_dir / "report.md", results)
    return results


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# WP-01 Larger Cached-Image Clustering Check",
        "",
        f"Images: {results['n_items']}",
        f"Encoder: `{results['encoder']}`",
        f"OCR: `{results['ocr_pipeline']}:{results['ocr_preprocess']}`",
        f"Mean OCR seconds/image: {results['ocr']['mean_seconds']:.2f}",
        "",
        "| Pipeline | Clusters | Noise | Clustered | Purity | Recall | Silhouette | Embed sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in results["pipelines"].items():
        sil = row["silhouette"]
        sil_s = f"{sil:.3f}" if sil is not None else "n/a"
        lines.append(
            f"| `{name}` | {row['n_clusters']} | {row['noise_fraction']:.2f} | "
            f"{row['n_clustered']} | {row['purity']:.3f} | {row['recall']:.3f} | "
            f"{sil_s} | {row['embed_seconds']:.2f} |"
        )

    for name, row in results["pipelines"].items():
        lines.extend(["", f"## Example Clusters: `{name}`", ""])
        if not row["examples"]:
            lines.append("No non-noise clusters.")
            continue
        for cluster in row["examples"]:
            lines.append(f"### Cluster {cluster['cluster']} (n={cluster['size']})")
            for item in cluster["items"]:
                title = item["title"] or "[no title]"
                ocr = item["ocr_text"] or "[no OCR text]"
                lines.append(f"- `{item['id']}` — {title} — OCR: {ocr}")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run larger cached WP-01 clustering check.")
    parser.add_argument("--wp01-root", type=Path, default=WP01_ROOT)
    parser.add_argument("--output-dir", type=Path, default=WP01_ROOT / "clustering-large-001")
    parser.add_argument("--target-items", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", default=PRIMARY_IMAGE_ENCODER)
    parser.add_argument("--ocr-pipeline", default=OCR_PIPELINE)
    parser.add_argument("--ocr-preprocess", default=OCR_PREPROCESS)
    args = parser.parse_args()

    results = run_large_clustering(
        wp01_root=args.wp01_root,
        output_dir=args.output_dir,
        target_items=args.target_items,
        seed=args.seed,
        encoder_key=args.encoder,
        ocr_pipeline=args.ocr_pipeline,
        ocr_preprocess=args.ocr_preprocess,
    )
    print(f"Wrote {args.output_dir / 'report.md'} for {results['n_items']} images")


if __name__ == "__main__":
    main()
