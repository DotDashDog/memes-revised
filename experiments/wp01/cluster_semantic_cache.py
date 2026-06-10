#!/usr/bin/env python3
"""Cluster a cached semantic embedding run and print example memes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.wp01.clustering_eval import silhouette_if_applicable
from experiments.wp01.config import WP01_ROOT
from experiments.wp01.schema import load_json, save_json


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return vectors / denom


def _load_embeddings(cache_dir: Path, text_key: str, image_weight: float, text_weight: float) -> tuple[list[str], np.ndarray]:
    image = np.load(cache_dir / "embeddings" / "image_openclip_b16_laion2b.npz")
    text = np.load(cache_dir / "embeddings" / f"text_{text_key}.npz")
    image_ids = [str(x) for x in image["ids"]]
    text_ids = [str(x) for x in text["ids"]]
    if image_ids != text_ids:
        raise ValueError("Image and text embedding ids do not align")

    image_vectors = _normalize_rows(image["embeddings"].astype(np.float32)) * image_weight
    text_vectors = _normalize_rows(text["embeddings"].astype(np.float32)) * text_weight
    return image_ids, np.concatenate([image_vectors, text_vectors], axis=1)


def _cluster_hdbscan(
    embeddings: np.ndarray,
    *,
    seed: int,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import hdbscan
    import umap

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    n = len(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, {
        "algorithm": "hdbscan_umap",
        "n_points": n,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / n if n else 0.0,
        "silhouette": silhouette_if_applicable(embeddings, labels),
    }


def _cluster_kmeans(
    embeddings: np.ndarray,
    *,
    seed: int,
    n_clusters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(embeddings)
    return labels, {
        "algorithm": "kmeans",
        "n_points": len(labels),
        "n_clusters": n_clusters,
        "n_noise": 0,
        "noise_fraction": 0.0,
        "silhouette": silhouette_if_applicable(embeddings, labels),
    }


def _item_lookup(cache_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    sample = load_json(cache_dir / "sample.json")
    texts = load_json(cache_dir / "text_inputs.json")
    return (
        {item["id"]: item for item in sample["items"]},
        {item["id"]: item["text"] for item in texts["items"]},
    )


def _examples(
    labels: np.ndarray,
    ids: list[str],
    sample_by_id: dict[str, dict[str, Any]],
    text_by_id: dict[str, str],
    *,
    n_clusters: int,
    examples_per_cluster: int,
) -> list[dict[str, Any]]:
    clusters = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = [idx for idx, value in enumerate(labels) if value == label]
        clusters.append((label, members))
    clusters.sort(key=lambda row: len(row[1]), reverse=True)

    out = []
    for label, members in clusters[:n_clusters]:
        rows = []
        for idx in members[:examples_per_cluster]:
            item_id = ids[idx]
            item = sample_by_id[item_id]
            rows.append(
                {
                    "id": item_id,
                    "title": item.get("title", ""),
                    "post_text": item.get("post_text", ""),
                    "image_path": item.get("local_path", ""),
                    "text_input": text_by_id.get(item_id, "")[:500],
                }
            )
        out.append({"cluster": int(label), "size": len(members), "items": rows})
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Semantic Cache Clustering",
        "",
        f"Cache: `{result['cache_dir']}`",
        f"Text embedding: `{result['text_key']}`",
        f"Algorithm: `{result['algorithm']}`",
        f"Image/Text weights: {result['image_weight']}/{result['text_weight']}",
        "",
        f"Clusters: {result['n_clusters']}",
        f"Noise fraction: {result['noise_fraction']:.3f}",
        f"Silhouette: {result['silhouette'] if result['silhouette'] is not None else 'n/a'}",
        "",
    ]
    for cluster in result["examples"]:
        lines.append(f"## Cluster {cluster['cluster']} (n={cluster['size']})")
        for item in cluster["items"]:
            title = item["title"] or "[no title]"
            text = item["text_input"].replace("\n", " | ")
            lines.append(f"- `{item['id']}` — {title} — {text}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster cached semantic embeddings.")
    parser.add_argument("--cache-dir", type=Path, default=WP01_ROOT / "semantic-embedding-1000-002")
    parser.add_argument("--text-key", default="openai_text-embedding-3-small")
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--algorithm", choices=("hdbscan", "kmeans"), default="hdbscan")
    parser.add_argument("--kmeans-clusters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-example-clusters", type=int, default=3)
    parser.add_argument("--examples-per-cluster", type=int, default=5)
    args = parser.parse_args()

    ids, embeddings = _load_embeddings(args.cache_dir, args.text_key, args.image_weight, args.text_weight)
    if args.algorithm == "kmeans":
        labels, info = _cluster_kmeans(embeddings, seed=args.seed, n_clusters=args.kmeans_clusters)
    else:
        labels, info = _cluster_hdbscan(
            embeddings,
            seed=args.seed,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
    sample_by_id, text_by_id = _item_lookup(args.cache_dir)
    result = {
        "cache_dir": str(args.cache_dir),
        "text_key": args.text_key,
        "image_weight": args.image_weight,
        "text_weight": args.text_weight,
        "labels": labels.tolist(),
        **info,
        "examples": _examples(
            labels,
            ids,
            sample_by_id,
            text_by_id,
            n_clusters=args.n_example_clusters,
            examples_per_cluster=args.examples_per_cluster,
        ),
    }

    out_dir = args.cache_dir / f"cluster_openai_text_embedding_3_small_{args.algorithm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "summary.json", result)
    write_markdown(out_dir / "report.md", result)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
