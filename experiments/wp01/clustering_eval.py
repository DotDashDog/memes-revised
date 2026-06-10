"""UMAP + HDBSCAN clustering evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics

from experiments.wp01.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    import hdbscan
    import umap

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    n = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    info = {
        "n_points": n,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / n if n else 0.0,
        "labels": labels.tolist(),
    }
    return labels, info


def purity_recall(
    cluster_labels: np.ndarray,
    template_ids: np.ndarray,
) -> dict[str, float]:
    """Purity/recall using proxy template ids (dHash groups)."""
    clusters = {}
    for c, t in zip(cluster_labels, template_ids):
        if c == -1:
            continue
        clusters.setdefault(c, []).append(t)

    if not clusters:
        return {"purity": 0.0, "recall": 0.0, "n_clustered": 0}

    purities = []
    for members in clusters.values():
        if not members:
            continue
        counts: dict[Any, int] = {}
        for m in members:
            counts[m] = counts.get(m, 0) + 1
        purities.append(max(counts.values()) / len(members))

    templates = set(template_ids)
    recalled = set()
    for members in clusters.values():
        counts: dict[Any, int] = {}
        for m in members:
            counts[m] = counts.get(m, 0) + 1
        dominant = max(counts, key=counts.get)
        recalled.add(dominant)

    return {
        "purity": float(np.mean(purities)),
        "recall": len(recalled) / len(templates) if templates else 0.0,
        "n_clustered": int(np.sum(cluster_labels != -1)),
    }


def assign_template_ids(sample_items: list[dict[str, Any]], max_hamming: int = 10) -> np.ndarray:
    from experiments.wp01.dhash import hamming

    ids = [it["id"] for it in sample_items]
    hashes = [int(it.get("dhash", 0)) for it in sample_items]
    template = list(range(len(ids)))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if hamming(hashes[i], hashes[j]) <= max_hamming:
                template[j] = min(template[i], template[j])
    # Relabel to contiguous ids
    mapping = {}
    out = []
    for t in template:
        if t not in mapping:
            mapping[t] = len(mapping)
        out.append(mapping[t])
    return np.array(out, dtype=np.int64)


def silhouette_if_applicable(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels != -1
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return None
    try:
        return float(metrics.silhouette_score(embeddings[mask], labels[mask], metric="cosine"))
    except Exception:
        return None
