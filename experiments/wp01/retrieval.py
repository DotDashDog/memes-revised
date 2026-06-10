"""Text-fidelity retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.wp01.embedders import ClipBackend


@dataclass
class RetrievalMetrics:
    n_items: int
    image_to_text_top1: float
    image_to_text_top5: float
    image_to_text_mrr: float
    text_to_image_top1: float
    text_to_image_top5: float
    text_to_image_mrr: float
    minimal_pair_accuracy: float
    median_true_rank_pct: float
    mean_margin_hardest: float
    by_label: dict[str, dict[str, float]]
    by_stratum: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "image_to_text": {
                "top1": self.image_to_text_top1,
                "top5": self.image_to_text_top5,
                "mrr": self.image_to_text_mrr,
                "median_true_rank_pct": self.median_true_rank_pct,
                "mean_margin_hardest": self.mean_margin_hardest,
            },
            "text_to_image": {
                "top1": self.text_to_image_top1,
                "top5": self.text_to_image_top5,
                "mrr": self.text_to_image_mrr,
            },
            "minimal_pair_accuracy": self.minimal_pair_accuracy,
            "by_negative_label": self.by_label,
            "by_stratum": self.by_stratum,
        }


def _rank_scores(image_vec: np.ndarray, text_matrix: np.ndarray) -> np.ndarray:
    return text_matrix @ image_vec


def evaluate_text_fidelity(
    backend: ClipBackend,
    sample_items: list[dict[str, Any]],
    candidates_manifest: dict[str, Any],
    *,
    stratum_by_id: dict[str, str] | None = None,
) -> tuple[RetrievalMetrics, dict[str, Any]]:
    stratum_by_id = stratum_by_id or {it["id"]: it.get("stratum", "unknown") for it in sample_items}

    import cv2

    item_ids: list[str] = []
    images = []
    for entry in sample_items:
        iid = entry["id"]
        if iid not in candidates_manifest["items"]:
            continue
        path = entry.get("_abs_path")
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        item_ids.append(iid)
        images.append(img)

    if not item_ids:
        raise ValueError("No evaluable items with images and candidates")

    image_embeds = backend.encode_images(images)
    n = len(item_ids)

    i2t_top1 = []
    i2t_top5 = []
    i2t_rr = []
    t2i_top1 = []
    t2i_top5 = []
    t2i_rr = []
    minimal_pairs = []
    rank_pcts = []
    margins = []
    by_label: dict[str, list[float]] = {}
    by_stratum: dict[str, list[float]] = {s: [] for s in set(stratum_by_id.values())}

    per_item_details: dict[str, Any] = {}

    for idx, iid in enumerate(item_ids):
        cand_entry = candidates_manifest["items"][iid]
        cands = cand_entry["candidates"]
        texts = [c["text"] for c in cands]
        labels = [c["label"] for c in cands]
        text_embeds = backend.encode_texts(texts)

        scores = _rank_scores(image_embeds[idx], text_embeds)
        order = np.argsort(-scores)
        ranks = {orig: rank + 1 for rank, orig in enumerate(order)}
        true_idx = labels.index("true")
        true_rank = ranks[true_idx]
        i2t_top1.append(1.0 if true_rank == 1 else 0.0)
        i2t_top5.append(1.0 if true_rank <= 5 else 0.0)
        i2t_rr.append(1.0 / true_rank)
        rank_pcts.append(true_rank / len(texts))

        hardest = max(
            (scores[i], labels[i])
            for i in range(len(labels))
            if labels[i] != "true"
        )
        margins.append(float(scores[true_idx] - hardest[0]))

        for i, lab in enumerate(labels):
            if lab == "true":
                continue
            by_label.setdefault(lab, []).append(1.0 if scores[true_idx] > scores[i] else 0.0)

        by_stratum[stratum_by_id.get(iid, "unknown")].append(1.0 if true_rank == 1 else 0.0)

        minimal_labels = {"negation_flip", "number_swap", "one_word_swap"}
        for i, lab in enumerate(labels):
            if lab in minimal_labels:
                minimal_pairs.append(1.0 if scores[true_idx] > scores[i] else 0.0)

        per_item_details[iid] = {
            "true_rank": int(true_rank),
            "n_candidates": len(texts),
            "top_prediction_label": labels[order[0]],
        }

    # Text-to-image: each true text vs all images in batch
    true_texts = [candidates_manifest["items"][iid]["true_text"] for iid in item_ids]
    text_embeds_all = backend.encode_texts(true_texts)
    for idx in range(n):
        scores = image_embeds @ text_embeds_all[idx]
        order = np.argsort(-scores)
        true_rank = int(np.where(order == idx)[0][0]) + 1
        t2i_top1.append(1.0 if true_rank == 1 else 0.0)
        t2i_top5.append(1.0 if true_rank <= 5 else 0.0)
        t2i_rr.append(1.0 / true_rank)

    def _summarize_buckets(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        return {k: {"top1": float(np.mean(v)), "n": len(v)} for k, v in buckets.items() if v}

    metrics = RetrievalMetrics(
        n_items=n,
        image_to_text_top1=float(np.mean(i2t_top1)),
        image_to_text_top5=float(np.mean(i2t_top5)),
        image_to_text_mrr=float(np.mean(i2t_rr)),
        text_to_image_top1=float(np.mean(t2i_top1)),
        text_to_image_top5=float(np.mean(t2i_top5)),
        text_to_image_mrr=float(np.mean(t2i_rr)),
        minimal_pair_accuracy=float(np.mean(minimal_pairs)) if minimal_pairs else 0.0,
        median_true_rank_pct=float(np.median(rank_pcts)),
        mean_margin_hardest=float(np.mean(margins)),
        by_label=_summarize_buckets(by_label),
        by_stratum=_summarize_buckets(by_stratum),
    )
    return metrics, per_item_details


def template_collapse_score(
    backend: ClipBackend,
    sample_items: list[dict[str, Any]],
    template_groups: dict[str, list[str]],
    transcripts: dict[str, Any],
) -> dict[str, float]:
    """Fraction of same-template pairs separated more than random caption pairs."""
    import cv2

    id_to_image = {}
    for entry in sample_items:
        path = entry.get("_abs_path")
        if not path:
            continue
        img = __import__("cv2").imread(str(path))
        if img is not None:
            id_to_image[entry["id"]] = img

    pairs_same = 0
    correct_same = 0
    pairs_diff = 0
    correct_diff = 0

    ids = list(id_to_image.keys())
    if len(ids) < 2:
        return {"same_template_separation_rate": 0.0, "n_same_pairs": 0}

    image_embeds = {iid: backend.encode_images([id_to_image[iid]])[0] for iid in ids}
    text_embeds = {
        iid: backend.encode_texts([transcripts["items"][iid]["in_image_text"]])[0]
        for iid in ids
        if iid in transcripts["items"]
    }

    seen_pairs: set[tuple[str, str]] = set()
    for iid, group in template_groups.items():
        if iid not in image_embeds or iid not in text_embeds:
            continue
        for other in group:
            if other == iid or other not in image_embeds or other not in text_embeds:
                continue
            key = tuple(sorted((iid, other)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs_same += 1
            # Image i should be closer to its text than other caption
            sim_own = float(image_embeds[iid] @ text_embeds[iid])
            sim_cross = float(image_embeds[iid] @ text_embeds[other])
            if sim_own > sim_cross:
                correct_same += 1

    import random

    rng = random.Random(0)
    attempts = min(200, len(ids) * 5)
    for _ in range(attempts):
        a, b = rng.sample(ids, 2)
        if b in template_groups.get(a, []):
            continue
        if a not in text_embeds or b not in text_embeds:
            continue
        pairs_diff += 1
        sim_own = float(image_embeds[a] @ text_embeds[a])
        sim_cross = float(image_embeds[a] @ text_embeds[b])
        if sim_own > sim_cross:
            correct_diff += 1

    return {
        "same_template_separation_rate": correct_same / pairs_same if pairs_same else 0.0,
        "n_same_pairs": pairs_same,
        "diff_template_separation_rate": correct_diff / pairs_diff if pairs_diff else 0.0,
        "n_diff_pairs": pairs_diff,
    }
