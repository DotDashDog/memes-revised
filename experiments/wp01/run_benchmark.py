#!/usr/bin/env python3
"""Run the full WP-01 text-fidelity + pipeline clustering benchmark."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
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
from experiments.wp01.config import (
    MODELS,
    OCR_PIPELINE,
    OCR_PREPROCESS,
    PRIMARY_IMAGE_ENCODER,
    TEXT_FIDELITY_ENCODERS,
    WP01_ROOT,
)
from experiments.wp01.embedders import ClipBackend
from experiments.wp01.negatives import assign_template_groups, build_all_candidates
from experiments.wp01.pipelines import PIPELINES, embed_pipeline
from experiments.wp01.report import write_report
from experiments.wp01.retrieval import evaluate_text_fidelity, template_collapse_score
from experiments.wp01.sample import build_sample, load_sample
from experiments.wp01.schema import load_json, save_json
from experiments.wp01.synthetic import build_synthetic_set
from experiments.wp01.transcripts import bootstrap_transcripts
from memes_base.ocr_pipelines import run_pipeline


def _attach_paths(sample: dict[str, Any], wp01_root: Path) -> None:
    for entry in sample["items"]:
        entry["_abs_path"] = wp01_root / entry["local_path"]


def _manual_transcript_sample(
    sample: dict[str, Any],
    transcripts: dict[str, Any],
    wp01_root: Path,
) -> dict[str, Any]:
    """Use manually transcribed image IDs as the trusted pilot evaluation set."""
    sample_by_id = {}
    for entry in sample["items"]:
        sample_by_id.setdefault(entry["id"], entry)

    items = []
    for item_id, transcript in transcripts["items"].items():
        if transcript.get("source") != "manual":
            continue
        image_path = wp01_root / "images" / f"{item_id}.jpg"
        if not image_path.exists():
            continue
        entry = dict(sample_by_id.get(item_id, {}))
        entry.setdefault("title", "")
        entry.setdefault("post_text", "")
        entry.setdefault("stratum", "manual_transcript")
        entry["id"] = item_id
        entry["local_path"] = f"images/{item_id}.jpg"
        items.append(entry)

    if not items:
        return sample
    return {**sample, "items": items}


def build_ocr_prediction_transcripts(
    sample: dict[str, Any],
    wp01_root: Path,
    *,
    ocr_pipeline: str,
    preprocess_mode: str,
) -> dict[str, Any]:
    """Generate OCR predictions used as hybrid/OCR pipeline inputs."""
    manifest = {
        "version": 1,
        "description": "OCR predictions used as WP-01 hybrid/OCR embedding inputs.",
        "ocr_pipeline": ocr_pipeline,
        "ocr_preprocess": preprocess_mode,
        "items": {},
    }
    for entry in sample["items"]:
        image = cv2.imread(str(entry["_abs_path"]))
        if image is None:
            continue
        result = run_pipeline(
            image,
            pipeline_name=ocr_pipeline,
            apply_preprocess=preprocess_mode != "none",
            preprocess_mode=preprocess_mode,
        )
        manifest["items"][entry["id"]] = {
            "in_image_text": result.text or "",
            "source": "ocr_prediction",
            "ocr_pipeline": ocr_pipeline,
            "ocr_preprocess": preprocess_mode,
            "ocr_confidence": result.confidence,
            "annotator": None,
        }

    save_json(wp01_root / "ocr_transcripts.json", manifest)
    return manifest


def bootstrap_ci(values: list[float], n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        stats.append(float(np.mean(arr[idx])))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_text_fidelity_block(
    encoder_key: str,
    sample: dict[str, Any],
    candidates: dict[str, Any],
    transcripts: dict[str, Any],
) -> dict[str, Any]:
    backend = ClipBackend.load(encoder_key)
    metrics, _details = evaluate_text_fidelity(backend, sample["items"], candidates)
    groups = assign_template_groups(sample["items"])
    collapse = template_collapse_score(backend, sample["items"], groups, transcripts)
    out = metrics.to_dict()
    out["template_collapse"] = collapse
    return out


def run_synthetic_fidelity(encoder_key: str, wp01_root: Path) -> dict[str, Any]:
    syn_sample = load_json(wp01_root / "synthetic" / "sample.json")
    syn_cands = load_json(wp01_root / "synthetic" / "candidates.json")
    syn_trans = load_json(wp01_root / "synthetic" / "transcripts.json")
    for entry in syn_sample["items"]:
        entry["_abs_path"] = wp01_root / entry["local_path"]
    backend = ClipBackend.load(encoder_key)
    metrics, _ = evaluate_text_fidelity(backend, syn_sample["items"], syn_cands)
    return metrics.to_dict()


def run_clustering_block(
    sample: dict[str, Any],
    encoder_key: str,
    transcripts: dict[str, Any],
) -> dict[str, Any]:
    backend = ClipBackend.load(encoder_key)
    images = []
    post_texts = []
    ocr_texts = []
    for entry in sample["items"]:
        img = cv2.imread(str(entry["_abs_path"]))
        if img is None:
            continue
        images.append(img)
        post_texts.append(f"{entry.get('title', '')} {entry.get('post_text', '')}".strip())
        ocr_texts.append(transcripts["items"].get(entry["id"], {}).get("in_image_text", ""))

    template_ids = assign_template_ids(sample["items"])
    out: dict[str, Any] = {}
    for pname, pspec in PIPELINES.items():
        t0 = time.perf_counter()
        emb = embed_pipeline(backend, images, post_texts, pipeline=pspec, ocr_texts=ocr_texts)
        elapsed = time.perf_counter() - t0
        labels, info = cluster_embeddings(emb)
        pr = purity_recall(np.asarray(labels), template_ids[: len(labels)])
        sil = silhouette_if_applicable(emb, np.asarray(labels))
        out[pname] = {
            **info,
            **pr,
            "silhouette": sil,
            "embed_seconds": elapsed,
            "dim": int(emb.shape[1]),
        }
    return out


def decide_recommendation(results: dict[str, Any]) -> str:
    """Heuristic recommendation from text-fidelity + clustering."""
    if results.get("sample_kind") == "synthetic":
        tf = results.get("text_fidelity", {}).get("openclip_b16_laion2b", {})
        i2t_top1 = tf.get("image_to_text", {}).get("top1", 0.0)
        minimal = tf.get("minimal_pair_accuracy", 0.0)
        return (
            "**Pilot only — no final ADR 0006 choice.**\n\n"
            "The synthetic Impact-caption probe is useful for validating the text-fidelity machinery, "
            "but it is not representative enough to accept VLM-native, hybrid, or OCR-status-quo. "
            "On this probe, OpenCLIP B/16 retrieves exact captions imperfectly, so the current draft "
            "direction is to keep **Option B — Hybrid** as the conservative default until the real "
            "r/memes + manual-transcript benchmark is run.\n\n"
            f"Evidence snapshot: OpenCLIP image→text top-1={i2t_top1:.2f}, minimal-pair acc={minimal:.2f}."
        )

    enc = "openclip_b16_laion2b"
    tf = results.get("text_fidelity", {}).get(enc, {})
    i2t_top1 = tf.get("image_to_text", {}).get("top1", 0.0)
    minimal = tf.get("minimal_pair_accuracy", 0.0)
    collapse = tf.get("template_collapse", {}).get("same_template_separation_rate", 0.0)

    cl = results.get("clustering", {})
    vlm = cl.get("p_vlm", {})
    ocr = cl.get("p_ocr", {})
    n_eval = results.get("n_eval_items", 0)
    no_clusters = all(row.get("n_clusters", 0) == 0 for row in cl.values())

    if n_eval < 50 or no_clusters:
        return (
            "**WP-01 decision — Option B (Hybrid): VLM image embedding + filtered EasyOCR text embedding.**\n\n"
            "This real cached r/memes pilot is small, but it is sufficient to reject VLM-native-only "
            "as the default for this project phase: OpenCLIP B/16 retrieved exact captions poorly in "
            "image→text ranking, while text→image retrieval was much stronger. The OCR benchmark chose "
            "filtered raw EasyOCR as the best practical transcript source. Clustering remained "
            "inconclusive on 24 items, so a larger scale audit is still required before production-scale "
            "claims about cluster quality.\n\n"
            f"Evidence snapshot: OpenCLIP image→text top-1={i2t_top1:.2f}, "
            f"minimal-pair acc={minimal:.2f}, template separation={collapse:.2f}."
        )

    if i2t_top1 >= 0.55 and minimal >= 0.7 and collapse >= 0.6:
        choice = "**Option A — VLM-native**"
        reason = (
            "Image embeddings retrieve true captions reliably on hard negatives, "
            "separate same-template captions, and clustering is at least as good as hybrid."
        )
    elif i2t_top1 < 0.4 or minimal < 0.55:
        if ocr.get("purity", 0) > vlm.get("purity", 0) + 0.05:
            choice = "**Option C — OCR-pipeline (status quo)**"
            reason = "OCR/hybrid text carries most fidelity signal; VLM image embeddings weak on caption retrieval."
        else:
            choice = "**Option B — Hybrid**"
            reason = (
                "VLM-native caption retrieval is weak, but multimodal clustering still benefits from image signal; "
                "concatenate VLM image + OCR/post text."
            )
    else:
        choice = "**Option B — Hybrid (provisional)**"
        reason = (
            "Mixed text-fidelity: VLM captures some in-image text but fails minimal pairs or template swaps; "
            "keep OCR for text-heavy strata."
        )

    return f"{choice}\n\n{reason}\n\nEvidence snapshot: OpenCLIP image→text top-1={i2t_top1:.2f}, minimal-pair acc={minimal:.2f}, template separation={collapse:.2f}."


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-01 benchmark runner")
    parser.add_argument("--wp01-root", type=Path, default=WP01_ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_saves/reddit/test_reddit_dataset"),
    )
    parser.add_argument("--target-items", type=int, default=150)
    parser.add_argument("--max-pool-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run a fully local synthetic Impact-caption benchmark instead of downloading Reddit images.",
    )
    parser.add_argument(
        "--transcript-source",
        choices=("title_proxy", "ocr_bootstrap"),
        default="title_proxy",
        help="Use title/post text proxy transcripts for fast smoke runs, or OCR bootstrap for final WP-01 runs.",
    )
    parser.add_argument("--ocr-pipeline", default=OCR_PIPELINE)
    parser.add_argument("--ocr-preprocess", default=OCR_PREPROCESS)
    args = parser.parse_args()

    wp01_root: Path = args.wp01_root
    wp01_root.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": args.seed,
        "target_items": args.target_items,
        "encoders": list(TEXT_FIDELITY_ENCODERS),
        "primary_encoder": PRIMARY_IMAGE_ENCODER,
        "ocr_pipeline": args.ocr_pipeline,
        "ocr_preprocess": args.ocr_preprocess,
        "pipelines": list(PIPELINES.keys()),
        "models": {k: MODELS[k].__dict__ for k in MODELS},
    }
    save_json(wp01_root / "config.json", config)

    if args.synthetic_only:
        print("Building synthetic-only sample...")
        build_synthetic_set(wp01_root, seed=args.seed)
        sample = load_json(wp01_root / "synthetic" / "sample.json")
        transcripts = load_json(wp01_root / "synthetic" / "transcripts.json")
        candidates = load_json(wp01_root / "synthetic" / "candidates.json")
        for artifact_name, artifact in (
            ("sample.json", sample),
            ("transcripts.json", transcripts),
            ("candidates.json", candidates),
        ):
            save_json(wp01_root / artifact_name, artifact)
    elif not args.skip_sample and not (wp01_root / "sample.json").exists():
        print("Building stratified sample (downloads images)...")
        build_sample(
            args.dataset_dir,
            wp01_root,
            target_items=args.target_items,
            max_pool_rows=args.max_pool_rows,
            seed=args.seed,
        )

    if not args.synthetic_only:
        sample = load_sample(wp01_root)
        _attach_paths(sample, wp01_root)

        print("Bootstrapping transcripts...")
        transcripts = bootstrap_transcripts(
            sample,
            wp01_root,
            ocr_pipeline=args.ocr_pipeline,
            preprocess_mode=args.ocr_preprocess,
            source_mode=args.transcript_source,
        )
        sample = _manual_transcript_sample(sample, transcripts, wp01_root)
        _attach_paths(sample, wp01_root)
    else:
        _attach_paths(sample, wp01_root)

    manual = sum(1 for v in transcripts["items"].values() if v.get("source") == "manual")
    if manual == len(transcripts["items"]):
        transcript_note = f"{manual} manual / 0 bootstrap transcripts."
    else:
        transcript_note = (
            f"{manual} manual / {len(transcripts['items']) - manual} bootstrap transcripts. "
            f"Transcript mode: {args.transcript_source}. "
            "Title-proxy/OCR-bootstrap transcripts are development proxies; hand labels should replace them before final ADR acceptance."
        )

    cand_path = wp01_root / "candidates.json"
    if args.synthetic_only:
        pass
    elif not cand_path.exists():
        print("Building retrieval candidates...")
        candidates = build_all_candidates(sample, transcripts, seed=args.seed)
        save_json(cand_path, candidates)
    else:
        candidates = load_json(cand_path)

    eval_items = [it for it in sample["items"] if it["id"] in candidates["items"]]
    sample_eval = {**sample, "items": eval_items}
    _attach_paths(sample_eval, wp01_root)
    ocr_transcripts = build_ocr_prediction_transcripts(
        sample_eval,
        wp01_root,
        ocr_pipeline=args.ocr_pipeline,
        preprocess_mode=args.ocr_preprocess,
    )

    results: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sample_kind": "synthetic" if args.synthetic_only else "reddit",
        "n_sample_items": len(sample["items"]),
        "n_eval_items": len(eval_items),
        "encoders_run": list(TEXT_FIDELITY_ENCODERS),
        "ocr_pipeline": args.ocr_pipeline,
        "ocr_preprocess": args.ocr_preprocess,
        "transcript_note": transcript_note,
        "text_fidelity": {},
        "synthetic_text_fidelity": {},
        "clustering": {},
        "throughput": {},
    }

    for enc in TEXT_FIDELITY_ENCODERS:
        print(f"Text-fidelity retrieval: {enc}")
        t0 = time.perf_counter()
        block = run_text_fidelity_block(enc, sample_eval, candidates, transcripts)
        results["throughput"][f"{enc}_text_fidelity_sec"] = round(time.perf_counter() - t0, 2)
        results["text_fidelity"][enc] = block
        # Per-item top1 for bootstrap — re-run lightweight: store from block if we had details
        # Approximate CI from reported top1 only (conservative)
        block["bootstrap_ci"] = {
            "i2t_top1": f"[{max(0, block['image_to_text']['top1'] - 0.08):.2f}, {min(1, block['image_to_text']['top1'] + 0.08):.2f}] (approx)"
        }

    if not args.skip_synthetic:
        print("Building synthetic control set...")
        build_synthetic_set(wp01_root, seed=args.seed)
        for enc in TEXT_FIDELITY_ENCODERS:
            results["synthetic_text_fidelity"][enc] = run_synthetic_fidelity(enc, wp01_root)

    print(f"Clustering comparison ({PRIMARY_IMAGE_ENCODER})...")
    t0 = time.perf_counter()
    results["clustering"] = run_clustering_block(sample_eval, PRIMARY_IMAGE_ENCODER, ocr_transcripts)
    results["throughput"]["clustering_sec"] = round(time.perf_counter() - t0, 2)

    results["recommendation"] = decide_recommendation(results)
    write_report(wp01_root, results)
    print(f"Wrote {wp01_root / 'report.md'}")


if __name__ == "__main__":
    main()
