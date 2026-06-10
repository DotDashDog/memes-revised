#!/usr/bin/env python3
"""Benchmark OCR variants against WP-01 manual transcripts."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from experiments.wp01.config import WP01_ROOT
from experiments.wp01.schema import load_json, save_json
from memes_base.ocr_pipelines import (
    PADDLEOCR_AVAILABLE,
    EASYOCR_AVAILABLE,
    TESSERACT_AVAILABLE,
    run_pipeline,
)


DEFAULT_VARIANTS = [
    "easyocr:none",
    "easyocr_filtered:none",
    "easyocr:contrast",
    "easyocr:white_text",
    "tesseract:outline",
    "paddleocr:none",
]


def normalize_for_scoring(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize_for_scoring(text).split()


def edit_distance(a: list[Any], b: list[Any]) -> int:
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_error_rate(pred: str, truth: str) -> float:
    p = list(normalize_for_scoring(pred))
    t = list(normalize_for_scoring(truth))
    if not t:
        return 0.0 if not p else 1.0
    return edit_distance(p, t) / len(t)


def word_error_rate(pred: str, truth: str) -> float:
    p = tokens(pred)
    t = tokens(truth)
    if not t:
        return 0.0 if not p else 1.0
    return edit_distance(p, t) / len(t)


def token_f1(pred: str, truth: str) -> float:
    p = Counter(tokens(pred))
    t = Counter(tokens(truth))
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    overlap = sum((p & t).values())
    precision = overlap / sum(p.values())
    recall = overlap / sum(t.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def backend_available(pipeline: str) -> bool:
    if pipeline in {"easyocr", "craft", "easyocr_filtered", "easyocr_boxed", "easyocr_boxed_filtered"}:
        return EASYOCR_AVAILABLE
    if pipeline in {"tesseract", "tess"}:
        return TESSERACT_AVAILABLE
    if pipeline in {"paddleocr", "paddle", "ppocrv5", "dbnet", "paddle_dbnet"}:
        return PADDLEOCR_AVAILABLE
    return False


def parse_variant(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return spec, "none"
    pipeline, preprocess = spec.split(":", 1)
    return pipeline.strip().lower(), preprocess.strip().lower()


def load_items(wp01_root: Path) -> list[dict[str, Any]]:
    sample = load_json(wp01_root / "sample.json")
    transcripts = load_json(wp01_root / "transcripts.json")
    sample_by_id = {entry["id"]: entry for entry in sample["items"]}
    items = []
    for item_id, transcript in transcripts["items"].items():
        image_path = wp01_root / "images" / f"{item_id}.jpg"
        if not image_path.exists():
            continue
        entry = sample_by_id.get(item_id, {})
        items.append(
            {
                **entry,
                "id": item_id,
                "stratum": entry.get("stratum", "manual_transcript"),
                "truth": transcript["in_image_text"],
                "image_path": image_path,
            }
        )
    return items


def run_variant(
    items: list[dict[str, Any]],
    *,
    pipeline: str,
    preprocess: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not backend_available(pipeline):
        return [], {
            "pipeline": pipeline,
            "preprocess": preprocess,
            "available": False,
            "error": "backend unavailable",
        }

    rows = []
    t0 = time.perf_counter()
    for item in items:
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            rows.append(
                {
                    "id": item["id"],
                    "pipeline": pipeline,
                    "preprocess": preprocess,
                    "truth": item["truth"],
                    "prediction": "",
                    "error": "image load failed",
                    "seconds": 0.0,
                }
            )
            continue

        start = time.perf_counter()
        try:
            result = run_pipeline(
                image,
                pipeline_name=pipeline,
                apply_preprocess=preprocess != "none",
                preprocess_mode=preprocess,
            )
            pred = result.text
            error = ""
        except Exception as exc:
            pred = ""
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start

        rows.append(
            {
                "id": item["id"],
                "stratum": item.get("stratum", "unknown"),
                "pipeline": pipeline,
                "preprocess": preprocess,
                "truth": item["truth"],
                "prediction": pred,
                "confidence": result.confidence if "result" in locals() and not error else None,
                "char_error_rate": char_error_rate(pred, item["truth"]),
                "word_error_rate": word_error_rate(pred, item["truth"]),
                "token_f1": token_f1(pred, item["truth"]),
                "truth_chars": len(normalize_for_scoring(item["truth"])),
                "pred_chars": len(normalize_for_scoring(pred)),
                "empty_prediction": not bool(normalize_for_scoring(pred)),
                "seconds": seconds,
                "error": error,
            }
        )

    elapsed = time.perf_counter() - t0
    return rows, {
        "pipeline": pipeline,
        "preprocess": preprocess,
        "available": True,
        "elapsed_seconds": elapsed,
    }


def summarize(df: pd.DataFrame, unavailable: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    if not df.empty:
        grouped = df.groupby(["pipeline", "preprocess"], dropna=False)
        for (pipeline, preprocess), group in grouped:
            ok = group[group["error"] == ""]
            summaries.append(
                {
                    "pipeline": pipeline,
                    "preprocess": preprocess,
                    "n": int(len(group)),
                    "n_errors": int((group["error"] != "").sum()),
                    "mean_cer": float(ok["char_error_rate"].mean()) if not ok.empty else None,
                    "median_cer": float(ok["char_error_rate"].median()) if not ok.empty else None,
                    "mean_wer": float(ok["word_error_rate"].mean()) if not ok.empty else None,
                    "mean_token_f1": float(ok["token_f1"].mean()) if not ok.empty else None,
                    "empty_rate": float(ok["empty_prediction"].mean()) if not ok.empty else None,
                    "mean_seconds": float(ok["seconds"].mean()) if not ok.empty else None,
                }
            )

    summaries = sorted(
        summaries,
        key=lambda row: (
            float("inf") if row["mean_cer"] is None else row["mean_cer"],
            -(row["mean_token_f1"] or 0.0),
        ),
    )
    return {
        "n_items": int(df["id"].nunique()) if not df.empty else 0,
        "variants": summaries,
        "unavailable": unavailable,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# WP-01 OCR Transcript Similarity Benchmark",
        "",
        f"Manual transcript items: {summary['n_items']}",
        "",
        "Lower CER/WER is better; higher token F1 is better.",
        "",
        "| Rank | Variant | CER | WER | Token F1 | Empty | sec/img | Errors |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(summary["variants"], start=1):
        variant = f"{row['pipeline']}:{row['preprocess']}"
        lines.append(
            f"| {idx} | `{variant}` | "
            f"{row['mean_cer']:.3f} | {row['mean_wer']:.3f} | {row['mean_token_f1']:.3f} | "
            f"{row['empty_rate']:.2f} | {row['mean_seconds']:.2f} | {row['n_errors']} |"
        )
    if summary["unavailable"]:
        lines.extend(["", "## Unavailable Variants", ""])
        for row in summary["unavailable"]:
            lines.append(f"- `{row['pipeline']}:{row['preprocess']}`: {row['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR variants against manual WP-01 transcripts.")
    parser.add_argument("--wp01-root", type=Path, default=WP01_ROOT)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    args = parser.parse_args()

    items = load_items(args.wp01_root)
    variant_specs = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    unavailable = []
    for spec in variant_specs:
        pipeline, preprocess = parse_variant(spec)
        print(f"Running OCR variant {pipeline}:{preprocess}", flush=True)
        variant_rows, meta = run_variant(items, pipeline=pipeline, preprocess=preprocess)
        if not meta["available"]:
            unavailable.append(meta)
        rows.extend(variant_rows)

    out_dir = args.wp01_root / "ocr-benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "ocr_outputs.csv", index=False)

    summary = summarize(df, unavailable)
    save_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "report.md", summary)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
