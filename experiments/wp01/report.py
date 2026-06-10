"""Render experiments/WP-01/report.md from metrics JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.wp01.config import MODELS
from experiments.wp01.schema import load_json, save_json


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_report(wp01_root: Path, results: dict[str, Any]) -> str:
    lines = [
        "# WP-01 — OCR vs. VLM-native embedding benchmark",
        "",
        f"**Generated:** {results.get('generated_utc', 'n/a')}",
        f"**Sample size:** {results.get('n_sample_items', 'n/a')} memes",
        f"**Text-fidelity eval items:** {results.get('n_eval_items', 'n/a')}",
        "",
        "## Models",
        "",
    ]
    for key in results.get("encoders_run", []):
        spec = MODELS[key]
        lines.extend(
            [
                f"### `{key}`",
                f"- Family: {spec.family}",
                f"- Weights: `{spec.model_id}` / `{spec.pretrained}`",
                f"- Training pool: {spec.release_or_pool}",
                f"- Cutoff note: {spec.cutoff_note}",
                f"- Throughput note: {spec.throughput_note}",
                "",
            ]
        )

    lines.extend(
        [
            "## Text-fidelity retrieval (primary diagnostic)",
            "",
            "Does the image embedding rank the true in-image caption above hard negatives?",
            "",
        ]
    )
    for enc, metrics in results.get("text_fidelity", {}).items():
        i2t = metrics["image_to_text"]
        t2i = metrics["text_to_image"]
        lines.extend(
            [
                f"### Encoder: `{enc}`",
                f"- Image→text top-1: **{_fmt_pct(i2t['top1'])}** (95% CI {metrics.get('bootstrap_ci', {}).get('i2t_top1', 'n/a')})",
                f"- Image→text top-5: {_fmt_pct(i2t['top5'])}",
                f"- Image→text MRR: {i2t['mrr']:.3f}",
                f"- Text→image top-1: **{_fmt_pct(t2i['top1'])}**",
                f"- Minimal-pair accuracy: {_fmt_pct(metrics['minimal_pair_accuracy'])}",
                f"- Median true-text rank percentile: {i2t['median_true_rank_pct']:.3f} (lower is better)",
                f"- Mean margin vs hardest negative: {i2t['mean_margin_hardest']:.4f}",
                "",
                "**By stratum (image→text top-1):**",
            ]
        )
        for stratum, vals in metrics.get("by_stratum", {}).items():
            lines.append(f"- `{stratum}`: {_fmt_pct(vals['top1'])} (n={vals['n']})")
        lines.append("")
        lines.append("**By negative type (true beats negative rate):**")
        for lab, vals in metrics.get("by_negative_label", {}).items():
            lines.append(f"- `{lab}`: {_fmt_pct(vals['top1'])} (n={vals['n']})")
        lines.append("")
        collapse = metrics.get("template_collapse", {})
        lines.append(
            f"- Template-collapse (same-template caption separation): "
            f"{_fmt_pct(collapse.get('same_template_separation_rate', 0))} "
            f"({collapse.get('n_same_pairs', 0)} pairs)"
        )
        lines.append("")

    syn = results.get("synthetic_text_fidelity", {})
    if syn:
        lines.extend(["## Synthetic control (mechanistic)", ""])
        for enc, metrics in syn.items():
            i2t = metrics["image_to_text"]
            lines.append(
                f"- `{enc}` image→text top-1 on rendered Impact captions: **{_fmt_pct(i2t['top1'])}**"
            )
        lines.append("")

    lines.extend(
        [
            "## Pipeline clustering (downstream evidence)",
            "",
            "UMAP → HDBSCAN with common settings (see `experiments/wp01/config.py`). "
            "Template labels are dHash proxy groups, not hand-labeled gold templates.",
            "",
            "| Pipeline | Clusters | Noise frac | Purity | Recall | Silhouette |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in results.get("clustering", {}).items():
        sil = row.get("silhouette")
        sil_s = f"{sil:.3f}" if sil is not None else "n/a"
        lines.append(
            f"| `{name}` | {row['n_clusters']} | {row['noise_fraction']:.2f} | "
            f"{row['purity']:.3f} | {row['recall']:.3f} | {sil_s} |"
        )

    lines.extend(
        [
            "",
            "## Transcript limitation",
            "",
            results.get(
                "transcript_note",
                "Most transcripts are OCR-bootstrap unless manually overridden in `transcripts.json`.",
            ),
            "",
        ]
    )

    ocr_summary_path = wp01_root / "ocr-benchmark" / "summary.json"
    if ocr_summary_path.exists():
        ocr_summary = load_json(ocr_summary_path)
        lines.extend(
            [
                "## OCR transcript similarity",
                "",
                "Benchmarked against the same manual in-image transcripts. Results are in `experiments/WP-01/ocr-benchmark/`.",
                "",
                "| Rank | Variant | CER | WER | Token F1 | sec/img |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, row in enumerate(ocr_summary.get("variants", [])[:6], start=1):
            variant = f"{row['pipeline']}:{row['preprocess']}"
            lines.append(
                f"| {idx} | `{variant}` | {row['mean_cer']:.3f} | "
                f"{row['mean_wer']:.3f} | {row['mean_token_f1']:.3f} | "
                f"{row['mean_seconds']:.2f} |"
            )
        lines.extend(
            [
                "",
                "Current OCR default: `easyocr_filtered:none`, which keeps raw EasyOCR detections but filters low-confidence or punctuation-heavy boxes before embedding.",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation for ADR 0006",
            "",
            results.get("recommendation", "See draft update in `docs/decisions/0006-ocr-vs-vlm.md`."),
            "",
            "## Throughput (this run)",
            "",
        ]
    )
    for k, v in results.get("throughput", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def write_report(wp01_root: Path, results: dict[str, Any]) -> None:
    text = render_report(wp01_root, results)
    path = wp01_root / "report.md"
    path.write_text(text, encoding="utf-8")
    save_json(wp01_root / "results.json", results)
