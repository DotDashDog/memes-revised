#!/usr/bin/env python3
"""Create an HTML gallery comparing OCR variants on WP-01 images."""

from __future__ import annotations

import argparse
import html
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.wp01.benchmark_ocr import DEFAULT_VARIANTS, backend_available, parse_variant
from experiments.wp01.config import WP01_ROOT
from experiments.wp01.schema import load_json, save_json
from memes_base.ocr_pipelines import preprocess_image, run_pipeline


def load_transcript_items(wp01_root: Path, limit: int) -> list[dict[str, Any]]:
    transcripts = load_json(wp01_root / "transcripts.json")
    items = []
    for item_id, transcript in transcripts["items"].items():
        image_path = wp01_root / "images" / f"{item_id}.jpg"
        if not image_path.exists():
            continue
        items.append(
            {
                "id": item_id,
                "image_path": image_path,
                "truth": transcript.get("in_image_text", ""),
            }
        )
        if len(items) >= limit:
            break
    return items


def write_image(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path.name


def render_text(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def draw_boxes(image, boxes):
    annotated = image.copy()
    for box in boxes:
        points = np.array(box.points, dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        x_min = int(points[:, 0].min())
        y_min = int(points[:, 1].min())
        label = box.text[:28]
        if box.confidence is not None:
            label = f"{box.confidence:.2f} {label}"
        cv2.putText(
            annotated,
            label,
            (x_min, max(12, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_visualization(
    *,
    wp01_root: Path,
    output_dir: Path,
    n_images: int,
    variant_specs: list[str],
) -> dict[str, Any]:
    items = load_transcript_items(wp01_root, n_images)
    image_dir = output_dir / "images"
    rows = []

    for item in items:
        print(f"Visualizing {item['id']}", flush=True)
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            rows.append({"id": item["id"], "error": "image load failed", "variants": []})
            continue

        original_name = write_image(image_dir / f"{item['id']}-original.jpg", image)
        variants = []
        for spec in variant_specs:
            pipeline, preprocess = parse_variant(spec)
            variant_key = f"{pipeline}:{preprocess}"
            if not backend_available(pipeline):
                variants.append(
                    {
                        "variant": variant_key,
                        "available": False,
                        "error": "backend unavailable",
                    }
                )
                continue

            try:
                input_image = image if preprocess == "none" else preprocess_image(image, preprocess)
                safe_key = variant_key.replace(":", "-")
                preprocessed_name = write_image(
                    image_dir / f"{item['id']}-{safe_key}.png",
                    input_image,
                )
                start = time.perf_counter()
                result = run_pipeline(
                    image,
                    pipeline_name=pipeline,
                    apply_preprocess=preprocess != "none",
                    preprocess_mode=preprocess,
                )
                seconds = time.perf_counter() - start
                annotated_name = None
                if result.boxes:
                    annotated_name = write_image(
                        image_dir / f"{item['id']}-{safe_key}-boxes.png",
                        draw_boxes(input_image, result.boxes),
                    )
                variants.append(
                    {
                        "variant": variant_key,
                        "available": True,
                        "preprocessed_image": preprocessed_name,
                        "annotated_image": annotated_name,
                        "text": result.text,
                        "confidence": result.confidence,
                        "seconds": seconds,
                        "error": "",
                    }
                )
            except Exception as exc:
                variants.append(
                    {
                        "variant": variant_key,
                        "available": True,
                        "text": "",
                        "confidence": None,
                        "seconds": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        rows.append(
            {
                "id": item["id"],
                "truth": item["truth"],
                "original_image": original_name,
                "variants": variants,
            }
        )

    payload = {
        "description": "Visual OCR comparison for WP-01 manual transcript images.",
        "n_images": len(rows),
        "variants": variant_specs,
        "items": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "results.json", payload)
    write_html(output_dir / "index.html", payload)
    return payload


def write_html(path: Path, payload: dict[str, Any]) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>WP-01 OCR Visual Comparison</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#fafafa;color:#111}",
        ".item{margin:0 0 36px;padding:18px;background:white;border:1px solid #ddd;border-radius:10px}",
        ".truth{padding:10px;background:#f2f6ff;border-left:4px solid #5b7cfa;margin:12px 0;line-height:1.4}",
        ".original{max-width:420px;max-height:360px;border:1px solid #ccc}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-top:12px}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fff}",
        ".card img{max-width:100%;max-height:220px;display:block;margin-bottom:8px;border:1px solid #ddd}",
        ".meta{font-size:12px;color:#555;margin-bottom:8px}",
        "pre{white-space:pre-wrap;font-size:13px;line-height:1.35;background:#f7f7f7;padding:8px;border-radius:6px;min-height:70px}",
        ".error{color:#a40000}",
        "</style></head><body>",
        "<h1>WP-01 OCR Visual Comparison</h1>",
        f"<p>{payload['n_images']} images, {len(payload['variants'])} OCR variants requested.</p>",
    ]

    for item in payload["items"]:
        parts.extend(
            [
                "<section class='item'>",
                f"<h2>{html.escape(item['id'])}</h2>",
                f"<img class='original' src='images/{html.escape(item['original_image'])}' alt='original'>",
                f"<div class='truth'><strong>Manual transcript:</strong><br>{render_text(item.get('truth', ''))}</div>",
                "<div class='grid'>",
            ]
        )
        for variant in item["variants"]:
            parts.append("<div class='card'>")
            parts.append(f"<h3>{html.escape(variant['variant'])}</h3>")
            if not variant.get("available", True):
                parts.append(f"<p class='error'>{html.escape(variant.get('error', 'unavailable'))}</p>")
            else:
                if variant.get("preprocessed_image"):
                    parts.append(
                        f"<img src='images/{html.escape(variant['preprocessed_image'])}' "
                        f"alt='{html.escape(variant['variant'])}'>"
                    )
                if variant.get("annotated_image"):
                    parts.append(
                        f"<img src='images/{html.escape(variant['annotated_image'])}' "
                        f"alt='{html.escape(variant['variant'])} boxes'>"
                    )
                meta = []
                if variant.get("seconds") is not None:
                    meta.append(f"{variant['seconds']:.2f}s")
                if variant.get("confidence") is not None:
                    meta.append(f"conf {variant['confidence']:.3f}")
                if variant.get("error"):
                    meta.append(html.escape(variant["error"]))
                parts.append(f"<div class='meta'>{' | '.join(meta)}</div>")
                parts.append(f"<pre>{render_text(variant.get('text', ''))}</pre>")
            parts.append("</div>")
        parts.extend(["</div>", "</section>"])

    parts.append("</body></html>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual OCR comparison gallery.")
    parser.add_argument("--wp01-root", type=Path, default=WP01_ROOT)
    parser.add_argument("--output-dir", type=Path, default=WP01_ROOT / "ocr-visual")
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    args = parser.parse_args()

    variant_specs = [v.strip() for v in args.variants.split(",") if v.strip()]
    payload = run_visualization(
        wp01_root=args.wp01_root,
        output_dir=args.output_dir,
        n_images=args.n_images,
        variant_specs=variant_specs,
    )
    print(f"Wrote {args.output_dir / 'index.html'} for {payload['n_images']} images")


if __name__ == "__main__":
    main()
