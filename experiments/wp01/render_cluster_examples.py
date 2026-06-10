#!/usr/bin/env python3
"""Render clustered meme examples as an HTML gallery."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any

from experiments.wp01.schema import load_json


def _escape(text: Any) -> str:
    return html.escape(str(text or ""))


def _render_multiline(text: str) -> str:
    return _escape(text).replace("\n", "<br>")


def render_cluster_gallery(summary_path: Path, output_path: Path) -> None:
    summary = load_json(summary_path)
    cache_dir = Path(summary["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() / cache_dir).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Cluster Examples</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#fafafa;color:#111}",
        ".meta{color:#555;margin-bottom:20px}",
        ".cluster{margin:0 0 36px;padding:18px;background:white;border:1px solid #ddd;border-radius:10px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}",
        ".card{border:1px solid #ddd;border-radius:8px;background:#fff;padding:10px}",
        ".card img{display:block;max-width:100%;max-height:300px;margin:0 auto 10px;border:1px solid #ddd}",
        ".id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#444;font-size:12px}",
        ".field{margin:8px 0}",
        ".label{font-weight:600;color:#333}",
        ".text{background:#f7f7f7;border-radius:6px;padding:8px;line-height:1.35;font-size:13px}",
        "</style></head><body>",
        "<h1>Cluster Examples</h1>",
        "<div class='meta'>",
        f"Cache: <code>{_escape(summary.get('cache_dir'))}</code><br>",
        f"Text embedding: <code>{_escape(summary.get('text_key'))}</code><br>",
        f"Algorithm: <code>{_escape(summary.get('algorithm'))}</code><br>",
        f"Clusters: {_escape(summary.get('n_clusters'))}; silhouette: {_escape(summary.get('silhouette'))}",
        "</div>",
    ]

    for cluster in summary.get("examples", []):
        parts.extend(
            [
                "<section class='cluster'>",
                f"<h2>Cluster {_escape(cluster.get('cluster'))} (n={_escape(cluster.get('size'))})</h2>",
                "<div class='grid'>",
            ]
        )
        for item in cluster.get("items", []):
            image_path = cache_dir / item.get("image_path", "")
            image_src = os.path.relpath(image_path, output_path.parent)
            parts.extend(
                [
                    "<article class='card'>",
                    f"<img src='{_escape(image_src)}' alt='{_escape(item.get('id'))}'>",
                    f"<div class='id'>{_escape(item.get('id'))}</div>",
                    f"<div class='field'><span class='label'>Title:</span> {_escape(item.get('title') or '[no title]')}</div>",
                    f"<div class='field'><span class='label'>Post text:</span><div class='text'>{_render_multiline(item.get('post_text') or '[none]')}</div></div>",
                    f"<div class='field'><span class='label'>Embedded text:</span><div class='text'>{_render_multiline(item.get('text_input') or '[none]')}</div></div>",
                    "</article>",
                ]
            )
        parts.extend(["</div>", "</section>"])

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cluster examples as HTML.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "experiments/WP-01/semantic-embedding-1000-002/"
            "cluster_openai_text_embedding_3_small_kmeans/summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.summary.parent / "index.html")
    render_cluster_gallery(args.summary, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
