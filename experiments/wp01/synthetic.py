"""Synthetic Impact-style captions for mechanistic text-fidelity probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.wp01.negatives import build_candidates_for_item
from experiments.wp01.schema import save_json


CAPTIONS = [
    "WHEN YOU FINALLY FIX THE BUG",
    "ME TRYING TO UNDERSTAND THE LECTURE",
    "NOBODY: ABSOLUTELY NOBODY:",
    "TEACHERS BE LIKE",
    "I SHOULD GO TO BED",
    "ONE DOES NOT SIMPLY",
    "STONKS",
    "BASED",
]


def _render_impact(text: str, width: int = 512, height: int = 384) -> np.ndarray:
    img = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Impact.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
        draw.multiline_text((x + dx, y + dy), text, font=font, fill="black", align="center")
    draw.multiline_text((x, y), text, font=font, fill="white", align="center")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def build_synthetic_set(output_dir: Path, seed: int = 42) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    syn_dir = output_dir / "synthetic" / "images"
    syn_dir.mkdir(parents=True, exist_ok=True)

    items = []
    all_texts: dict[str, str] = {}
    for i, caption in enumerate(CAPTIONS):
        item_id = f"syn-{i+1:03d}"
        image = _render_impact(caption)
        path = syn_dir / f"{item_id}.jpg"
        cv2.imwrite(str(path), image)
        items.append(
            {
                "id": item_id,
                "local_path": f"synthetic/images/{item_id}.jpg",
                "stratum": "synthetic",
                "dhash": 0,
            }
        )
        all_texts[item_id] = caption

    template_groups = {iid: list(all_texts.keys()) for iid in all_texts}
    cand_items = {}
    for item_id, text in all_texts.items():
        cand_items[item_id] = {
            "true_text": text,
            "candidates": build_candidates_for_item(
                item_id, text, all_texts, template_groups, rng
            ),
        }

    manifest = {
        "version": 1,
        "seed": seed,
        "items": items,
        "n_items": len(items),
    }
    save_json(output_dir / "synthetic" / "sample.json", manifest)
    save_json(
        output_dir / "synthetic" / "candidates.json",
        {"version": 1, "seed": seed, "items": cand_items, "n_items": len(cand_items)},
    )
    transcripts = {
        "version": 1,
        "items": {iid: {"in_image_text": all_texts[iid], "source": "synthetic"} for iid in all_texts},
    }
    save_json(output_dir / "synthetic" / "transcripts.json", transcripts)
    return manifest
