"""Prepare blind labelling batches for stage 1 of the labelling protocol.

Writes downscaled JPEG copies (longest side <= 768 px, first frame of GIFs)
to <sample>/label_images/{post_id}.jpg and a seeded-shuffle batch manifest
<sample>/stage1/batches.json. A seeded 10% is repeated in extra batches for
the self-consistency check. Labellers are given image paths only.

Usage: uv run python -m experiments.wp04.prep_label_batches --sample experiments/WP-04/sample-001
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

ORDER_SEED = 42
RELABEL_SEED = 43
BATCH = 40
MAX_SIDE = 768


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    args = ap.parse_args()
    d = Path(args.sample)
    src = {p.stem: p for p in (d / "images").iterdir()}
    ids = pd.read_parquet(d / "sample.parquet").post_id.tolist()
    missing = [i for i in ids if i not in src]
    out = d / "label_images"
    out.mkdir(exist_ok=True)
    bad = []
    for pid in ids:
        if pid in missing:
            continue
        try:
            im = Image.open(src[pid])
            im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(out / f"{pid}.jpg", quality=88)
        except Exception as e:  # noqa: BLE001
            bad.append((pid, repr(e)))
    ok = sorted(p.stem for p in out.glob("*.jpg"))
    rng = random.Random(ORDER_SEED)
    order = ok[:]
    rng.shuffle(order)
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    rel = random.Random(RELABEL_SEED).sample(ok, k=round(0.10 * len(ok)))
    rel_batches = [rel[i:i + BATCH] for i in range(0, len(rel), BATCH)]
    (d / "stage1").mkdir(exist_ok=True)
    (d / "stage1" / "batches.json").write_text(json.dumps(
        {"main": batches, "relabel": rel_batches, "missing": missing, "undecodable": bad},
        indent=1))
    print(len(ok), "images;", len(batches), "batches;", len(rel_batches),
          "relabel batches; missing", len(missing), "undecodable", len(bad))


if __name__ == "__main__":
    main()
