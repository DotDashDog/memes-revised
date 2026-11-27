"""WP-04 Method §5: extract sampled images, run OCR + embeddings on the 3090.

Three stages, all measured for the throughput note (deliverable f):

1. Extract the sample's images from the ADR 0015 tar shards to the NVMe
   working set (`<out>/images/{post_id}.{ext}`), reading in (shard, offset)
   order so the USB drive only ever seeks forward.
2. OCR every image with the ADR 0006 accepted config (easyocr_filtered,
   preprocess none), GPU reader, recognizer batch-size sweep on a probe
   subset, then the best batch size for the rest.
3. Embed the ADR 0017 clustering composition p_img_ocr (image ⊕ OCR) and the
   comparison arm p_vlm_only (image only) with openclip_b16_laion2b at
   several batch sizes.

Outputs under <out>/: images/, ocr.json (post_id -> text/confidence/seconds),
embeddings_{p_img_ocr,p_vlm_only}.npy + ids.json, timing.json.

Usage:
    uv run python -m experiments.wp04.ocr_embed --ids <sample.parquet> --out <dir>

The ids parquet needs a `post_id` column; membership in the shard index with
fetch_status == ok is required (others are reported and skipped).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

INDEX = Path("/mnt/d/memes/index/shard_index.parquet")
IMAGES_DIR = Path("/mnt/d/memes/images")

EXT_BY_MAGIC = [(b"\xff\xd8\xff", "jpg"), (b"\x89PNG", "png"), (b"GIF8", "gif"), (b"RIFF", "webp")]


def extract(ids: list[str], out_images: Path) -> tuple[dict[str, Path], dict]:
    idx = pq.read_table(INDEX).to_pandas()
    idx = idx[(idx.fetch_status == "ok") & idx.post_id.isin(ids)]
    missing = sorted(set(ids) - set(idx.post_id))
    out_images.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    t0 = time.monotonic()
    n_read = 0
    by_shard = idx.sort_values(["shard_id", "offset"]).groupby("shard_id", sort=True)
    for shard_id, grp in by_shard:
        with open(IMAGES_DIR / f"{shard_id}.tar", "rb") as f:
            for r in grp.itertuples():
                f.seek(int(r.offset))
                body = f.read(int(r.length))
                ext = next((e for m, e in EXT_BY_MAGIC if body.startswith(m)), "bin")
                p = out_images / f"{r.post_id}.{ext}"
                if not p.exists():
                    p.write_bytes(body)
                paths[r.post_id] = p
                n_read += 1
    stats = {
        "n_requested": len(ids),
        "n_extracted": n_read,
        "n_missing_or_failed": len(missing),
        "seconds": round(time.monotonic() - t0, 2),
    }
    if missing:
        print(f"WARNING: {len(missing)} ids not fetchable: {missing[:10]}...")
    return paths, stats


def load_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        try:  # gif/webp fall through PIL
            pil = Image.open(path).convert("RGB")
            img = np.asarray(pil)[:, :, ::-1].copy()
        except Exception:
            return None
    return img


def run_ocr(paths: dict[str, Path], probe_n: int = 64) -> tuple[dict, dict]:
    from memes_base.ocr_pipelines import run_pipeline

    order = sorted(paths)
    # batch-size sweep on a probe subset (easyocr recognizer batch), then bulk
    sweep: dict[int, float] = {}
    probe = order[:probe_n]
    for bs in (1, 8, 32):
        t0 = time.monotonic()
        for pid in probe:
            img = load_bgr(paths[pid])
            if img is not None:
                run_pipeline(img, "easyocr_filtered", apply_preprocess=False,
                             batch_size=bs)
        sweep[bs] = round(probe_n / (time.monotonic() - t0), 2)
    best = max(sweep, key=sweep.get)
    print(f"easyocr img/s by recognizer batch: {sweep} -> using {best}")

    results: dict[str, dict] = {}
    t0 = time.monotonic()
    for k, pid in enumerate(order):
        img = load_bgr(paths[pid])
        if img is None:
            results[pid] = {"text": "", "confidence": None, "decode_failed": True}
            continue
        t1 = time.monotonic()
        res = run_pipeline(img, "easyocr_filtered", apply_preprocess=False,
                           batch_size=best)
        results[pid] = {
            "text": res.text or "",
            "confidence": getattr(res, "confidence", None),
            "seconds": round(time.monotonic() - t1, 3),
        }
        if (k + 1) % 200 == 0:
            rate = (k + 1) / (time.monotonic() - t0)
            print(f"ocr {k + 1}/{len(order)}  {rate:.2f} img/s")
    timing = {
        "sweep_img_per_s": sweep,
        "chosen_batch_size": best,
        "bulk_img_per_s": round(len(order) / (time.monotonic() - t0), 2),
    }
    return results, timing


def run_embeddings(paths: dict[str, Path], ocr: dict, out: Path) -> dict:
    from experiments.wp01.config import PRIMARY_IMAGE_ENCODER
    from experiments.wp01.embedders import ClipBackend
    from experiments.wp01.pipelines import PIPELINES, embed_pipeline

    backend = ClipBackend.load(PRIMARY_IMAGE_ENCODER)
    order = sorted(paths)
    images, ids = [], []
    for pid in order:
        img = load_bgr(paths[pid])
        if img is not None:
            images.append(img)
            ids.append(pid)
    ocr_texts = [ocr[p]["text"] for p in ids]
    dummy_posts = [""] * len(ids)  # ADR 0017: no post text in these arms

    timing: dict[str, dict] = {}
    # image-encoder batch sweep on a probe subset
    probe = images[:128]
    sweep = {}
    for bs in (16, 32, 64, 128):
        t0 = time.monotonic()
        backend.encode_images(probe, batch_size=bs)
        sweep[bs] = round(len(probe) / (time.monotonic() - t0), 1)
    best = max(sweep, key=sweep.get)
    timing["image_encode_img_per_s_by_batch"] = sweep
    timing["chosen_batch_size"] = best
    print(f"openclip img/s by batch: {sweep} -> using {best}")

    for name in ("p_img_ocr", "p_vlm_only"):
        t0 = time.monotonic()
        vecs = embed_pipeline(
            backend, images, dummy_posts, pipeline=PIPELINES[name],
            ocr_texts=ocr_texts,
        )
        dt = time.monotonic() - t0
        np.save(out / f"embeddings_{name}.npy", vecs)
        timing[name] = {"n": len(ids), "seconds": round(dt, 2),
                        "img_per_s": round(len(ids) / dt, 1),
                        "dim": int(vecs.shape[1])}
        print(f"{name}: {vecs.shape} in {dt:.1f}s")
    (out / "ids.json").write_text(json.dumps(ids))
    return timing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="parquet with post_id column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = pd.read_parquet(args.ids)["post_id"].tolist()
    paths, extract_stats = extract(ids, out / "images")
    print(f"extract: {extract_stats}")

    ocr, ocr_timing = run_ocr(paths)
    (out / "ocr.json").write_text(json.dumps(ocr, indent=1))

    timing = {"extract": extract_stats, "ocr": ocr_timing}
    if not args.skip_embed:
        timing["embed"] = run_embeddings(paths, ocr, out)
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print("timing:", json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
