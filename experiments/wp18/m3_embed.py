"""WP-18 measurement 3, step 1: balanced sample, OCR, and embeddings.

Sample: the same n images per community from the measurement-2 window (usable, decodable,
one post per byte-identical image), seed 18016. Embeddings: OpenCLIP ViT-B/16 laion2b via the
project's `embed_pipeline`: `p_img_ocr` (image (+) CLIP-text of EasyOCR text) and `p_vlm_only`
(image). OCR settings are those of experiments/wp04/ocr_embed.py (easyocr_filtered, no
preprocessing). Images are streamed from the tar shards in (shard, offset) order and processed
in chunks so nothing large is held in memory or written as loose files.

    uv run python -m experiments.wp18.m3_embed [--per-class 1500]
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.wp18.common import OUT, SEED, SUBS

ROOT = Path("/mnt/d/memes-multisub")
CHUNK = 128
IMG_BATCH = 32      # moderate: the GPU is shared
OCR_BATCH = 8
Image.MAX_IMAGE_PIXELS = 200_000_000


def draw_sample(per_class: int) -> pd.DataFrame:
    u = pd.read_parquet(OUT / "m2_usable_posts.parquet")
    idx = pd.read_parquet(OUT / "m2_shard_index.parquet")[["post_id", "shard_id", "offset", "length"]]
    dh = pd.read_parquet(OUT / "m2_dhash.parquet").drop_duplicates("sha256") \
        .set_index("sha256")[["width", "height", "n_frames", "ext", "n_bytes"]]
    u = u.merge(idx, on="post_id").join(dh, on="sha256")
    # one post per byte-identical image (earliest); drop images byte-identical across communities
    n_comm = u.groupby("sha256").community.nunique()
    u["sha_multi_comm"] = u.sha256.map(n_comm) > 1
    dedup = u.sort_values("created_utc").drop_duplicates("sha256", keep="first")
    avail = dedup.groupby("community").size().reindex(SUBS)
    n = int(min(per_class, avail.min()))
    rng = np.random.default_rng(SEED)
    parts = []
    for s in SUBS:
        d = dedup[dedup.community == s].sort_values("post_id")
        parts.append(d.iloc[np.sort(rng.choice(len(d), size=n, replace=False))])
    sample = pd.concat(parts, ignore_index=True)
    meta = {"seed": SEED, "per_class": n, "available_after_dedup": avail.to_dict(),
            "byte_identical_across_communities_in_sample": int(sample.sha_multi_comm.sum())}
    (OUT / "m3_sample_meta.json").write_text(json.dumps(meta, indent=1, default=int))
    print(meta)
    return sample


def load_bgr(body: bytes) -> np.ndarray | None:
    try:
        im = Image.open(io.BytesIO(body))
        im.seek(0)
        rgb = np.asarray(im.convert("RGB"))
        return rgb[:, :, ::-1].copy()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N sample rows")
    args = ap.parse_args()

    sample = draw_sample(args.per_class)
    sample.to_parquet(OUT / "m3_sample.parquet", index=False)
    order = sample.sort_values(["shard_id", "offset"]).reset_index(drop=True)
    if args.limit:
        order = order.sample(args.limit, random_state=0).sort_values(["shard_id", "offset"])

    import torch
    from memes_base.ocr_pipelines import run_pipeline
    from experiments.wp01.config import PRIMARY_IMAGE_ENCODER
    from experiments.wp01.embedders import ClipBackend
    from experiments.wp01.pipelines import PIPELINES, embed_pipeline

    torch.manual_seed(SEED)
    backend = ClipBackend.load(PRIMARY_IMAGE_ENCODER)
    print("device:", backend.device, torch.cuda.get_device_name(0))

    ids, vec, ocr, failed = [], [], {}, []
    handles: dict[str, object] = {}
    t0 = time.monotonic()
    t_ocr = t_emb = 0.0
    checked = False
    for s in range(0, len(order), CHUNK):
        rows = order.iloc[s:s + CHUNK]
        imgs, pids = [], []
        for r in rows.itertuples():
            f = handles.setdefault(r.shard_id, open(ROOT / "images" / f"{r.shard_id}.tar", "rb"))
            f.seek(int(r.offset))
            img = load_bgr(f.read(int(r.length)))
            if img is None:
                failed.append(r.post_id)
                continue
            imgs.append(img)
            pids.append(r.post_id)
        t1 = time.monotonic()
        texts = []
        for pid, img in zip(pids, imgs):
            try:
                o = run_pipeline(img, "easyocr_filtered", apply_preprocess=False, batch_size=OCR_BATCH)
                txt = o.text or ""
            except Exception as exc:  # recorded, not hidden
                txt = ""
                ocr.setdefault("_errors", {})[pid] = repr(exc)[:200]
            ocr[pid] = txt
            texts.append(txt)
        t2 = time.monotonic()
        v = embed_pipeline(backend, imgs, [""] * len(imgs), pipeline=PIPELINES["p_img_ocr"],
                           ocr_texts=texts)
        if not checked:  # image half of p_img_ocr must equal the image-only pipeline
            v2 = embed_pipeline(backend, imgs, [""] * len(imgs), pipeline=PIPELINES["p_vlm_only"])
            assert v2.shape[1] * 2 == v.shape[1] and np.allclose(v[:, :v2.shape[1]], v2, atol=1e-4)
            checked = True
        t3 = time.monotonic()
        t_ocr += t2 - t1
        t_emb += t3 - t2
        ids.extend(pids)
        vec.append(v.astype(np.float32))
        done = len(ids)
        print(f"{done}/{len(order)}  ocr {done / max(t_ocr, 1e-9):.1f} img/s  "
              f"embed {done / max(t_emb, 1e-9):.1f} img/s  elapsed {time.monotonic() - t0:.0f}s",
              flush=True)
    V = np.vstack(vec)
    tag = "_smoke" if args.limit else ""
    np.savez(OUT / f"m3_embeddings{tag}.npz", post_id=np.array(ids), p_img_ocr=V,
             p_vlm_only=V[:, :V.shape[1] // 2], ocr_text_vec=V[:, V.shape[1] // 2:])
    (OUT / f"m3_ocr{tag}.json").write_text(json.dumps(ocr, indent=0))
    (OUT / f"m3_embed_meta{tag}.json").write_text(json.dumps({
        "encoder": PRIMARY_IMAGE_ENCODER, "ocr": "easyocr_filtered, preprocess none",
        "n": len(ids), "decode_failed": failed, "ocr_seconds": round(t_ocr, 1),
        "embed_seconds": round(t_emb, 1), "img_batch": IMG_BATCH, "ocr_batch": OCR_BATCH,
        "torch": torch.__version__, "seed": SEED,
        "empty_ocr_share": float(np.mean([not ocr[p].strip() for p in ids])),
    }, indent=1))
    print("done", V.shape, "failed:", len(failed))


if __name__ == "__main__":
    main()
