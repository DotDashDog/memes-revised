"""WP-18 measurement 3, step 1 (resumable rewrite of m3_embed.py, written by the third agent
after two unannounced shutdowns).

Same sample, encoder, OCR settings and seed as m3_embed.py; differences are only in ordering and
persistence:
  phase 1  image vectors for all sampled images (fast, GPU)        -> m3_img_vec.npz
  phase 2  EasyOCR text, checkpointed every chunk (append, fsync)   -> m3_ocr.jsonl
  phase 3  CLIP text vectors of the OCR text                        -> m3_embeddings.npz

The "[empty]" bug: `embed_pipeline` replaces an empty OCR string by the literal "[empty]", so all
text-free images share one text vector. Here the text half for an image whose OCR text is empty
(after .strip(), exactly the condition embed_pipeline uses) is set to ZERO. The buggy variant
(text half = vector of "[empty]") is also saved, as `p_img_ocr_bugged`, for comparison only.
Phase 1 also checks on the first chunk that encode_images == the image half of embed_pipeline
p_img_ocr, and that encode_texts(text) == its text half for non-empty texts.

    uv run python -m experiments.wp18.m3_embed_resumable
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.wp18.common import OUT, SEED

ROOT = Path("/mnt/d/memes-multisub")
CHUNK = 64
IMG_BATCH = 32      # moderate: the GPU is shared
OCR_BATCH = 8
Image.MAX_IMAGE_PIXELS = 200_000_000
OCR_CKPT = OUT / "m3_ocr.jsonl"
IMG_OUT = OUT / "m3_img_vec.npz"


def load_bgr(f, offset: int, length: int) -> np.ndarray | None:
    f.seek(offset)
    try:
        im = Image.open(io.BytesIO(f.read(length)))
        im.seek(0)  # GIF: first frame only, as in measurement 2
        return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    except Exception:
        return None


def iter_chunks(order: pd.DataFrame):
    handles: dict[str, object] = {}
    for s in range(0, len(order), CHUNK):
        rows = order.iloc[s:s + CHUNK]
        pids, imgs, failed = [], [], []
        for r in rows.itertuples():
            f = handles.setdefault(r.shard_id, open(ROOT / "images" / f"{r.shard_id}.tar", "rb"))
            img = load_bgr(f, int(r.offset), int(r.length))
            (failed.append(r.post_id) if img is None else (pids.append(r.post_id), imgs.append(img)))
        yield pids, imgs, failed


def read_ocr() -> dict:
    """All OCR checkpoint files (single-worker and sharded); a truncated last line is skipped."""
    done = {}
    for p in sorted(OUT.glob("m3_ocr*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["post_id"]] = r
            except json.JSONDecodeError:
                pass  # truncated last line after a crash: redo that image
    return done


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    import torch
    from experiments.wp01.config import PRIMARY_IMAGE_ENCODER
    from experiments.wp01.embedders import ClipBackend
    from experiments.wp01.pipelines import PIPELINES, embed_pipeline
    from memes_base.ocr_pipelines import run_pipeline

    sample = pd.read_parquet(OUT / "m3_sample.parquet")
    assert len(sample) == 6909 and sample.post_id.is_unique
    order = sample.sort_values(["shard_id", "offset"]).reset_index(drop=True)
    torch.manual_seed(SEED)
    backend = ClipBackend.load(PRIMARY_IMAGE_ENCODER)
    log(f"device {backend.device} {torch.cuda.get_device_name(0)}")

    only = os.environ.get("M3_ONLY")  # "img" | "ocr" | "text" | unset = all, in order
    # ---- phase 1: image vectors -------------------------------------------------------------
    if only in (None, "img") and not IMG_OUT.exists():
        ids, vecs, failed, checked = [], [], [], False
        t0 = time.monotonic()
        for pids, imgs, bad in iter_chunks(order):
            failed += bad
            v = backend.encode_images(imgs, batch_size=IMG_BATCH)
            if not checked and imgs:
                texts = ["some text", ""] + ["x"] * (len(imgs) - 2)
                ref = embed_pipeline(backend, imgs, [""] * len(imgs), pipeline=PIPELINES["p_img_ocr"],
                                     ocr_texts=texts)
                d = v.shape[1]
                assert np.allclose(ref[:, :d], v, atol=1e-4), "image half mismatch"
                tv = backend.encode_texts(["some text", "[empty]"])
                assert np.allclose(ref[0, d:], tv[0], atol=1e-4) and np.allclose(ref[1, d:], tv[1], atol=1e-4)
                checked = True
                log("check passed: encode_images/encode_texts reproduce embed_pipeline p_img_ocr")
            ids += pids
            vecs.append(v.astype(np.float32))
            log(f"img {len(ids)}/{len(order)} {len(ids) / (time.monotonic() - t0):.1f} img/s")
        tmp = IMG_OUT.with_suffix(".tmp.npz")
        np.savez(tmp, post_id=np.array(ids), img=np.vstack(vecs), decode_failed=np.array(failed, dtype=str))
        os.replace(tmp, IMG_OUT)
        log(f"phase 1 done, decode failures {len(failed)}")

    if only == "img":
        return
    # ---- phase 2: OCR, checkpointed -----------------------------------------------------------
    done = read_ocr()
    todo = order[~order.post_id.isin(done)]
    shard = os.environ.get("M3_SHARD")  # "k/n": this worker takes every n-th chunk of the todo list
    ckpt = OCR_CKPT
    if shard:
        k, nsh = map(int, shard.split("/"))
        todo = todo[(np.arange(len(todo)) // CHUNK) % nsh == k]
        ckpt = OUT / f"m3_ocr_s{k}of{nsh}.jsonl"
    log(f"OCR: {len(done)} already done, {len(todo)} to do")
    if only == "text":
        assert len(todo) == 0, "OCR incomplete"
    t0, n = time.monotonic(), 0
    with open(ckpt, "a") as fh:
        for pids, imgs, bad in iter_chunks(todo):
            for pid, img in zip(pids, imgs):
                try:
                    txt, err = run_pipeline(img, "easyocr_filtered", apply_preprocess=False,
                                            batch_size=OCR_BATCH).text or "", None
                except Exception as exc:  # recorded, not hidden
                    txt, err = "", repr(exc)[:200]
                fh.write(json.dumps({"post_id": pid, "text": txt, "error": err}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n += len(pids)
            el = time.monotonic() - t0
            log(f"ocr {len(done) + n}/{len(order)} {n / el:.2f} img/s  eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.0f} min")

    if only == "ocr":
        return
    # ---- phase 3: text vectors, empty -> zero ------------------------------------------------
    ocr = read_ocr()
    assert len(ocr) >= len(order), "OCR incomplete"
    z = np.load(IMG_OUT)
    ids = [str(p) for p in z["post_id"]]
    img = z["img"]
    texts = [(ocr[p]["text"] or "") for p in ids]
    empty = np.array([not t.strip() for t in texts])
    tv = np.zeros_like(img)
    nonempty_idx = np.flatnonzero(~empty)
    tv[nonempty_idx] = backend.encode_texts([texts[i].strip() for i in nonempty_idx]).astype(np.float32)
    empty_vec = backend.encode_texts(["[empty]"]).astype(np.float32)[0]
    tv_bug = tv.copy()
    tv_bug[empty] = empty_vec
    np.savez(OUT / "m3_embeddings.npz", post_id=np.array(ids),
             p_img_ocr=np.hstack([img, tv]), p_vlm_only=img, ocr_text_vec=tv,
             p_img_ocr_bugged=np.hstack([img, tv_bug]), ocr_empty=empty)
    meta = {
        "encoder": PRIMARY_IMAGE_ENCODER, "ocr": "easyocr_filtered, preprocess none",
        "n": len(ids), "decode_failed": [str(x) for x in z["decode_failed"]],
        "ocr_errors": sum(1 for p in ids if ocr[p]["error"]),
        "empty_ocr_n": int(empty.sum()), "empty_ocr_share": float(empty.mean()),
        "empty_text_handling": "text half set to zero for images with empty OCR text "
                               "(embed_pipeline would encode the literal '[empty]'); bugged variant "
                               "saved as p_img_ocr_bugged",
        "img_batch": IMG_BATCH, "ocr_batch": OCR_BATCH, "text_batch": 64,
        "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED,
        "gif_first_frame_only": True,
    }
    (OUT / "m3_embed_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done {meta}")


if __name__ == "__main__":
    main()
