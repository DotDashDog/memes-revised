#!/usr/bin/env python3
"""Build cached image/OCR/text embeddings for a larger r/memes semantic run."""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from experiments.wp01.config import OCR_PIPELINE, OCR_PREPROCESS, PRIMARY_IMAGE_ENCODER, WP01_ROOT
from experiments.wp01.dhash import dhash
from experiments.wp01.embedders import ClipBackend
from experiments.wp01.io_utils import load_image_from_url
from experiments.wp01.sample import classify_stratum, collect_parquet_paths
from experiments.wp01.schema import load_json, save_json
from memes_base.ocr_pipelines import run_pipeline


LOCAL_TEXT_MODELS: dict[str, str] = {
    "minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_base_en_v1_5": "BAAI/bge-base-en-v1.5",
    "e5_base_v2": "intfloat/e5-base-v2",
}
OPENAI_TEXT_MODEL = "text-embedding-3-small"


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _post_id_from_link(link: str, fallback: str) -> str:
    match = re.search(r"/comments/([^/]+)/", link or "")
    if match:
        return match.group(1)
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", link or fallback)
    return cleaned[:48] or fallback


def _load_pool(dataset_dir: Path, max_pool_rows: int, seed: int) -> list[dict[str, Any]]:
    paths = collect_parquet_paths(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        df = pd.read_parquet(path, columns=["img_url", "title", "text", "link", "score", "created_utc"])
        for _, row in df.iterrows():
            url = row.get("img_url")
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            rows.append(
                {
                    "img_url": url,
                    "title": str(row.get("title") or ""),
                    "post_text": str(row.get("text") or ""),
                    "link": str(row.get("link") or ""),
                    "score": int(row.get("score") or 0),
                    "created_utc": int(row.get("created_utc") or 0),
                }
            )
            if 0 < max_pool_rows <= len(rows):
                break
        if 0 < max_pool_rows <= len(rows):
            break

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def build_or_resume_sample(
    *,
    dataset_dir: Path,
    output_dir: Path,
    target_items: int,
    max_pool_rows: int,
    seed: int,
    download_workers: int,
) -> dict[str, Any]:
    sample_path = output_dir / "sample.json"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if sample_path.exists():
        sample = load_json(sample_path)
    else:
        sample = {
            "version": 1,
            "seed": seed,
            "target_items": target_items,
            "description": "Cached 1000-example r/memes semantic embedding sample.",
            "items": [],
        }

    items = sample["items"]
    if len(items) >= target_items:
        return sample

    seen_urls = {item.get("img_url") for item in items}
    seen_post_ids = {item.get("post_id") for item in items}
    pool = _load_pool(dataset_dir, max_pool_rows=max_pool_rows, seed=seed)

    candidates = []
    for row in pool:
        if row["img_url"] in seen_urls:
            continue

        fallback_id = f"sem-{len(candidates) + len(items) + 1:05d}"
        post_id = _post_id_from_link(row["link"], fallback_id)
        if post_id in seen_post_ids:
            continue
        item_id = f"meme-{post_id}"
        image_path = images_dir / f"{item_id}.jpg"
        candidates.append((row, post_id, item_id, image_path))
        seen_urls.add(row["img_url"])
        seen_post_ids.add(post_id)
        if len(candidates) >= max(target_items * 4, target_items + 500):
            break

    def load_candidate(candidate):
        row, post_id, item_id, image_path = candidate
        if image_path.exists():
            image = cv2.imread(str(image_path))
        else:
            image = load_image_from_url(row["img_url"])
            if image is not None:
                cv2.imwrite(str(image_path), image)
        return row, post_id, item_id, image_path, image

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as executor:
        futures = [executor.submit(load_candidate, candidate) for candidate in candidates]
        for future in as_completed(futures):
            if len(items) >= target_items:
                for pending in futures:
                    pending.cancel()
                break
            completed += 1
            row, post_id, item_id, image_path, image = future.result()
            if completed % 100 == 0:
                print(f"  checked {completed}/{len(candidates)} image URLs", flush=True)
            if any(item.get("id") == item_id for item in items):
                continue
            if image is None:
                continue

            item = {
                "id": item_id,
                "post_id": post_id,
                "img_url": row["img_url"],
                "title": row["title"],
                "post_text": row["post_text"],
                "link": row["link"],
                "score": row["score"],
                "created_utc": row["created_utc"],
                "stratum": classify_stratum(row["title"], row["post_text"]),
                "local_path": f"images/{image_path.name}",
                "dhash": dhash(image),
            }
            items.append(item)

            if len(items) % 10 == 0:
                sample["n_items"] = len(items)
                save_json(sample_path, sample)
                print(f"  sample {len(items)}/{target_items}", flush=True)

    sample["n_items"] = len(items)
    sample["stratum_counts"] = {
        stratum: sum(1 for item in items if item.get("stratum") == stratum)
        for stratum in sorted({item.get("stratum", "unknown") for item in items})
    }
    save_json(sample_path, sample)
    return sample


def build_or_resume_ocr(
    *,
    output_dir: Path,
    sample: dict[str, Any],
    ocr_pipeline: str,
    ocr_preprocess: str,
) -> dict[str, Any]:
    path = output_dir / "ocr.json"
    if path.exists():
        manifest = load_json(path)
    else:
        manifest = {
            "version": 1,
            "ocr_pipeline": ocr_pipeline,
            "ocr_preprocess": ocr_preprocess,
            "items": {},
        }

    items = manifest["items"]
    for idx, entry in enumerate(sample["items"], start=1):
        if entry["id"] in items:
            continue
        image = cv2.imread(str(output_dir / entry["local_path"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side > 1400:
            scale = 1400 / max_side
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        start = time.perf_counter()
        result = run_pipeline(
            image,
            pipeline_name=ocr_pipeline,
            apply_preprocess=ocr_preprocess != "none",
            preprocess_mode=ocr_preprocess,
        )
        items[entry["id"]] = {
            "text": result.text or "",
            "confidence": result.confidence,
            "seconds": time.perf_counter() - start,
        }
        if idx % 25 == 0:
            save_json(path, manifest)
            print(f"  OCR {idx}/{len(sample['items'])}", flush=True)

    save_json(path, manifest)
    return manifest


def _combined_text(entry: dict[str, Any], ocr_text: str) -> str:
    fields = []
    if entry.get("title", "").strip():
        fields.append(f"title: {entry['title'].strip()}")
    if entry.get("post_text", "").strip():
        fields.append(f"post: {entry['post_text'].strip()}")
    if ocr_text.strip():
        fields.append(f"image text: {ocr_text.strip()}")
    return "\n".join(fields) if fields else "[empty]"


def save_text_inputs(output_dir: Path, sample: dict[str, Any], ocr: dict[str, Any]) -> list[str]:
    texts = []
    payload = {"version": 1, "items": []}
    for entry in sample["items"]:
        ocr_text = ocr["items"].get(entry["id"], {}).get("text", "")
        text = _combined_text(entry, ocr_text)
        texts.append(text)
        payload["items"].append({"id": entry["id"], "text": text})
    save_json(output_dir / "text_inputs.json", payload)
    return texts


def cache_image_embeddings(output_dir: Path, sample: dict[str, Any], encoder_key: str) -> Path:
    path = output_dir / "embeddings" / f"image_{encoder_key}.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = ClipBackend.load(encoder_key)
    images = [cv2.imread(str(output_dir / entry["local_path"])) for entry in sample["items"]]
    vectors = backend.encode_images(images)
    np.savez_compressed(
        path,
        ids=np.asarray([entry["id"] for entry in sample["items"]]),
        embeddings=vectors.astype(np.float32),
        model=np.asarray(encoder_key),
    )
    return path


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return vectors / denom


def cache_local_text_embeddings(
    output_dir: Path,
    sample: dict[str, Any],
    texts: list[str],
    *,
    model_key: str,
    model_name: str,
    batch_size: int,
) -> Path:
    path = output_dir / "embeddings" / f"text_{model_key}.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    # CPU is slower, but avoids MPS memory spikes when running several encoders in sequence.
    model = SentenceTransformer(model_name, device="cpu")
    input_texts = [f"passage: {text}" for text in texts] if model_key.startswith("e5_") else texts
    vectors = model.encode(
        input_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.savez_compressed(
        path,
        ids=np.asarray([entry["id"] for entry in sample["items"]]),
        embeddings=vectors.astype(np.float32),
        model=np.asarray(model_name),
    )
    return path


def cache_openai_embeddings(
    output_dir: Path,
    sample: dict[str, Any],
    texts: list[str],
    *,
    model_name: str = OPENAI_TEXT_MODEL,
    batch_size: int = 256,
) -> Path | None:
    path = output_dir / "embeddings" / f"text_openai_{model_name}.npz"
    if path.exists():
        return path
    if not os.environ.get("OPENAI_API_KEY"):
        save_json(
            output_dir / "embeddings" / f"text_openai_{model_name}.skipped.json",
            {
                "model": model_name,
                "reason": "OPENAI_API_KEY is not set",
                "n_items": len(texts),
            },
        )
        return None

    from openai import OpenAI

    client = OpenAI()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend([row.embedding for row in response.data])
        print(f"  OpenAI {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    arr = _normalize_rows(np.asarray(vectors, dtype=np.float32))
    np.savez_compressed(
        path,
        ids=np.asarray([entry["id"] for entry in sample["items"]]),
        embeddings=arr,
        model=np.asarray(model_name),
    )
    return path


def write_summary(
    output_dir: Path,
    *,
    sample: dict[str, Any],
    ocr: dict[str, Any],
    image_path: Path,
    text_paths: dict[str, str | None],
) -> None:
    ocr_rows = list(ocr["items"].values())
    summary = {
        "description": "Cached 1000-example r/memes image/OCR/text embedding run.",
        "n_items": len(sample["items"]),
        "stratum_counts": sample.get("stratum_counts", {}),
        "image_embedding": str(image_path.relative_to(output_dir)),
        "text_embeddings": text_paths,
        "ocr": {
            "pipeline": ocr.get("ocr_pipeline"),
            "preprocess": ocr.get("ocr_preprocess"),
            "mean_seconds": float(np.mean([row["seconds"] for row in ocr_rows])) if ocr_rows else None,
            "mean_confidence": float(
                np.mean([row["confidence"] for row in ocr_rows if row.get("confidence") is not None])
            )
            if ocr_rows
            else None,
        },
    }
    save_json(output_dir / "summary.json", summary)

    lines = [
        "# WP-01 Semantic Embedding Cache",
        "",
        f"Items: {summary['n_items']}",
        f"Image embedding: `{summary['image_embedding']}`",
        f"OCR: `{summary['ocr']['pipeline']}:{summary['ocr']['preprocess']}`",
        f"Mean OCR seconds/image: {summary['ocr']['mean_seconds']:.2f}",
        "",
        "## Text Embeddings",
        "",
    ]
    for key, value in text_paths.items():
        lines.append(f"- `{key}`: {value or 'skipped'}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Cache image/OCR/text embeddings for 1000 r/memes examples.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_saves/reddit/test_reddit_dataset"))
    parser.add_argument("--output-dir", type=Path, default=WP01_ROOT / "semantic-embedding-1000-001")
    parser.add_argument("--target-items", type=int, default=1000)
    parser.add_argument("--max-pool-rows", type=int, default=12000)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-encoder", default=PRIMARY_IMAGE_ENCODER)
    parser.add_argument("--ocr-pipeline", default=OCR_PIPELINE)
    parser.add_argument("--ocr-preprocess", default=OCR_PREPROCESS)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--skip-openai", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = build_or_resume_sample(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        target_items=args.target_items,
        max_pool_rows=args.max_pool_rows,
        seed=args.seed,
        download_workers=args.download_workers,
    )
    if len(sample["items"]) < args.target_items:
        print(f"Warning: only cached {len(sample['items'])}/{args.target_items} images", flush=True)

    ocr = build_or_resume_ocr(
        output_dir=args.output_dir,
        sample=sample,
        ocr_pipeline=args.ocr_pipeline,
        ocr_preprocess=args.ocr_preprocess,
    )
    texts = save_text_inputs(args.output_dir, sample, ocr)
    image_path = cache_image_embeddings(args.output_dir, sample, args.image_encoder)

    text_paths: dict[str, str | None] = {}
    for model_key, model_name in LOCAL_TEXT_MODELS.items():
        print(f"Text embeddings: {model_key}", flush=True)
        path = cache_local_text_embeddings(
            args.output_dir,
            sample,
            texts,
            model_key=model_key,
            model_name=model_name,
            batch_size=args.text_batch_size,
        )
        text_paths[model_key] = str(path.relative_to(args.output_dir))

    if args.skip_openai:
        text_paths["openai_text_embedding_3_small"] = None
    else:
        path = cache_openai_embeddings(args.output_dir, sample, texts)
        text_paths["openai_text_embedding_3_small"] = str(path.relative_to(args.output_dir)) if path else None

    write_summary(args.output_dir, sample=sample, ocr=ocr, image_path=image_path, text_paths=text_paths)
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
