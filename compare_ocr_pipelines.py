import argparse
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memes_base.image_processing import load_from_url
from memes_base.ocr_pipelines import OcrResult, available_pipelines, run_pipeline, preprocess_image


def _collect_parquet_files(dataset_dir: Path) -> List[Path]:
    if dataset_dir.is_file() and dataset_dir.suffix == ".parquet":
        return [dataset_dir]
    if dataset_dir.is_dir():
        return sorted(dataset_dir.glob("*.parquet"))
    return []


def _load_urls(parquet_paths: List[Path], max_rows: int) -> List[str]:
    urls = []
    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(parquet_path, columns=["img_url"])
            if max_rows > 0:
                df = df.head(max_rows)
            urls.extend([u for u in df["img_url"].tolist() if isinstance(u, str)])
        except Exception:
            continue
    return urls


def _download_samples(urls: List[str], target_count: int, seed: int) -> List[Tuple[str, np.ndarray]]:
    rng = random.Random(seed)
    shuffled = urls[:]
    rng.shuffle(shuffled)

    samples = []
    for url in shuffled:
        image = load_from_url(url)
        if image is None:
            continue
        samples.append((url, image))
        if len(samples) >= target_count:
            break
    return samples


def _render_results(
    image: np.ndarray,
    processed_image: np.ndarray,
    url: str,
    results: Dict[str, OcrResult],
    no_display: bool,
) -> None:
    rows = 2 + len(results)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows))
    if rows == 1:
        axes = [axes]

    axes[0].imshow(image[:, :, ::-1])
    axes[0].set_title(url)
    axes[0].axis("off")

    if processed_image.ndim == 2:
        axes[1].imshow(processed_image, cmap="gray")
    else:
        axes[1].imshow(processed_image[:, :, ::-1])
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")

    for idx, (name, result) in enumerate(results.items(), start=2):
        ax = axes[idx]
        ax.axis("off")
        text = result.text or "[no text]"
        conf = f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
        display_text = f"{name} (conf={conf}):\n{textwrap.fill(text, width=90)}"
        ax.text(0.01, 0.95, display_text, va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    if not no_display:
        plt.show()
    plt.close(fig)

def _print_results(url: str, results: Dict[str, OcrResult]) -> None:
    print(f"\nURL: {url}")
    for name, result in results.items():
        conf = f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
        text = result.text if result.text else "[no text]"
        print(f"  - {name} (conf={conf}): {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR pipelines on random meme images.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_saves/reddit/test_reddit_dataset",
        help="Directory containing parquet files with img_url column.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rows-per-parquet", type=int, default=0)
    parser.add_argument("--pipelines", type=str, default="paddleocr,dbnet,easyocr")
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="white_text",
        help="Preprocess mode: outline, default, contrast, white_text, white_or_black, or none.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    parquet_paths = _collect_parquet_files(dataset_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    urls = _load_urls(parquet_paths, max_rows=args.max_rows_per_parquet)
    if not urls:
        raise RuntimeError("No image URLs found in dataset.")

    samples = _download_samples(urls, target_count=args.num_samples, seed=args.seed)
    if not samples:
        raise RuntimeError("Failed to download any images.")

    pipeline_names = [p.strip().lower() for p in args.pipelines.split(",") if p.strip()]
    for name in pipeline_names:
        if name not in available_pipelines():
            raise ValueError(f"Unsupported pipeline: {name}")

    for url, image in samples:
        results: Dict[str, OcrResult] = {}
        for pipeline_name in pipeline_names:
            try:
                results[pipeline_name] = run_pipeline(
                    image,
                    pipeline_name=pipeline_name,
                    apply_preprocess=args.preprocess,
                    preprocess_mode="none" if args.preprocess_mode == "none" else args.preprocess_mode,
                )
            except Exception as exc:
                print(f"OCR pipeline error: {pipeline_name} on {url}: {exc}")
                results[pipeline_name] = OcrResult(text=f"[error: {exc}]")

        _print_results(url, results)
        processed = preprocess_image(
            image, "none" if args.preprocess_mode == "none" else args.preprocess_mode
        )
        _render_results(image, processed, url, results, no_display=args.no_display)


if __name__ == "__main__":
    main()
