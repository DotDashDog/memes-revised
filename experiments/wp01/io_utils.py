"""Lightweight image download helpers (no legacy image_processing import)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import requests


def get_direct_image_url(url: str) -> str:
    if "imgur.com" in url and not any(url.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")):
        return url + ".jpg"
    if "imgflip.com/i/" in url:
        image_id = url.split("/")[-1].split("#")[0]
        return f"https://i.imgflip.com/{image_id}.jpg"
    return url


def load_image_from_url(image_url: str, max_retries: int = 3) -> np.ndarray | None:
    image_url = get_direct_image_url(image_url)
    headers = {"User-Agent": "memes-revised-wp01/1.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(image_url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = np.frombuffer(resp.content, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None
