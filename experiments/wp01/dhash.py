"""Simple difference-hash for coarse template grouping."""

from __future__ import annotations

import numpy as np


def dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    """64-bit dHash from BGR uint8 image."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
