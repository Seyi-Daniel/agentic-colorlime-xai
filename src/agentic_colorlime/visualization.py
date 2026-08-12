from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries, mark_boundaries


def explanation_overlay(image: np.ndarray, critical_mask: np.ndarray) -> np.ndarray:
    image_float = np.asarray(image, dtype=np.float32) / 255.0
    mask = np.asarray(critical_mask, dtype=bool)
    overlay = image_float.copy()
    gray = np.mean(overlay, axis=2, keepdims=True)
    overlay[~mask] = 0.25 * overlay[~mask] + 0.75 * gray[~mask]
    boundaries = find_boundaries(mask, mode="thick")
    overlay[boundaries] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(np.rint(overlay * 255), 0, 255).astype(np.uint8)


def segmentation_preview(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    preview = mark_boundaries(
        np.asarray(image, dtype=np.float32) / 255.0,
        np.asarray(segments, dtype=np.int32),
        mode="thick",
    )
    return np.clip(np.rint(preview * 255), 0, 255).astype(np.uint8)


def mask_on_white(image: np.ndarray, critical_mask: np.ndarray) -> np.ndarray:
    output = np.full_like(np.asarray(image, dtype=np.uint8), 255)
    mask = np.asarray(critical_mask, dtype=bool)
    output[mask] = np.asarray(image, dtype=np.uint8)[mask]
    return output


def save_rgb(array: np.ndarray, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(destination)
    return str(destination.resolve())


def save_mask(mask: np.ndarray, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(destination)
    return str(destination.resolve())
