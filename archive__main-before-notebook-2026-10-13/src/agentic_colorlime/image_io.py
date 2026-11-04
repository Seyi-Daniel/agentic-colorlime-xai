from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.load()
    return np.asarray(image, dtype=np.uint8)


def load_image_from_path(path: str | Path) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    with Image.open(resolved) as image:
        return pil_to_rgb_array(image)


def load_image_from_upload(uploaded: BinaryIO | bytes) -> np.ndarray:
    if isinstance(uploaded, bytes):
        stream = io.BytesIO(uploaded)
    else:
        stream = uploaded
    with Image.open(stream) as image:
        return pil_to_rgb_array(image)


def image_to_data_url(image: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    """Create a compact JPEG data URL for an optional multimodal agent input."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
