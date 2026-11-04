from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel
from skimage.util import img_as_float


@dataclass(frozen=True)
class ImageProfile:
    """Cheap image observations available before any expensive LIME call."""

    height: int
    width: int
    aspect_ratio: float
    unique_color_ratio: float
    colorfulness: float
    saturation_mean: float
    luminance_entropy: float
    edge_density: float
    texture_variance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def qualitative_summary(self) -> str:
        edge = "high" if self.edge_density >= 0.20 else "moderate" if self.edge_density >= 0.10 else "low"
        color = "high" if self.colorfulness >= 0.25 else "moderate" if self.colorfulness >= 0.12 else "low"
        texture = "high" if self.texture_variance >= 0.012 else "moderate" if self.texture_variance >= 0.004 else "low"
        return (
            f"{self.width}x{self.height} RGB image; {edge} edge density; "
            f"{texture} texture variation; {color} colorfulness; "
            f"mean saturation {self.saturation_mean:.3f}; luminance entropy {self.luminance_entropy:.3f}."
        )


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities /= total
    probabilities = probabilities[probabilities > 0]
    raw = -float(np.sum(probabilities * np.log2(probabilities)))
    return raw / np.log2(float(bins))


def profile_image(image: np.ndarray) -> ImageProfile:
    rgb_uint8 = np.asarray(image, dtype=np.uint8)
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb_uint8.shape}")

    height, width, _ = rgb_uint8.shape
    rgb = img_as_float(rgb_uint8)
    gray = rgb2gray(rgb)
    hsv = rgb2hsv(rgb)
    gradient = sobel(gray)

    # Sobel values are normalized for float images. A fixed threshold makes the
    # proportion informative across images instead of forcing a constant quantile.
    edge_density = float(np.mean(gradient >= 0.08))

    # Hasler/Süsstrunk-inspired normalized colourfulness proxy.
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    # Cap unique-colour work on very large images by deterministic striding.
    flat = rgb_uint8.reshape(-1, 3)
    if len(flat) > 200_000:
        stride = max(1, len(flat) // 200_000)
        flat = flat[::stride]
    unique_color_ratio = float(len(np.unique(flat, axis=0)) / max(len(flat), 1))

    local_dx = np.diff(gray, axis=1)
    local_dy = np.diff(gray, axis=0)
    texture_variance = float(0.5 * (np.var(local_dx) + np.var(local_dy)))

    return ImageProfile(
        height=int(height),
        width=int(width),
        aspect_ratio=float(width / max(height, 1)),
        unique_color_ratio=unique_color_ratio,
        colorfulness=colorfulness,
        saturation_mean=float(np.mean(hsv[..., 1])),
        luminance_entropy=_entropy(gray),
        edge_density=edge_density,
        texture_variance=texture_variance,
    )
