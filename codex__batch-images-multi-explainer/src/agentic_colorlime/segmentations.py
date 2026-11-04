from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    relabel_sequential,
    slic,
    watershed,
)
from skimage.util import img_as_float, regular_grid

from .config import ExperimentConfig


@dataclass(frozen=True)
class SegmentationResult:
    method: str
    labels: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any]


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    # np.unique(..., return_inverse=True) is compatible across scikit-image
    # versions and gives contiguous feature IDs starting at zero.
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def segment_slic(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "n_segments": config.slic_n_segments,
        "compactness": config.slic_compactness,
        "sigma": config.slic_sigma,
    }
    labels = slic(
        image,
        n_segments=config.slic_n_segments,
        compactness=config.slic_compactness,
        sigma=config.slic_sigma,
        start_label=0,
        channel_axis=-1,
    )
    return SegmentationResult("slic", _normalize_labels(labels), params, {})


def segment_quickshift(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "kernel_size": config.quickshift_kernel_size,
        "max_dist": config.quickshift_max_dist,
        "ratio": config.quickshift_ratio,
    }
    kwargs = {
        "kernel_size": config.quickshift_kernel_size,
        "max_dist": config.quickshift_max_dist,
        "ratio": config.quickshift_ratio,
        "convert2lab": True,
        "channel_axis": -1,
    }
    # scikit-image renamed random_seed to rng. Support both APIs.
    import inspect

    if "rng" in inspect.signature(quickshift).parameters:
        kwargs["rng"] = config.random_seed
    else:
        kwargs["random_seed"] = config.random_seed
    labels = quickshift(image, **kwargs)
    return SegmentationResult("quickshift", _normalize_labels(labels), params, {})


def segment_felzenszwalb(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "scale": config.felzenszwalb_scale,
        "sigma": config.felzenszwalb_sigma,
        "min_size": config.felzenszwalb_min_size,
    }
    labels = felzenszwalb(
        image,
        scale=config.felzenszwalb_scale,
        sigma=config.felzenszwalb_sigma,
        min_size=config.felzenszwalb_min_size,
        channel_axis=-1,
    )
    return SegmentationResult("felzenszwalb", _normalize_labels(labels), params, {})


def segment_watershed(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    gray = rgb2gray(img_as_float(image))
    gradient = sobel(gray)
    markers = np.zeros(gray.shape, dtype=np.int32)
    seed_slices = regular_grid(gray.shape, n_points=config.watershed_markers)
    seed_mask = np.zeros(gray.shape, dtype=bool)
    seed_mask[seed_slices] = True
    coordinates = np.argwhere(seed_mask)
    if len(coordinates) == 0:
        coordinates = np.array([[gray.shape[0] // 2, gray.shape[1] // 2]])
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1, dtype=np.int32)
    labels = watershed(
        gradient,
        markers=markers,
        compactness=config.watershed_compactness,
    )
    params = {
        "requested_markers": config.watershed_markers,
        "actual_markers": int(len(coordinates)),
        "compactness": config.watershed_compactness,
    }
    return SegmentationResult("watershed", _normalize_labels(labels), params, {})


def segment_colorlime(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    """Weighted K-means Color-LIME segmentation used by the benchmark."""
    rgb = np.asarray(image, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, received shape {rgb.shape}")

    flat_pixels = rgb.reshape(-1, 3)
    unique_colors, inverse_indices, color_counts = np.unique(
        flat_pixels,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    original_unique_colors = int(len(unique_colors))
    requested_k = int(config.colorlime_k)

    if requested_k >= original_unique_colors:
        labels = inverse_indices.reshape(height, width).astype(np.int32)
        metadata = {
            "requested_k": requested_k,
            "actual_color_features": original_unique_colors,
            "original_unique_colors": original_unique_colors,
            "kmeans_inertia": 0.0,
            "kmeans_iterations": 0,
        }
        return SegmentationResult(
            "colorlime",
            _normalize_labels(labels),
            {"k": requested_k},
            metadata,
        )

    kmeans = KMeans(
        n_clusters=requested_k,
        init="k-means++",
        n_init=int(config.colorlime_n_init),
        max_iter=int(config.colorlime_max_iter),
        tol=float(config.colorlime_tol),
        random_state=int(config.random_seed),
        algorithm="lloyd",
        copy_x=False,
    )
    kmeans.fit(
        unique_colors.astype(np.float32, copy=False),
        sample_weight=color_counts.astype(np.float64, copy=False),
    )

    # Preserve the benchmark's canonical feature definition: every unique RGB
    # value inherits its weighted K-means cluster and every pixel inherits the
    # cluster of its original RGB value.
    unique_color_features = kmeans.labels_.astype(np.int32, copy=False)
    labels = unique_color_features[inverse_indices].reshape(height, width)

    metadata = {
        "requested_k": requested_k,
        "actual_color_features": int(len(np.unique(labels))),
        "original_unique_colors": original_unique_colors,
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_iterations": int(kmeans.n_iter_),
    }
    return SegmentationResult(
        "colorlime",
        _normalize_labels(labels),
        {"k": requested_k},
        metadata,
    )


SEGMENTATION_FUNCTIONS: dict[str, Callable[[np.ndarray, ExperimentConfig], SegmentationResult]] = {
    "slic": segment_slic,
    "quickshift": segment_quickshift,
    "felzenszwalb": segment_felzenszwalb,
    "watershed": segment_watershed,
    "colorlime": segment_colorlime,
}
