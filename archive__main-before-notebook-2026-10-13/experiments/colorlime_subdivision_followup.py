#!/usr/bin/env python3
"""
Optimized ViT / ImageNet experiment for hierarchical Color-LIME subdivisions.

The runner compares eight explanation methods on exactly 100 correctly
predicted ImageNet validation images by default:

1. Default image LIME.
2. Standard black-off Color-LIME with K=100 parent colors.
3. Direct black-off Color-LIME with K=300 colors (feature-count control).
4. K=100 x 3 balanced spatial-PCA children.
5. K=100 x 3 connected-component children.
6. K=100 x 3 ViT-patch-aware children.
7. K=100 x 3 texture-aware children.
8. K=100 x 3 fine-shade children in CIELAB color space.

The five hierarchical methods reuse the exact same K=100 parent color
segmentation.  Their requested child count is three, or fewer only when a
parent contains fewer than three pixels.  Semantic methods use a deterministic
balanced spatial-PCA fallback when necessary so the hierarchical methods have
identical feature counts.  Fallback parent and pixel counts are recorded.

Experimental controls and recovery behavior:

* num_samples=1000 by default for every explanation.
* Every Color-LIME method uses black as its LIME off-state.
* CIR/DIR also use black deletion for every method.
* Equal-feature-count Color-LIME methods share the same binary perturbation
  matrix, prediction batches, kernel, and surrogate random seed.
* Top 10%/20% means a percentage of positive feature count, not image area.
* White-background views copy actual selected RGB pixels from the image.
* The dataset is scanned until exactly 100 correct ViT predictions are found.
* results.csv, packed masks, and a comparison sheet are saved after each image.
* summary.csv, checkpoint metadata, and CIR/DIR plots refresh every five images.
* --resume continues a compatible interrupted run.

Typical installation:

    python -m pip install torch transformers datasets accelerate lime \
        scikit-learn scikit-image scipy pillow matplotlib numpy

ImageNet may require a Hugging Face read token in HF_TOKEN.  A normal run is:

    python imagenet_vit_colorlime_subdivision_experiment.py

The model is google/vit-base-patch16-224 by default.  Its trained 16x16 patch
size is read from the model configuration and used only by the patch-aware
subdivision; this script does not alter the model's trained patch embedding.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    short_name: str
    method_type: str

    @property
    def is_color_lime(self) -> bool:
        return self.method_type != "default"

    @property
    def is_hierarchical(self) -> bool:
        return self.method_type == "hierarchical"


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("default_lime", "Default LIME", "Default LIME", "default"),
    MethodSpec(
        "color_parent",
        "Color-LIME — Direct Parent K",
        "Direct Parent K",
        "parent",
    ),
    MethodSpec(
        "color_direct",
        "Color-LIME — Direct K×Subdivisions Control",
        "Direct K×S",
        "direct",
    ),
    MethodSpec(
        "spatial_pca",
        "Hierarchical — Balanced Spatial PCA",
        "Spatial PCA",
        "hierarchical",
    ),
    MethodSpec(
        "connected_components",
        "Hierarchical — Connected Components",
        "Components",
        "hierarchical",
    ),
    MethodSpec(
        "vit_patch",
        "Hierarchical — ViT Patch Aware",
        "ViT Patch",
        "hierarchical",
    ),
    MethodSpec(
        "texture",
        "Hierarchical — Texture Aware",
        "Texture",
        "hierarchical",
    ),
    MethodSpec(
        "fine_shade",
        "Hierarchical — Fine Shade (CIELAB)",
        "Fine Shade",
        "hierarchical",
    ),
)

METHOD_BY_KEY = {method.key: method for method in METHODS}
COLOR_METHODS = tuple(method for method in METHODS if method.is_color_lime)
HIERARCHICAL_KEYS = tuple(method.key for method in METHODS if method.is_hierarchical)


@dataclass
class ImageRecord:
    image_position: int
    dataset_index: int
    label: int
    image: np.ndarray
    label_name: str = ""
    predicted_class: int = -1
    predicted_label: str = ""
    predicted_confidence: float = float("nan")


@dataclass
class ParentColorData:
    segments: np.ndarray
    unique_colors: np.ndarray
    pixel_to_unique: np.ndarray
    unique_counts: np.ndarray
    unique_to_parent: np.ndarray
    parent_pixel_counts: np.ndarray

    @property
    def n_parents(self) -> int:
        return int(len(self.parent_pixel_counts))


@dataclass
class SegmentationResult:
    method_key: str
    segments: np.ndarray
    construction_seconds: float
    parent_count: int
    requested_children: int
    ideal_feature_count: int
    fallback_parent_count: int = 0
    fallback_pixel_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.segments.max()) + 1


@dataclass
class ExplanationResult:
    method_key: str
    segments: np.ndarray
    weights: list[tuple[int, float]]
    surrogate_score: float
    runtime_seconds: float


@dataclass
class VisualRecord:
    image_position: int
    dataset_index: int
    label_name: str
    predicted_label: str
    predicted_confidence: float
    image: np.ndarray
    masks: dict[str, tuple[np.ndarray, np.ndarray]]
    feature_counts: dict[str, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare five K=100 x 3 Color-LIME subdivision ideas with Default "
            "LIME and direct K=100/K=300 controls."
        )
    )
    parser.add_argument(
        "--target-correct",
        "--num-images",
        dest="target_correct",
        type=int,
        default=100,
        help="Number of correctly predicted images to explain (default: 100).",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=50000,
        help="Maximum dataset examples to screen while finding correct images.",
    )
    parser.add_argument("--screening-batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--k-colors", type=int, default=100)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--critical-area-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-n-init", type=int, default=3)
    parser.add_argument("--lime-batch-size", type=int, default=32)
    parser.add_argument("--model-microbatch", type=int, default=128)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help=(
            "Refresh summary.csv, checkpoint state, and plots after this many "
            "new images. Per-image numerical and visual files are always saved."
        ),
    )
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--dataset-name", default="ILSVRC/imagenet-1k")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("imagenet_vit_colorlime_subdivision_outputs"),
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream ImageNet rather than using an indexed local dataset.",
    )
    parser.add_argument(
        "--fast-gpu-preprocess",
        action="store_true",
        help=(
            "Normalize resized uint8 images directly with torch. Faster, while "
            "the exact official processor path remains the default."
        ),
    )
    parser.add_argument(
        "--no-comparison-sheets",
        action="store_true",
        help="Skip per-image eight-row visual comparison sheets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted run in --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing nonempty output directory.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test all subdivision and output invariants without ViT/ImageNet.",
    )
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.target_correct < 1:
        parser.error("--target-correct must be positive")
    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    if args.max_scan < args.target_correct:
        parser.error("--max-scan must be at least --target-correct")
    if args.screening_batch_size < 1:
        parser.error("--screening-batch-size must be positive")
    if args.num_samples < 2:
        parser.error("--num-samples must be at least 2")
    if args.k_colors < 2:
        parser.error("--k-colors must be at least 2")
    if args.subdivisions not in (2, 3, 4):
        parser.error("--subdivisions must be 2, 3, or 4")
    if not 0.0 < args.critical_area_fraction <= 1.0:
        parser.error("--critical-area-fraction must be in (0, 1]")
    if args.kmeans_n_init < 1:
        parser.error("--kmeans-n-init must be positive")
    if args.lime_batch_size < 1 or args.model_microbatch < 1:
        parser.error("batch sizes must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")


def import_experiment_dependencies() -> dict[str, Any]:
    try:
        import matplotlib  # noqa: F401 - fail before a long run if plots cannot save
        import torch
        from datasets import load_dataset
        from lime import lime_image
        from lime.lime_base import LimeBase
        from sklearn.cluster import KMeans
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise SystemExit(
            "Missing experiment dependency: "
            f"{exc}.\nInstall the packages listed at the top of this file."
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "lime_image": lime_image,
        "LimeBase": LimeBase,
        "KMeans": KMeans,
        "AutoImageProcessor": AutoImageProcessor,
        "AutoModelForImageClassification": AutoModelForImageClassification,
    }


def stable_seed(*parts: Any, bits: int = 64) -> int:
    digest_size = 8 if bits > 32 else 4
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=digest_size).digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value if bits > 32 else value & ((1 << bits) - 1)


def prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> Path:
    resolved = path.expanduser().resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to use protected output path: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if resume:
            return resolved
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {resolved}\n"
                "Use --resume, choose another directory, or explicitly use --overwrite."
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def processor_image_size(processor: Any) -> tuple[int, int]:
    size = getattr(processor, "size", None) or {"height": 224, "width": 224}
    if isinstance(size, int):
        return size, size
    if "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    shortest = int(size.get("shortest_edge", 224))
    return shortest, shortest


def resize_to_model_space(image: Image.Image, processor: Any) -> np.ndarray:
    height, width = processor_image_size(processor)
    resample_value = getattr(processor, "resample", int(Image.Resampling.BILINEAR))
    try:
        resample = Image.Resampling(int(resample_value))
    except (TypeError, ValueError):
        resample = Image.Resampling.BILINEAR
    resized = image.convert("RGB").resize((width, height), resample=resample)
    return np.asarray(resized, dtype=np.uint8).copy()


def model_patch_size(model: Any) -> tuple[int, int]:
    config = getattr(model, "config", None)
    patch_size = getattr(config, "patch_size", None)
    if patch_size is None and getattr(config, "vision_config", None) is not None:
        patch_size = getattr(config.vision_config, "patch_size", None)
    if patch_size is None:
        raise ValueError("The model configuration does not expose a ViT patch_size.")
    if isinstance(patch_size, Sequence) and not isinstance(patch_size, (str, bytes)):
        values = list(patch_size)
        if len(values) != 2:
            raise ValueError(f"Unexpected patch_size: {patch_size!r}")
        return int(values[0]), int(values[1])
    value = int(patch_size)
    return value, value


class ViTPredictor:
    def __init__(
        self,
        torch: Any,
        processor: Any,
        model: Any,
        device: Any,
        microbatch: int,
        precision: str,
        fast_gpu_preprocess: bool,
    ) -> None:
        self.torch = torch
        self.processor = processor
        self.model = model
        self.device = device
        self.microbatch = microbatch
        self.fast_gpu_preprocess = fast_gpu_preprocess
        self.autocast_dtype = self._resolve_autocast_dtype(precision)
        self.image_mean = np.asarray(getattr(processor, "image_mean", [0.5] * 3))
        self.image_std = np.asarray(getattr(processor, "image_std", [0.5] * 3))

    def _resolve_autocast_dtype(self, precision: str) -> Any | None:
        if self.device.type != "cuda" or precision == "float32":
            return None
        if precision == "bfloat16":
            return self.torch.bfloat16
        if precision in ("auto", "float16"):
            return self.torch.float16
        return None

    def _autocast(self) -> Any:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        )

    def _prepare_inputs(self, images: Sequence[np.ndarray]) -> dict[str, Any]:
        if self.fast_gpu_preprocess:
            stacked = np.stack(images, axis=0)
            tensor = self.torch.from_numpy(stacked).permute(0, 3, 1, 2)
            tensor = tensor.to(self.device, dtype=self.torch.float32, non_blocking=True)
            tensor.div_(255.0)
            mean = self.torch.as_tensor(
                self.image_mean, device=self.device, dtype=tensor.dtype
            ).view(1, 3, 1, 1)
            std = self.torch.as_tensor(
                self.image_std, device=self.device, dtype=tensor.dtype
            ).view(1, 3, 1, 1)
            return {"pixel_values": (tensor - mean) / std}
        processed = self.processor(images=list(images), return_tensors="pt")
        return {
            key: value.to(self.device, non_blocking=True)
            if self.torch.is_tensor(value)
            else value
            for key, value in processed.items()
        }

    def predict_proba(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(images, np.ndarray) and images.ndim == 4:
            image_sequence: Sequence[np.ndarray] = images
        else:
            image_sequence = images  # type: ignore[assignment]
        total = len(image_sequence)
        if total == 0:
            return np.empty((0, 0), dtype=np.float32)
        outputs: list[np.ndarray] = []
        position = 0
        current_microbatch = min(self.microbatch, total)
        while position < total:
            end = min(position + current_microbatch, total)
            try:
                inputs = self._prepare_inputs(image_sequence[position:end])
                with self.torch.inference_mode(), self._autocast():
                    logits = self.model(**inputs).logits
                    probabilities = self.torch.softmax(logits.float(), dim=-1)
                outputs.append(probabilities.cpu().numpy().astype(np.float32, copy=False))
                position = end
                del inputs, logits, probabilities
            except self.torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or current_microbatch <= 1:
                    raise
                self.torch.cuda.empty_cache()
                current_microbatch = max(1, current_microbatch // 2)
                self.microbatch = min(self.microbatch, current_microbatch)
                print(
                    f"CUDA memory pressure: model microbatch reduced to {current_microbatch}",
                    flush=True,
                )
        return np.concatenate(outputs, axis=0)


def get_label_name(model: Any, class_id: int) -> str:
    id2label = getattr(model.config, "id2label", {}) or {}
    return str(id2label.get(class_id, id2label.get(str(class_id), class_id)))


def load_dataset_split(args: argparse.Namespace, load_dataset: Callable[..., Any]) -> Any:
    token = os.environ.get("HF_TOKEN") or None
    kwargs: dict[str, Any] = {
        "split": args.dataset_split,
        "streaming": args.streaming,
    }
    if token:
        kwargs["token"] = token
    try:
        return load_dataset(args.dataset_name, **kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        return load_dataset(args.dataset_name, **kwargs)


def iter_dataset_examples(
    dataset: Any, start_index: int, max_scan: int, streaming: bool
) -> Iterator[tuple[int, Any]]:
    stop = start_index + max_scan
    if streaming:
        for dataset_index, sample in enumerate(dataset):
            if dataset_index < start_index:
                continue
            if dataset_index >= stop:
                break
            yield dataset_index, sample
        return
    dataset_stop = min(stop, len(dataset))
    for dataset_index in range(start_index, dataset_stop):
        yield dataset_index, dataset[dataset_index]


def select_correct_records(
    args: argparse.Namespace,
    dataset: Any,
    processor: Any,
    predictor: ViTPredictor,
    model: Any,
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    """Scan deterministically until exactly target_correct predictions are correct."""
    source = iter(
        iter_dataset_examples(dataset, args.start_index, args.max_scan, args.streaming)
    )
    selected: list[ImageRecord] = []
    prediction_rows: list[dict[str, Any]] = []
    exhausted = False
    while len(selected) < args.target_correct and not exhausted:
        candidates: list[tuple[int, int, np.ndarray]] = []
        for _ in range(args.screening_batch_size):
            try:
                dataset_index, sample = next(source)
            except StopIteration:
                exhausted = True
                break
            image_value = sample[args.image_column]
            if not isinstance(image_value, Image.Image):
                image_value = Image.fromarray(np.asarray(image_value))
            image = resize_to_model_space(image_value, processor)
            candidates.append((int(dataset_index), int(sample[args.label_column]), image))
        if not candidates:
            break
        probabilities_batch = predictor.predict_proba([item[2] for item in candidates])
        for (dataset_index, label, image), probabilities in zip(
            candidates, probabilities_batch
        ):
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            correct = prediction == label
            selected_position: int | str = ""
            if correct:
                selected_position = len(selected) + 1
                selected.append(
                    ImageRecord(
                        image_position=int(selected_position),
                        dataset_index=dataset_index,
                        label=label,
                        image=image,
                        label_name=get_label_name(model, label),
                        predicted_class=prediction,
                        predicted_label=get_label_name(model, prediction),
                        predicted_confidence=confidence,
                    )
                )
            prediction_rows.append(
                {
                    "scan_position": len(prediction_rows) + 1,
                    "dataset_index": dataset_index,
                    "ground_truth_class": label,
                    "ground_truth_label": get_label_name(model, label),
                    "predicted_class": prediction,
                    "predicted_label": get_label_name(model, prediction),
                    "predicted_confidence": confidence,
                    "correct": int(correct),
                    "selected_correct_position": selected_position,
                }
            )
            if len(selected) == args.target_correct:
                break
        print(
            f"  screened {len(prediction_rows)} images; found "
            f"{len(selected)}/{args.target_correct} correct",
            flush=True,
        )
    if len(selected) != args.target_correct:
        raise RuntimeError(
            f"Found only {len(selected)} correct predictions after screening "
            f"{len(prediction_rows)} images. Increase --max-scan or check labels/model."
        )
    return selected, prediction_rows


def canonicalize_cluster_labels(labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float64)
    keys = tuple(centers[:, column] for column in range(centers.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    mapping = np.empty(len(order), dtype=np.int32)
    mapping[order] = np.arange(len(order), dtype=np.int32)
    return mapping[labels]


def build_parent_color_data(
    image: np.ndarray,
    k_colors: int,
    seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> ParentColorData:
    flat = image.reshape(-1, 3)
    unique_colors, pixel_to_unique, unique_counts = np.unique(
        flat, axis=0, return_inverse=True, return_counts=True
    )
    n_clusters = min(k_colors, len(unique_colors))
    if n_clusters < 2:
        raise ValueError("Color-LIME requires at least two distinct RGB colors.")
    estimator = KMeans(
        n_clusters=n_clusters,
        random_state=int(seed % (2**31 - 1)),
        n_init=kmeans_n_init,
        algorithm="lloyd",
    )
    estimator.fit(unique_colors.astype(np.float32), sample_weight=unique_counts)
    unique_to_parent = canonicalize_cluster_labels(
        estimator.labels_, estimator.cluster_centers_
    )
    segment_flat = unique_to_parent[pixel_to_unique]
    parent_pixel_counts = np.bincount(
        segment_flat, minlength=n_clusters
    ).astype(np.int64)
    return ParentColorData(
        segments=segment_flat.reshape(image.shape[:2]).astype(np.int32, copy=False),
        unique_colors=unique_colors.astype(np.uint8, copy=False),
        pixel_to_unique=pixel_to_unique.astype(np.int32, copy=False),
        unique_counts=unique_counts.astype(np.int64, copy=False),
        unique_to_parent=unique_to_parent.astype(np.int32, copy=False),
        parent_pixel_counts=parent_pixel_counts,
    )


def canonical_group_order(
    groups: Sequence[np.ndarray], height: int, width: int
) -> list[np.ndarray]:
    del height
    decorated: list[tuple[float, float, int, np.ndarray]] = []
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=np.int64)
        ys = group // width
        xs = group % width
        decorated.append((float(ys.mean()), float(xs.mean()), int(group.min()), group))
    return [item[3] for item in sorted(decorated, key=lambda item: item[:3])]


def balanced_spatial_groups(
    flat_indices: np.ndarray,
    group_count: int,
    height: int,
    width: int,
) -> list[np.ndarray]:
    indices = np.asarray(flat_indices, dtype=np.int64)
    group_count = min(int(group_count), len(indices))
    if group_count < 1:
        return []
    if group_count == 1:
        return [indices]
    if group_count == 4:
        # The four-child design is recursive: first bisect the parent, then
        # recompute the spatial PCA axis and bisect each half.
        halves = balanced_spatial_groups(indices, 2, height, width)
        quarters = [
            quarter
            for half in halves
            for quarter in balanced_spatial_groups(half, 2, height, width)
        ]
        return canonical_group_order(quarters, height, width)
    y_scale = max(height - 1, 1)
    x_scale = max(width - 1, 1)
    coordinates = np.column_stack(
        ((indices % width) / x_scale, (indices // width) / y_scale)
    ).astype(np.float64)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(indices) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    largest_component = int(np.argmax(np.abs(axis)))
    if axis[largest_component] < 0.0:
        axis = -axis
    projection = centered @ axis
    order = np.lexsort((indices, projection))
    groups = [indices[chunk] for chunk in np.array_split(order, group_count)]
    return canonical_group_order(groups, height, width)


def split_groups_to_target(
    groups: Sequence[np.ndarray],
    target_count: int,
    height: int,
    width: int,
) -> list[np.ndarray]:
    output = [np.asarray(group, dtype=np.int64) for group in groups]
    while len(output) < target_count:
        splittable = [
            (len(group), -int(group.min()), index)
            for index, group in enumerate(output)
            if len(group) >= 2
        ]
        if not splittable:
            raise RuntimeError("Cannot create the requested number of nonempty children.")
        _, _, chosen_index = max(splittable)
        chosen = output.pop(chosen_index)
        output.extend(balanced_spatial_groups(chosen, 2, height, width))
    return canonical_group_order(output, height, width)


def assemble_hierarchical_segments(
    parent_segments: np.ndarray,
    subdivisions: int,
    group_builder: Callable[[int, np.ndarray, int], tuple[list[np.ndarray], bool]],
) -> tuple[np.ndarray, int, int]:
    height, width = parent_segments.shape
    flat_parent = parent_segments.reshape(-1)
    output = np.full(flat_parent.shape, -1, dtype=np.int32)
    next_feature = 0
    fallback_parents = 0
    fallback_pixels = 0
    n_parents = int(flat_parent.max()) + 1
    for parent_id in range(n_parents):
        parent_indices = np.flatnonzero(flat_parent == parent_id).astype(np.int64)
        target = min(subdivisions, len(parent_indices))
        groups, used_fallback = group_builder(parent_id, parent_indices, target)
        groups = canonical_group_order(groups, height, width)
        if len(groups) != target or any(len(group) == 0 for group in groups):
            raise AssertionError(
                f"Parent {parent_id}: expected {target} nonempty children, got {len(groups)}"
            )
        assigned = np.concatenate(groups)
        if len(assigned) != len(parent_indices) or not np.array_equal(
            np.sort(assigned), parent_indices
        ):
            raise AssertionError(f"Parent {parent_id}: child groups do not partition it")
        for group in groups:
            output[group] = next_feature
            next_feature += 1
        if used_fallback:
            fallback_parents += 1
            fallback_pixels += len(parent_indices)
    if np.any(output < 0):
        raise AssertionError("Some image pixels were not assigned to a child feature.")
    return output.reshape(parent_segments.shape), fallback_parents, fallback_pixels


def spatial_pca_subdivision(
    parent: ParentColorData, subdivisions: int
) -> tuple[np.ndarray, int, int]:
    height, width = parent.segments.shape

    def builder(
        parent_id: int, indices: np.ndarray, target: int
    ) -> tuple[list[np.ndarray], bool]:
        del parent_id
        return balanced_spatial_groups(indices, target, height, width), False

    return assemble_hierarchical_segments(parent.segments, subdivisions, builder)


def connected_components_8(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    mask_flat = mask.reshape(-1)
    visited = np.zeros(mask_flat.shape, dtype=bool)
    components: list[np.ndarray] = []
    for start in np.flatnonzero(mask_flat):
        start_int = int(start)
        if visited[start_int]:
            continue
        visited[start_int] = True
        queue: deque[int] = deque([start_int])
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            y, x = divmod(current, width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor = ny * width + nx
                        if mask_flat[neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def item_groups_from_kmeans(
    item_coordinates: np.ndarray,
    item_weights: np.ndarray,
    item_pixel_groups: Sequence[np.ndarray],
    group_count: int,
    seed: int,
    kmeans_n_init: int,
    KMeans: Any,
    height: int,
    width: int,
) -> tuple[list[np.ndarray], bool]:
    estimator = KMeans(
        n_clusters=group_count,
        random_state=int(seed % (2**31 - 1)),
        n_init=kmeans_n_init,
        algorithm="lloyd",
    )
    estimator.fit(item_coordinates.astype(np.float32), sample_weight=item_weights)
    labels = np.asarray(estimator.labels_, dtype=np.int32)
    if len(np.unique(labels)) != group_count:
        # Deterministic atomic repair: order item centroids spatially and split
        # the items into nonempty groups without breaking an item.
        pseudo_indices = np.arange(len(item_pixel_groups), dtype=np.int64)
        centered = item_coordinates - item_coordinates.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        order = np.lexsort((pseudo_indices, centered @ axis))
        item_chunks = np.array_split(order, group_count)
        groups = [
            np.concatenate([item_pixel_groups[int(item)] for item in chunk])
            for chunk in item_chunks
        ]
        return canonical_group_order(groups, height, width), True
    groups = [
        np.concatenate(
            [item_pixel_groups[item] for item in np.flatnonzero(labels == group_id)]
        )
        for group_id in range(group_count)
    ]
    return canonical_group_order(groups, height, width), False


def connected_component_subdivision(
    parent: ParentColorData,
    subdivisions: int,
    image_seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> tuple[np.ndarray, int, int]:
    height, width = parent.segments.shape

    def builder(
        parent_id: int, indices: np.ndarray, target: int
    ) -> tuple[list[np.ndarray], bool]:
        mask = np.zeros(height * width, dtype=bool)
        mask[indices] = True
        components = connected_components_8(mask.reshape(height, width))
        if len(components) < target:
            return (
                split_groups_to_target(components, target, height, width),
                True,
            )
        if len(components) == target:
            return canonical_group_order(components, height, width), False
        centroids = np.asarray(
            [
                [np.mean(component // width), np.mean(component % width)]
                for component in components
            ],
            dtype=np.float64,
        )
        weights = np.asarray([len(component) for component in components], dtype=np.float64)
        groups, repaired = item_groups_from_kmeans(
            centroids,
            weights,
            components,
            target,
            stable_seed(image_seed, "components", parent_id, bits=32),
            kmeans_n_init,
            KMeans,
            height,
            width,
        )
        return groups, repaired

    return assemble_hierarchical_segments(parent.segments, subdivisions, builder)


def vit_patch_subdivision(
    parent: ParentColorData,
    subdivisions: int,
    patch_size: tuple[int, int],
    image_seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> tuple[np.ndarray, int, int]:
    height, width = parent.segments.shape
    patch_height, patch_width = patch_size
    patch_columns = int(math.ceil(width / patch_width))

    def builder(
        parent_id: int, indices: np.ndarray, target: int
    ) -> tuple[list[np.ndarray], bool]:
        ys = indices // width
        xs = indices % width
        patch_ids = (ys // patch_height) * patch_columns + (xs // patch_width)
        occupied = np.unique(patch_ids)
        pixel_groups = [indices[patch_ids == patch_id] for patch_id in occupied]
        if len(occupied) < target:
            return (
                split_groups_to_target(pixel_groups, target, height, width),
                True,
            )
        if len(occupied) == target:
            return canonical_group_order(pixel_groups, height, width), False
        patch_coordinates = np.column_stack(
            (occupied // patch_columns, occupied % patch_columns)
        ).astype(np.float64)
        weights = np.asarray([len(group) for group in pixel_groups], dtype=np.float64)
        groups, repaired = item_groups_from_kmeans(
            patch_coordinates,
            weights,
            pixel_groups,
            target,
            stable_seed(image_seed, "vit_patch", parent_id, bits=32),
            kmeans_n_init,
            KMeans,
            height,
            width,
        )
        return groups, repaired

    return assemble_hierarchical_segments(parent.segments, subdivisions, builder)


def box_mean(array: np.ndarray, window: int) -> np.ndarray:
    """Fast reflect-padded local mean using an integral image."""
    if window % 2 != 1 or window < 1:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(np.asarray(array, dtype=np.float64), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ) / float(window * window)


def convolve3_reflect(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if weights.shape != (3, 3):
        raise ValueError("Only 3x3 kernels are supported")
    height, width = values.shape
    padded = np.pad(values, 1, mode="reflect")
    output = np.zeros((height, width), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            output += weights[row, column] * padded[
                row : row + height, column : column + width
            ]
    return output


def texture_descriptors(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    sobel_x = convolve3_reflect(
        gray, np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    )
    sobel_y = convolve3_reflect(
        gray, np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    )
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    laplacian = np.abs(
        convolve3_reflect(
            gray, np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        )
    )
    mean3 = box_mean(gray, 3)
    mean7 = box_mean(gray, 7)
    std3 = np.sqrt(np.maximum(box_mean(gray * gray, 3) - mean3 * mean3, 0.0))
    std7 = np.sqrt(np.maximum(box_mean(gray * gray, 7) - mean7 * mean7, 0.0))
    return np.stack((std3, std7, gradient_magnitude, laplacian), axis=-1).astype(
        np.float32
    )


def texture_subdivision(
    parent: ParentColorData,
    descriptors: np.ndarray,
    subdivisions: int,
    image_seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> tuple[np.ndarray, int, int]:
    height, width = parent.segments.shape
    flat_descriptors = descriptors.reshape(-1, descriptors.shape[-1])

    def builder(
        parent_id: int, indices: np.ndarray, target: int
    ) -> tuple[list[np.ndarray], bool]:
        if target == 1:
            return [indices], False
        values = flat_descriptors[indices].astype(np.float64, copy=True)
        standard_deviation = values.std(axis=0)
        varying = standard_deviation > 1e-12
        if not np.any(varying):
            return balanced_spatial_groups(indices, target, height, width), True
        values = values[:, varying]
        values = (values - values.mean(axis=0)) / values.std(axis=0)
        if len(np.unique(np.round(values, decimals=8), axis=0)) < target:
            return balanced_spatial_groups(indices, target, height, width), True
        estimator = KMeans(
            n_clusters=target,
            random_state=int(
                stable_seed(image_seed, "texture", parent_id, bits=32) % (2**31 - 1)
            ),
            n_init=kmeans_n_init,
            algorithm="lloyd",
        )
        estimator.fit(values.astype(np.float32))
        labels = np.asarray(estimator.labels_, dtype=np.int32)
        if len(np.unique(labels)) != target:
            return balanced_spatial_groups(indices, target, height, width), True
        groups = [indices[labels == child] for child in range(target)]
        return canonical_group_order(groups, height, width), False

    return assemble_hierarchical_segments(parent.segments, subdivisions, builder)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB values to CIE L*a*b* under a D65 white point."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz /= np.asarray([0.95047, 1.00000, 1.08883], dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    ).astype(np.float32)


def fine_shade_subdivision(
    parent: ParentColorData,
    subdivisions: int,
    image_seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> tuple[np.ndarray, int, int]:
    height, width = parent.segments.shape
    unique_lab = rgb_to_lab(parent.unique_colors)
    flat_unique_ids = parent.pixel_to_unique

    def builder(
        parent_id: int, indices: np.ndarray, target: int
    ) -> tuple[list[np.ndarray], bool]:
        unique_ids = np.flatnonzero(parent.unique_to_parent == parent_id)
        if len(unique_ids) < target:
            natural_groups = [
                indices[flat_unique_ids[indices] == unique_id]
                for unique_id in unique_ids
            ]
            return (
                split_groups_to_target(natural_groups, target, height, width),
                True,
            )
        estimator = KMeans(
            n_clusters=target,
            random_state=int(
                stable_seed(image_seed, "fine_shade", parent_id, bits=32)
                % (2**31 - 1)
            ),
            n_init=kmeans_n_init,
            algorithm="lloyd",
        )
        estimator.fit(
            unique_lab[unique_ids],
            sample_weight=parent.unique_counts[unique_ids],
        )
        unique_child_labels = np.asarray(estimator.labels_, dtype=np.int32)
        if len(np.unique(unique_child_labels)) != target:
            return balanced_spatial_groups(indices, target, height, width), True
        lookup = np.full(len(parent.unique_colors), -1, dtype=np.int32)
        lookup[unique_ids] = unique_child_labels
        pixel_labels = lookup[flat_unique_ids[indices]]
        groups = [indices[pixel_labels == child] for child in range(target)]
        return canonical_group_order(groups, height, width), False

    return assemble_hierarchical_segments(parent.segments, subdivisions, builder)


def hierarchical_ideal_feature_count(
    parent: ParentColorData, subdivisions: int
) -> int:
    return int(np.minimum(parent.parent_pixel_counts, subdivisions).sum())


def build_color_segmentations(
    image: np.ndarray,
    k_colors: int,
    subdivisions: int,
    patch_size: tuple[int, int],
    global_seed: int,
    dataset_index: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> dict[str, SegmentationResult]:
    image_seed = stable_seed(global_seed, dataset_index, "segmentations", bits=32)
    segmentations: dict[str, SegmentationResult] = {}

    started = time.perf_counter()
    parent = build_parent_color_data(
        image,
        k_colors,
        stable_seed(image_seed, "parent_kmeans", bits=32),
        kmeans_n_init,
        KMeans,
    )
    segmentations["color_parent"] = SegmentationResult(
        method_key="color_parent",
        segments=parent.segments,
        construction_seconds=time.perf_counter() - started,
        parent_count=parent.n_parents,
        requested_children=1,
        ideal_feature_count=parent.n_parents,
    )

    direct_k = k_colors * subdivisions
    started = time.perf_counter()
    direct = build_parent_color_data(
        image,
        direct_k,
        stable_seed(image_seed, "direct_kmeans", bits=32),
        kmeans_n_init,
        KMeans,
    )
    segmentations["color_direct"] = SegmentationResult(
        method_key="color_direct",
        segments=direct.segments,
        construction_seconds=time.perf_counter() - started,
        parent_count=direct.n_parents,
        requested_children=1,
        ideal_feature_count=direct.n_parents,
        metadata={"direct_k_requested": direct_k},
    )

    ideal_hierarchical = hierarchical_ideal_feature_count(parent, subdivisions)

    started = time.perf_counter()
    segments, fallback_parents, fallback_pixels = spatial_pca_subdivision(
        parent, subdivisions
    )
    segmentations["spatial_pca"] = SegmentationResult(
        "spatial_pca",
        segments,
        time.perf_counter() - started,
        parent.n_parents,
        subdivisions,
        ideal_hierarchical,
        fallback_parents,
        fallback_pixels,
    )

    started = time.perf_counter()
    segments, fallback_parents, fallback_pixels = connected_component_subdivision(
        parent, subdivisions, image_seed, kmeans_n_init, KMeans
    )
    segmentations["connected_components"] = SegmentationResult(
        "connected_components",
        segments,
        time.perf_counter() - started,
        parent.n_parents,
        subdivisions,
        ideal_hierarchical,
        fallback_parents,
        fallback_pixels,
        {"connectivity": 8},
    )

    started = time.perf_counter()
    segments, fallback_parents, fallback_pixels = vit_patch_subdivision(
        parent,
        subdivisions,
        patch_size,
        image_seed,
        kmeans_n_init,
        KMeans,
    )
    segmentations["vit_patch"] = SegmentationResult(
        "vit_patch",
        segments,
        time.perf_counter() - started,
        parent.n_parents,
        subdivisions,
        ideal_hierarchical,
        fallback_parents,
        fallback_pixels,
        {"patch_height": patch_size[0], "patch_width": patch_size[1]},
    )

    started = time.perf_counter()
    descriptors = texture_descriptors(image)
    segments, fallback_parents, fallback_pixels = texture_subdivision(
        parent,
        descriptors,
        subdivisions,
        image_seed,
        kmeans_n_init,
        KMeans,
    )
    segmentations["texture"] = SegmentationResult(
        "texture",
        segments,
        time.perf_counter() - started,
        parent.n_parents,
        subdivisions,
        ideal_hierarchical,
        fallback_parents,
        fallback_pixels,
        {"descriptors": "local_std_3,local_std_7,sobel_magnitude,abs_laplacian"},
    )

    started = time.perf_counter()
    segments, fallback_parents, fallback_pixels = fine_shade_subdivision(
        parent, subdivisions, image_seed, kmeans_n_init, KMeans
    )
    segmentations["fine_shade"] = SegmentationResult(
        "fine_shade",
        segments,
        time.perf_counter() - started,
        parent.n_parents,
        subdivisions,
        ideal_hierarchical,
        fallback_parents,
        fallback_pixels,
        {"color_space": "CIELAB_D65"},
    )

    hierarchical_counts = {
        segmentations[key].n_features for key in HIERARCHICAL_KEYS
    }
    if len(hierarchical_counts) != 1:
        raise AssertionError(
            "Hierarchical methods produced unequal feature counts: "
            f"{sorted(hierarchical_counts)}"
        )
    return segmentations


def make_binary_perturbations(
    num_samples: int, n_features: int, seed: int
) -> np.ndarray:
    rng = np.random.RandomState(int(seed % (2**31 - 1)))
    data = rng.randint(0, 2, size=(num_samples, n_features)).astype(np.int8)
    data[0, :] = 1
    return data


def cosine_distances_from_all_ones(data: np.ndarray) -> np.ndarray:
    n_features = data.shape[1]
    on_counts = data.sum(axis=1, dtype=np.float64)
    distances = 1.0 - np.sqrt(on_counts / float(n_features))
    distances[on_counts == 0] = 1.0
    distances[0] = 0.0
    return distances


def fit_lime_surrogate(
    binary_data: np.ndarray,
    probabilities: np.ndarray,
    target_class: int,
    seed: int,
    LimeBase: Any,
) -> tuple[list[tuple[int, float]], float]:
    kernel_width = 0.25

    def kernel(distances: np.ndarray) -> np.ndarray:
        return np.sqrt(np.exp(-(distances**2) / (kernel_width**2)))

    base = LimeBase(
        kernel_fn=kernel,
        verbose=False,
        random_state=int(seed % (2**31 - 1)),
    )
    distances = cosine_distances_from_all_ones(binary_data)
    _, local_exp, score, _ = base.explain_instance_with_data(
        binary_data.astype(np.float64),
        probabilities,
        distances,
        int(target_class),
        num_features=binary_data.shape[1],
        feature_selection="auto",
    )
    return [(int(feature), float(weight)) for feature, weight in local_exp], float(score)


def explain_default_lime(
    image: np.ndarray,
    target_class: int,
    num_samples: int,
    batch_size: int,
    image_seed: int,
    predictor: ViTPredictor,
    lime_image: Any,
) -> ExplanationResult:
    started = time.perf_counter()
    seed32 = int(image_seed % (2**31 - 1))
    explainer = lime_image.LimeImageExplainer(random_state=seed32)
    arguments: dict[str, Any] = {
        "image": image,
        "classifier_fn": predictor.predict_proba,
        "labels": (int(target_class),),
        "top_labels": None,
        "hide_color": None,
        "num_features": 100000,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "random_seed": seed32,
    }
    # Older LIME versions reject progress_bar.  Pass it only when supported.
    signature = inspect.signature(explainer.explain_instance)
    if "progress_bar" in signature.parameters:
        arguments["progress_bar"] = False
    explanation = explainer.explain_instance(**arguments)
    weights = [
        (int(feature), float(weight))
        for feature, weight in explanation.local_exp[int(target_class)]
    ]
    score_container = getattr(explanation, "score", {})
    if isinstance(score_container, dict):
        score = float(score_container.get(int(target_class), float("nan")))
    else:
        score = float(score_container)
    return ExplanationResult(
        "default_lime",
        np.asarray(explanation.segments, dtype=np.int32),
        weights,
        score,
        time.perf_counter() - started,
    )


def run_color_explanations(
    image: np.ndarray,
    segmentations: dict[str, SegmentationResult],
    target_class: int,
    num_samples: int,
    batch_size: int,
    global_seed: int,
    dataset_index: int,
    predictor: ViTPredictor,
    LimeBase: Any,
) -> dict[str, ExplanationResult]:
    """Share masks/prediction calls among methods with the same feature count."""
    grouped: dict[int, list[SegmentationResult]] = {}
    for result in segmentations.values():
        grouped.setdefault(result.n_features, []).append(result)
    explanations: dict[str, ExplanationResult] = {}
    for feature_count, group in sorted(grouped.items()):
        binary_data = make_binary_perturbations(
            num_samples,
            feature_count,
            stable_seed(
                global_seed,
                dataset_index,
                "shared_color_binary",
                feature_count,
                bits=32,
            ),
        )
        probability_chunks: dict[str, list[np.ndarray]] = {
            item.method_key: [] for item in group
        }
        prediction_started = time.perf_counter()
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            binary_chunk = binary_data[start:end]
            synthesized: list[np.ndarray] = []
            for item in group:
                on_pixels = binary_chunk[:, item.segments].astype(bool, copy=False)
                synthesized.append(
                    np.where(on_pixels[..., None], image[None, ...], 0).astype(
                        np.uint8, copy=False
                    )
                )
            combined = np.concatenate(synthesized, axis=0)
            combined_probabilities = predictor.predict_proba(combined)
            chunk_length = end - start
            for method_number, item in enumerate(group):
                left = method_number * chunk_length
                probability_chunks[item.method_key].append(
                    combined_probabilities[left : left + chunk_length]
                )
            del synthesized, combined, combined_probabilities
        shared_prediction_seconds = time.perf_counter() - prediction_started
        shared_seed = stable_seed(
            global_seed,
            dataset_index,
            "shared_surrogate",
            feature_count,
            bits=32,
        )
        for item in group:
            fit_started = time.perf_counter()
            probabilities = np.concatenate(probability_chunks[item.method_key], axis=0)
            weights, score = fit_lime_surrogate(
                binary_data,
                probabilities,
                target_class,
                shared_seed,
                LimeBase,
            )
            explanations[item.method_key] = ExplanationResult(
                item.method_key,
                item.segments,
                weights,
                score,
                item.construction_seconds
                + shared_prediction_seconds / len(group)
                + (time.perf_counter() - fit_started),
            )
        del binary_data, probability_chunks
    return explanations


def positive_feature_ids(weights: Sequence[tuple[int, float]]) -> list[int]:
    return [
        feature
        for feature, weight in sorted(weights, key=lambda item: item[1], reverse=True)
        if weight > 0.0
    ]


def mask_for_positive_fraction(
    segments: np.ndarray,
    weights: Sequence[tuple[int, float]],
    fraction: float,
) -> tuple[np.ndarray, int, int]:
    positive = positive_feature_ids(weights)
    if not positive:
        return np.zeros(segments.shape, dtype=bool), 0, 0
    selected_count = max(1, int(math.ceil(len(positive) * fraction)))
    return np.isin(segments, positive[:selected_count]), selected_count, len(positive)


def mask_for_critical_area(
    segments: np.ndarray,
    weights: Sequence[tuple[int, float]],
    area_fraction: float,
) -> tuple[np.ndarray, int, int]:
    positive = positive_feature_ids(weights)
    mask = np.zeros(segments.shape, dtype=bool)
    if not positive:
        return mask, 0, 0
    target_pixels = int(math.ceil(mask.size * area_fraction))
    selected = 0
    for feature in positive:
        mask |= segments == feature
        selected += 1
        if int(mask.sum()) >= target_pixels:
            break
    return mask, selected, len(positive)


def black_delete(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    deleted = image.copy()
    deleted[mask] = 0
    return deleted


def white_cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cutout = np.full_like(image, 255)
    cutout[mask] = image[mask]
    return cutout


def green_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    if mask.any():
        source = output[mask].astype(np.float32)
        green = np.asarray([55.0, 220.0, 80.0], dtype=np.float32)
        output[mask] = np.clip(
            np.rint(0.68 * source + 0.32 * green), 0, 255
        ).astype(np.uint8)
    return output


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_result_rows(path: Path, valid_dataset_indices: set[int]) -> list[dict[str, Any]]:
    """Keep only complete eight-method image groups from a prior run."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        try:
            dataset_index = int(row["dataset_index"])
        except (KeyError, TypeError, ValueError):
            continue
        method_key = str(row.get("method_key", ""))
        if dataset_index in valid_dataset_indices and method_key in METHOD_BY_KEY:
            latest[(dataset_index, method_key)] = row
    expected = set(METHOD_BY_KEY)
    complete_indices = {
        dataset_index
        for dataset_index in valid_dataset_indices
        if {
            method_key
            for row_index, method_key in latest
            if row_index == dataset_index
        }
        == expected
    }
    return [
        latest[(dataset_index, method.key)]
        for dataset_index in sorted(complete_indices)
        for method in METHODS
    ]


def completed_dataset_indices(rows: Sequence[dict[str, Any]]) -> set[int]:
    grouped: dict[int, set[str]] = {}
    for row in rows:
        grouped.setdefault(int(row["dataset_index"]), set()).add(str(row["method_key"]))
    expected = set(METHOD_BY_KEY)
    return {index for index, keys in grouped.items() if keys == expected}


def mask_cache_path(output_dir: Path, record: ImageRecord) -> Path:
    return (
        output_dir
        / "mask_cache"
        / f"image_{record.image_position:03d}_dataset_{record.dataset_index:06d}_masks.npz"
    )


def comparison_sheet_path(output_dir: Path, record: ImageRecord) -> Path:
    return (
        output_dir
        / "comparison_sheets"
        / (
            f"image_{record.image_position:03d}_dataset_{record.dataset_index:06d}_"
            "subdivision_comparison.png"
        )
    )


def save_top_mask_cache(
    path: Path, masks: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    ordered = np.stack(
        [mask for method in METHODS for mask in masks[method.key]], axis=0
    ).astype(bool, copy=False)
    height, width = ordered.shape[1:]
    packed = np.packbits(ordered.reshape(len(METHODS) * 2, -1), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            packed_masks=packed,
            image_shape=np.asarray([height, width], dtype=np.int32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_top_mask_cache(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        packed = np.asarray(payload["packed_masks"], dtype=np.uint8)
        height, width = [int(value) for value in payload["image_shape"]]
    unpacked = np.unpackbits(packed, axis=1, count=height * width).reshape(
        len(METHODS) * 2, height, width
    )
    return {
        method.key: (
            unpacked[number * 2].astype(bool, copy=False),
            unpacked[number * 2 + 1].astype(bool, copy=False),
        )
        for number, method in enumerate(METHODS)
    }


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_panel(image: np.ndarray, width: int, height: int) -> Image.Image:
    panel = Image.fromarray(image, mode="RGB")
    if panel.size != (width, height):
        panel = panel.resize((width, height), Image.Resampling.BILINEAR)
    return panel


def save_comparison_sheet(record: VisualRecord, output_path: Path) -> None:
    panel_width, panel_height = 224, 224
    row_label_width = 255
    header_height = 118
    column_header_height = 42
    row_height = panel_height + 8
    columns = (
        "Original",
        "Top 10% positive",
        "Top 20% positive",
        "Top 10% on white",
        "Top 20% on white",
    )
    width = row_label_width + len(columns) * panel_width
    height = header_height + column_header_height + len(METHODS) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    regular_font = load_font(15)
    label_font = load_font(15, bold=True)
    small_font = load_font(13)
    column_font = load_font(14, bold=True)
    draw.text(
        (16, 12),
        (
            f"Correct image {record.image_position:03d} | dataset index "
            f"{record.dataset_index} | confidence {record.predicted_confidence:.4f}"
        ),
        fill="black",
        font=title_font,
    )
    draw.text((16, 49), f"Ground truth: {record.label_name}", fill="black", font=regular_font)
    draw.text((16, 75), f"Prediction: {record.predicted_label}", fill="black", font=regular_font)
    for column_index, column_name in enumerate(columns):
        x = row_label_width + column_index * panel_width
        draw.rectangle(
            (x, header_height, x + panel_width, header_height + column_header_height),
            fill=(238, 242, 247),
        )
        draw.text((x + 8, header_height + 11), column_name, fill="black", font=column_font)
    for row_index, method in enumerate(METHODS):
        y = header_height + column_header_height + row_index * row_height
        draw.rectangle((0, y, row_label_width, y + panel_height), fill=(247, 247, 247))
        label_lines = method.display_name.replace(" — ", "\n")
        draw.multiline_text((12, y + 66), label_lines, fill="black", font=label_font, spacing=5)
        draw.text(
            (12, y + 142),
            f"Features: {record.feature_counts.get(method.key, 0)}",
            fill=(65, 65, 65),
            font=small_font,
        )
        mask10, mask20 = record.masks[method.key]
        panels = (
            record.image,
            green_overlay(record.image, mask10),
            green_overlay(record.image, mask20),
            white_cutout(record.image, mask10),
            white_cutout(record.image, mask20),
        )
        for column_index, panel_array in enumerate(panels):
            x = row_label_width + column_index * panel_width
            canvas.paste(fit_panel(panel_array, panel_width, panel_height), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    canvas.save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, output_path)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method_key"] == method.key]
        if not selected:
            continue
        cir = [float(row["cir"]) for row in selected]
        decision = [int(row["dir"]) for row in selected]
        area = [float(row["critical_area_fraction_actual"]) for row in selected]
        feature_counts = [int(row["feature_count"]) for row in selected]
        fallback_parents = [int(row["fallback_parent_count"]) for row in selected]
        fallback_pixel_fraction = [
            float(row["fallback_pixel_fraction"]) for row in selected
        ]
        feature_size_cv = [float(row["feature_pixel_count_cv"]) for row in selected]
        maximum_feature_area = [
            float(row["maximum_feature_area_fraction"]) for row in selected
        ]
        scores = np.asarray([float(row["surrogate_score"]) for row in selected])
        runtimes = [float(row["explanation_runtime_seconds"]) for row in selected]
        finite_scores = scores[np.isfinite(scores)]
        summary.append(
            {
                "method_key": method.key,
                "method": method.display_name,
                "n_correct_images": len(selected),
                "feature_count_mean": float(np.mean(feature_counts)),
                "fallback_parent_count_mean": float(np.mean(fallback_parents)),
                "fallback_pixel_fraction_mean": float(
                    np.mean(fallback_pixel_fraction)
                ),
                "feature_pixel_count_cv_mean": float(np.mean(feature_size_cv)),
                "maximum_feature_area_fraction_mean": float(
                    np.mean(maximum_feature_area)
                ),
                "cir_mean": float(np.mean(cir)),
                "cir_median": float(np.median(cir)),
                "cir_std": float(np.std(cir, ddof=1)) if len(cir) > 1 else 0.0,
                "cir_q1": percentile(cir, 25),
                "cir_q3": percentile(cir, 75),
                "cir_min": float(np.min(cir)),
                "cir_max": float(np.max(cir)),
                "dir_ratio": float(np.mean(decision)),
                "decision_changes": int(np.sum(decision)),
                "critical_area_fraction_mean": float(np.mean(area)),
                "surrogate_score_mean": float(np.mean(finite_scores))
                if finite_scores.size
                else float("nan"),
                "runtime_seconds_mean": float(np.mean(runtimes)),
                "runtime_seconds_total": float(np.sum(runtimes)),
            }
        )
    return summary


def plot_colors() -> tuple[str, ...]:
    return (
        "#8DA0CB",
        "#66C2A5",
        "#FC8D62",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
    )


def save_cir_boxplot(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [
        [float(row["cir"]) for row in rows if row["method_key"] == method.key]
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(15, 7))
    boxplot_arguments: dict[str, Any] = {
        "patch_artist": True,
        "showmeans": True,
        "meanprops": {
            "marker": "D",
            "markerfacecolor": "black",
            "markersize": 5,
        },
    }
    label_argument = (
        "tick_labels"
        if "tick_labels" in inspect.signature(axis.boxplot).parameters
        else "labels"
    )
    boxplot_arguments[label_argument] = [method.short_name for method in METHODS]
    box = axis.boxplot(data, **boxplot_arguments)
    for patch, color in zip(box["boxes"], plot_colors()):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    axis.set_title("CIR comparison — black deletion for every method", fontsize=15)
    axis.set_ylabel("Nonnegative target-confidence drop")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_dir_plot(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = [
        float(
            np.mean(
                [int(row["dir"]) for row in rows if row["method_key"] == method.key]
            )
        )
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(15, 7))
    positions = np.arange(len(METHODS))
    bars = axis.bar(positions, ratios, color=plot_colors(), alpha=0.85)
    for bar, value in zip(bars, ratios):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.02),
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [method.short_name for method in METHODS], rotation=20
    )
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Fraction whose predicted class changed")
    axis.set_title("DIR comparison — black deletion for every method", fontsize=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_plots(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    save_cir_boxplot(rows, output_dir / "cir_boxplot.png")
    save_dir_plot(rows, output_dir / "dir_barplot.png")


def save_config(
    args: argparse.Namespace,
    output_dir: Path,
    device: Any,
    patch_size: tuple[int, int],
) -> None:
    payload = vars(args).copy()
    payload["output_dir"] = str(output_dir)
    payload["device_resolved"] = str(device)
    payload["vit_patch_size_resolved"] = list(patch_size)
    payload["direct_k_control"] = args.k_colors * args.subdivisions
    payload["methods"] = [asdict(method) for method in METHODS]
    payload["fairness_rule"] = (
        "Each hierarchical parent has min(subdivisions, parent_pixel_count) "
        "nonempty children; deterministic spatial fallback preserves feature-count parity."
    )
    payload["color_cluster_postprocessing"] = (
        "Raw weighted RGB KMeans labels are retained; centroids are not rounded "
        "to uint8 and duplicate rounded centroids are not collapsed."
    )
    atomic_write_json(output_dir / "run_config.json", payload)


def validate_resume_config(args: argparse.Namespace, output_dir: Path) -> None:
    config_path = output_dir / "run_config.json"
    if not args.resume:
        return
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: {config_path} is missing. Use a new output directory."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    compatibility_keys = (
        "target_correct",
        "start_index",
        "max_scan",
        "num_samples",
        "k_colors",
        "subdivisions",
        "critical_area_fraction",
        "seed",
        "kmeans_n_init",
        "model_id",
        "dataset_name",
        "dataset_split",
        "image_column",
        "label_column",
        "streaming",
        "precision",
        "fast_gpu_preprocess",
    )
    current = vars(args)
    mismatches = [
        key for key in compatibility_keys if key in saved and saved[key] != current[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved[key]!r}, requested={current[key]!r}"
            for key in mismatches
        )
        raise ValueError(
            "Cannot resume because experiment-defining settings changed: " + details
        )


def save_checkpoint(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    newly_completed: int,
    total_complete: int,
    target_correct: int,
    last_record: ImageRecord,
) -> None:
    atomic_write_csv(output_dir / "summary.csv", summarize_results(rows))
    save_plots(rows, output_dir)
    atomic_write_json(
        output_dir / "checkpoint_state.json",
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "newly_completed_this_run": newly_completed,
            "complete_correct_images": total_complete,
            "target_correct_images": target_correct,
            "last_correct_position": last_record.image_position,
            "last_dataset_index": last_record.dataset_index,
            "results_csv": str(output_dir / "results.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "cir_plot": str(output_dir / "cir_boxplot.png"),
            "dir_plot": str(output_dir / "dir_barplot.png"),
        },
    )


def feature_counts_for_record(
    rows: Sequence[dict[str, Any]], dataset_index: int
) -> dict[str, int]:
    return {
        str(row["method_key"]): int(row["feature_count"])
        for row in rows
        if int(row["dataset_index"]) == dataset_index
    }


def render_cached_sheet_if_possible(
    args: argparse.Namespace,
    output_dir: Path,
    record: ImageRecord,
    rows: Sequence[dict[str, Any]],
) -> bool:
    if args.no_comparison_sheets:
        return True
    sheet_path = comparison_sheet_path(output_dir, record)
    if sheet_path.exists():
        return True
    cache_path = mask_cache_path(output_dir, record)
    if not cache_path.exists():
        return False
    save_comparison_sheet(
        VisualRecord(
            record.image_position,
            record.dataset_index,
            record.label_name,
            record.predicted_label,
            record.predicted_confidence,
            record.image,
            load_top_mask_cache(cache_path),
            feature_counts_for_record(rows, record.dataset_index),
        ),
        sheet_path,
    )
    return True


def run_experiment(args: argparse.Namespace) -> None:
    deps = import_experiment_dependencies()
    torch = deps["torch"]
    if args.num_samples != 1000:
        print(
            f"Note: --num-samples={args.num_samples}; the main experiment uses 1000.",
            flush=True,
        )
    if args.k_colors != 100 or args.subdivisions != 3:
        print(
            "Note: the requested starting experiment uses --k-colors 100 "
            "--subdivisions 3.",
            flush=True,
        )
    output_dir = prepare_output_dir(args.output_dir, args.overwrite, args.resume)
    validate_resume_config(args, output_dir)
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Loading {args.model_id} on {device} ...", flush=True)
    processor = deps["AutoImageProcessor"].from_pretrained(args.model_id)
    model = deps["AutoModelForImageClassification"].from_pretrained(args.model_id)
    model.eval().to(device)
    patch_size = model_patch_size(model)
    image_height, image_width = processor_image_size(processor)
    patch_rows = math.ceil(image_height / patch_size[0])
    patch_columns = math.ceil(image_width / patch_size[1])
    print(
        f"ViT input {image_width}x{image_height}; native patch size "
        f"{patch_size[1]}x{patch_size[0]} ({patch_rows * patch_columns} patch tokens).",
        flush=True,
    )
    predictor = ViTPredictor(
        torch,
        processor,
        model,
        device,
        args.model_microbatch,
        args.precision,
        args.fast_gpu_preprocess,
    )
    save_config(args, output_dir, device, patch_size)

    print(
        f"Scanning ImageNet until {args.target_correct} correct predictions are found ...",
        flush=True,
    )
    dataset = load_dataset_split(args, deps["load_dataset"])
    records, prediction_rows = select_correct_records(
        args, dataset, processor, predictor, model
    )
    atomic_write_csv(output_dir / "prediction_summary.csv", prediction_rows)
    atomic_write_json(
        output_dir / "selection_summary.json",
        {
            "target_correct": args.target_correct,
            "screened_count": len(prediction_rows),
            "accuracy_within_screened_prefix": args.target_correct / len(prediction_rows),
            "selected_dataset_indices": [record.dataset_index for record in records],
        },
    )
    print(
        f"Selected exactly {len(records)} correct images after screening "
        f"{len(prediction_rows)} examples.",
        flush=True,
    )

    valid_indices = {record.dataset_index for record in records}
    all_rows = (
        load_result_rows(output_dir / "results.csv", valid_indices)
        if args.resume
        else []
    )
    completed_indices = completed_dataset_indices(all_rows)
    if all_rows:
        atomic_write_csv(output_dir / "results.csv", all_rows)
        print(
            f"Resume found {len(completed_indices)}/{len(records)} complete images.",
            flush=True,
        )

    newly_completed = 0
    for record in records:
        if record.dataset_index in completed_indices and render_cached_sheet_if_possible(
            args, output_dir, record, all_rows
        ):
            print(
                f"[{record.image_position}/{len(records)} correct] dataset index "
                f"{record.dataset_index} — resumed",
                flush=True,
            )
            continue
        if record.dataset_index in completed_indices:
            # Numerical results without the requested mask cache cannot recreate
            # the visual sheet, so only this image is recomputed.
            all_rows = [
                row for row in all_rows if int(row["dataset_index"]) != record.dataset_index
            ]
            completed_indices.discard(record.dataset_index)

        print(
            f"[{record.image_position}/{len(records)} correct] dataset index "
            f"{record.dataset_index}",
            flush=True,
        )
        explanations: dict[str, ExplanationResult] = {}
        default_result = explain_default_lime(
            record.image,
            record.predicted_class,
            args.num_samples,
            args.lime_batch_size,
            stable_seed(args.seed, record.dataset_index, "default_lime", bits=32),
            predictor,
            deps["lime_image"],
        )
        explanations[default_result.method_key] = default_result

        segmentations = build_color_segmentations(
            record.image,
            args.k_colors,
            args.subdivisions,
            patch_size,
            args.seed,
            record.dataset_index,
            args.kmeans_n_init,
            deps["KMeans"],
        )
        explanations.update(
            run_color_explanations(
                record.image,
                segmentations,
                record.predicted_class,
                args.num_samples,
                args.lime_batch_size,
                args.seed,
                record.dataset_index,
                predictor,
                deps["LimeBase"],
            )
        )

        deletion_images: list[np.ndarray] = []
        masks_by_method: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]
        ] = {}
        for method in METHODS:
            explanation = explanations[method.key]
            mask10, selected10, _ = mask_for_positive_fraction(
                explanation.segments, explanation.weights, 0.10
            )
            mask20, selected20, _ = mask_for_positive_fraction(
                explanation.segments, explanation.weights, 0.20
            )
            critical_mask, critical_selected, _ = mask_for_critical_area(
                explanation.segments,
                explanation.weights,
                args.critical_area_fraction,
            )
            masks_by_method[method.key] = (
                mask10,
                mask20,
                critical_mask,
                selected10,
                selected20,
                critical_selected,
            )
            deletion_images.append(black_delete(record.image, critical_mask))

        deletion_probabilities = predictor.predict_proba(deletion_images)
        image_rows: list[dict[str, Any]] = []
        feature_counts: dict[str, int] = {}
        for method_number, method in enumerate(METHODS):
            explanation = explanations[method.key]
            mask10, mask20, critical_mask, selected10, selected20, critical_selected = (
                masks_by_method[method.key]
            )
            deleted_probabilities = deletion_probabilities[method_number]
            deleted_target_confidence = float(
                deleted_probabilities[record.predicted_class]
            )
            deleted_prediction = int(np.argmax(deleted_probabilities))
            _, feature_pixel_counts_integer = np.unique(
                explanation.segments, return_counts=True
            )
            feature_count = int(len(feature_pixel_counts_integer))
            feature_counts[method.key] = feature_count
            feature_pixel_counts = feature_pixel_counts_integer.astype(np.float64)
            feature_pixel_count_mean = float(feature_pixel_counts.mean())
            feature_pixel_count_cv = (
                float(feature_pixel_counts.std() / feature_pixel_count_mean)
                if feature_pixel_count_mean > 0.0
                else 0.0
            )
            segmentation = segmentations.get(method.key)
            fallback_parent_count = (
                segmentation.fallback_parent_count if segmentation is not None else 0
            )
            fallback_pixel_count = (
                segmentation.fallback_pixel_count if segmentation is not None else 0
            )
            ideal_feature_count: int | str = (
                segmentation.ideal_feature_count if segmentation is not None else ""
            )
            parent_count: int | str = (
                segmentation.parent_count if segmentation is not None else ""
            )
            requested_children: int | str = (
                segmentation.requested_children if segmentation is not None else ""
            )
            positive_total = len(positive_feature_ids(explanation.weights))
            image_rows.append(
                {
                    "image_position": record.image_position,
                    "dataset_index": record.dataset_index,
                    "ground_truth_class": record.label,
                    "ground_truth_label": record.label_name,
                    "predicted_class": record.predicted_class,
                    "predicted_label": record.predicted_label,
                    "original_target_confidence": record.predicted_confidence,
                    "method_key": method.key,
                    "method": method.display_name,
                    "num_samples": args.num_samples,
                    "k_requested": (
                        args.k_colors * args.subdivisions
                        if method.key == "color_direct"
                        else args.k_colors
                        if method.is_color_lime
                        else ""
                    ),
                    "parent_k_requested": (
                        args.k_colors
                        if method.key == "color_parent" or method.is_hierarchical
                        else ""
                    ),
                    "subdivisions_requested": requested_children,
                    "parent_count_actual": parent_count,
                    "ideal_feature_count": ideal_feature_count,
                    "feature_count": feature_count,
                    "feature_pixel_count_min": int(feature_pixel_counts.min()),
                    "feature_pixel_count_median": float(
                        np.median(feature_pixel_counts)
                    ),
                    "feature_pixel_count_max": int(feature_pixel_counts.max()),
                    "feature_pixel_count_cv": feature_pixel_count_cv,
                    "maximum_feature_area_fraction": float(
                        feature_pixel_counts.max() / critical_mask.size
                    ),
                    "fallback_parent_count": fallback_parent_count,
                    "fallback_pixel_count": fallback_pixel_count,
                    "fallback_pixel_fraction": fallback_pixel_count / critical_mask.size,
                    "positive_feature_count": positive_total,
                    "top10_selected_feature_count": selected10,
                    "top20_selected_feature_count": selected20,
                    "top10_area_fraction": float(mask10.mean()),
                    "top20_area_fraction": float(mask20.mean()),
                    "critical_selected_feature_count": critical_selected,
                    "critical_area_fraction_target": args.critical_area_fraction,
                    "critical_area_fraction_actual": float(critical_mask.mean()),
                    "deleted_target_confidence": deleted_target_confidence,
                    "deleted_predicted_class": deleted_prediction,
                    "cir": max(
                        0.0,
                        record.predicted_confidence - deleted_target_confidence,
                    ),
                    "dir": int(deleted_prediction != record.predicted_class),
                    "surrogate_score": explanation.surrogate_score,
                    "segmentation_runtime_seconds": (
                        segmentation.construction_seconds
                        if segmentation is not None
                        else ""
                    ),
                    "explanation_runtime_seconds": explanation.runtime_seconds,
                }
            )

        all_rows.extend(image_rows)
        completed_indices.add(record.dataset_index)
        newly_completed += 1
        # These are intentionally written after every completed image.  The CSV
        # is tiny beside 8,000 model evaluations, and atomic replacement makes
        # interruption recovery reliable.
        atomic_write_csv(output_dir / "results.csv", all_rows)
        top_masks = {
            method.key: (
                masks_by_method[method.key][0],
                masks_by_method[method.key][1],
            )
            for method in METHODS
        }
        if not args.no_comparison_sheets:
            save_top_mask_cache(mask_cache_path(output_dir, record), top_masks)
            save_comparison_sheet(
                VisualRecord(
                    record.image_position,
                    record.dataset_index,
                    record.label_name,
                    record.predicted_label,
                    record.predicted_confidence,
                    record.image,
                    top_masks,
                    feature_counts,
                ),
                comparison_sheet_path(output_dir, record),
            )
        if args.checkpoint_every > 0 and newly_completed % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir,
                all_rows,
                newly_completed,
                len(completed_indices),
                len(records),
                record,
            )
            print(
                f"  checkpoint, summary, and plots saved after {newly_completed} new images",
                flush=True,
            )
        del explanations, segmentations, deletion_images, deletion_probabilities
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError("No complete explanation results were produced.")
    atomic_write_csv(output_dir / "results.csv", all_rows)
    save_checkpoint(
        output_dir,
        all_rows,
        newly_completed,
        len(completed_indices),
        len(records),
        records[-1],
    )
    print(f"Finished. Results are in: {output_dir}", flush=True)


class _QuantileKMeans:
    """Small deterministic KMeans-shaped test double used only by --self-test."""

    def __init__(
        self,
        n_clusters: int,
        random_state: int | None = None,
        n_init: int = 1,
        algorithm: str = "lloyd",
    ) -> None:
        del random_state, n_init, algorithm
        self.n_clusters = int(n_clusters)
        self.labels_: np.ndarray
        self.cluster_centers_: np.ndarray

    def fit(
        self, values: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "_QuantileKMeans":
        data = np.asarray(values, dtype=np.float64)
        weights = (
            np.ones(len(data), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        centered = data - data.mean(axis=0, keepdims=True)
        if data.shape[1] == 1:
            projection = centered[:, 0]
        else:
            covariance = centered.T @ centered / max(len(data) - 1, 1)
            _, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, -1]
            if axis[int(np.argmax(np.abs(axis)))] < 0:
                axis = -axis
            projection = centered @ axis
        order = np.lexsort((np.arange(len(data)), projection))
        chunks = np.array_split(order, self.n_clusters)
        labels = np.empty(len(data), dtype=np.int32)
        centers: list[np.ndarray] = []
        for cluster, chunk in enumerate(chunks):
            labels[chunk] = cluster
            centers.append(np.average(data[chunk], axis=0, weights=weights[chunk]))
        self.labels_ = labels
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        return self


def run_self_test() -> None:
    height, width = 12, 12
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                image[y, x] = [120 + 5 * y, 20 + 3 * x, 35 + (x + y) % 9]
            else:
                image[y, x] = [25 + y, 35 + 2 * y, 120 + 5 * (x - width // 2)]

    parent = build_parent_color_data(image, 2, 42, 1, _QuantileKMeans)
    expected_features = hierarchical_ideal_feature_count(parent, 3)
    descriptors = texture_descriptors(image)
    subdivision_outputs = {
        "spatial_pca": spatial_pca_subdivision(parent, 3),
        "connected_components": connected_component_subdivision(
            parent, 3, 42, 1, _QuantileKMeans
        ),
        "vit_patch": vit_patch_subdivision(
            parent, 3, (4, 4), 42, 1, _QuantileKMeans
        ),
        "texture": texture_subdivision(
            parent, descriptors, 3, 42, 1, _QuantileKMeans
        ),
        "fine_shade": fine_shade_subdivision(
            parent, 3, 42, 1, _QuantileKMeans
        ),
    }
    for key, (segments, fallback_parents, fallback_pixels) in subdivision_outputs.items():
        del fallback_parents, fallback_pixels
        assert segments.shape == (height, width), f"{key}: wrong shape"
        unique = np.unique(segments)
        assert np.array_equal(unique, np.arange(expected_features)), (
            f"{key}: labels are not contiguous or feature count differs"
        )
        for child in unique:
            parent_ids = np.unique(parent.segments[segments == child])
            assert len(parent_ids) == 1, f"{key}: a child crossed parent colors"

    spatial_four, _, _ = spatial_pca_subdivision(parent, 4)
    expected_four = hierarchical_ideal_feature_count(parent, 4)
    assert np.array_equal(np.unique(spatial_four), np.arange(expected_four))

    complete = build_color_segmentations(
        image, 2, 3, (4, 4), 42, 7, 1, _QuantileKMeans
    )
    assert set(complete) == {method.key for method in COLOR_METHODS}
    assert len({complete[key].n_features for key in HIERARCHICAL_KEYS}) == 1
    data = make_binary_perturbations(1000, expected_features, 123)
    assert np.all(data[0] == 1)
    assert data.shape == (1000, expected_features)

    mask = subdivision_outputs["spatial_pca"][0] == 0
    cutout = white_cutout(image, mask)
    assert np.array_equal(cutout[mask], image[mask])
    assert np.all(cutout[~mask] == 255)
    deleted = black_delete(image, mask)
    assert np.all(deleted[mask] == 0)
    assert np.array_equal(deleted[~mask], image[~mask])

    cache_masks = {method.key: (mask, ~mask) for method in METHODS}
    with tempfile.TemporaryDirectory() as temporary_directory:
        cache_path = Path(temporary_directory) / "masks.npz"
        save_top_mask_cache(cache_path, cache_masks)
        restored = load_top_mask_cache(cache_path)
        for method in METHODS:
            assert np.array_equal(restored[method.key][0], mask)
            assert np.array_equal(restored[method.key][1], ~mask)

    default_segments = np.zeros((height, width), dtype=np.int32)

    class OldLimeExplanation:
        def __init__(self) -> None:
            self.segments = default_segments
            self.local_exp = {3: [(0, 0.2)]}
            self.score = {3: 0.75}

    class OldLimeExplainer:
        def __init__(self, random_state: int) -> None:
            self.random_state = random_state

        # Deliberately no progress_bar argument: this is the older API that
        # caused the original server error.
        def explain_instance(
            self,
            image: np.ndarray,
            classifier_fn: Callable[..., np.ndarray],
            labels: tuple[int, ...],
            top_labels: None,
            hide_color: None,
            num_features: int,
            num_samples: int,
            batch_size: int,
            random_seed: int,
        ) -> OldLimeExplanation:
            del image, classifier_fn, labels, top_labels, hide_color
            del num_features, num_samples, batch_size, random_seed
            return OldLimeExplanation()

    class OldLimeModule:
        LimeImageExplainer = OldLimeExplainer

    class DummyPredictor:
        def predict_proba(self, values: Sequence[np.ndarray]) -> np.ndarray:
            return np.zeros((len(values), 4), dtype=np.float32)

    compatibility = explain_default_lime(
        image,
        3,
        1000,
        32,
        42,
        DummyPredictor(),  # type: ignore[arg-type]
        OldLimeModule,
    )
    assert compatibility.surrogate_score == 0.75
    assert box_mean(np.ones((5, 5)), 3).shape == (5, 5)
    print(
        "Self-test passed: all five subdivisions, equal feature counts, parent "
        "purity, black deletion, RGB white cutouts, packed masks, 1000-sample "
        "generation, recursive four-way splitting, and old/new LIME API "
        "compatibility are correct."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
