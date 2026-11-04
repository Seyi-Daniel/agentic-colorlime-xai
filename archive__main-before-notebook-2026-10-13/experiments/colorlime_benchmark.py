#!/usr/bin/env python3
"""
Optimized six-method ViT / ImageNet LIME experiment.

The six explanation methods are:

1. Default LIME (ordinary Quickshift superpixels, LIME's standard mean off-state)
2. Color-LIME, black off-state
3. Color-LIME, structure-preserving fixed random transformation
4. Color-LIME, structure-preserving fresh random transformation per perturbation
5. Color-LIME, independent-original-color fixed random mapping
6. Color-LIME, independent-original-color fresh random mapping per perturbation

Important controls:

* num_samples defaults to 1,000 for every method.
* Color-LIME uses weighted RGB K-means with k=256 by default.
* The five Color-LIME methods share one segmentation and one binary perturbation
  matrix for each image. Only their off-state image synthesis differs.
* All randomness is deterministic. Fresh mappings depend on the global seed,
  image, method, perturbation row, feature, and (where relevant) original RGB.
* Random replacement is used only while fitting the LIME explanation.
* CIR/DIR always black-delete the method's positive critical region.
* CIR/DIR critical regions accumulate the strongest positive features until at
  least 20% of the image area is covered.
* Top-10% and top-20% displays mean percentages of the count of positive LIME
  features, not percentages of pixel area.
* White-background cutouts copy the selected RGB pixels from the actual resized
  input image. They never replace selected pixels with flat green.
* results.csv and compact packed visual masks are saved after every completed
  correct image. summary.csv/checkpoint_state.json refresh every five newly
  completed images by default. Use --resume after an interrupted run.

Typical installation:

    python -m pip install torch transformers datasets accelerate lime \
        scikit-learn scikit-image pillow matplotlib numpy

ImageNet is gated on Hugging Face. Set HF_TOKEN when the local cache is not
already authorized:

    export HF_TOKEN="your_huggingface_read_token"
    python imagenet_vit_six_method_colorlime_optimized.py

The full experiment is intentionally expensive: at most 100 correctly
predicted images x 6 methods x 1,000 perturbations = 600,000 perturbed ViT
evaluations. Incorrectly predicted images are screened before K-means or LIME.
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
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TOTAL_RGB_COLORS = 1 << 24
RANDOM_OTHER_COLOR_COUNT = TOTAL_RGB_COLORS - 1


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    short_name: str
    is_color_lime: bool


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("default_lime", "Default LIME", "Default LIME", False),
    MethodSpec("color_black", "Color-LIME — Black", "Color Black", True),
    MethodSpec(
        "structure_fixed",
        "Color-LIME — Structure-Preserving Fixed",
        "Structure Fixed",
        True,
    ),
    MethodSpec(
        "structure_fresh",
        "Color-LIME — Structure-Preserving Fresh",
        "Structure Fresh",
        True,
    ),
    MethodSpec(
        "independent_fixed",
        "Color-LIME — Independent-Color Fixed",
        "Independent Fixed",
        True,
    ),
    MethodSpec(
        "independent_fresh",
        "Color-LIME — Independent-Color Fresh",
        "Independent Fresh",
        True,
    ),
)

METHOD_BY_KEY = {method.key: method for method in METHODS}
COLOR_METHOD_KEYS = tuple(method.key for method in METHODS if method.is_color_lime)


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
    correct: bool = False


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


@dataclass
class ColorFeatureData:
    segments: np.ndarray
    unique_colors: np.ndarray
    pixel_to_unique: np.ndarray
    unique_to_feature: np.ndarray
    feature_means: np.ndarray
    feature_pixel_counts: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.feature_means.shape[0])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimized six-method ViT / ImageNet Color-LIME experiment."
    )
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--k-colors", type=int, default=256)
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
            "Refresh summary.csv and checkpoint_state.json after this many newly "
            "completed correct images. results.csv and compact masks are saved "
            "after every completed image. Use 0 to refresh summaries only at the end."
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
        default=Path("imagenet_vit_six_method_colorlime_outputs"),
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
        help="Stream the requested ImageNet examples instead of using an indexed dataset.",
    )
    parser.add_argument(
        "--fast-gpu-preprocess",
        action="store_true",
        help=(
            "Normalize already-resized uint8 images directly with torch. Faster, but "
            "the exact official processor path remains the default."
        ),
    )
    parser.add_argument(
        "--no-comparison-sheets",
        action="store_true",
        help="Skip the per-image six-row comparison sheets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a compatible interrupted run in --output-dir. Completed "
            "six-method images with saved masks are not recomputed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing nonempty output directory after safety checks.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test off-state generation and cutout semantics without loading ViT/ImageNet.",
    )
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    if args.num_samples < 2:
        parser.error("--num-samples must be at least 2")
    if args.k_colors < 2:
        parser.error("--k-colors must be at least 2")
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
    if bits <= 32:
        value &= (1 << bits) - 1
    return value


def rgb_to_codes(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.uint32)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def codes_to_rgb(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint32)
    return np.stack(
        ((values >> 16) & 255, (values >> 8) & 255, values & 255), axis=-1
    ).astype(np.uint8)


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized deterministic 64-bit mixing used as a stateless PRNG."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stateless_other_color_codes(
    base_seed: int,
    perturbation_ids: np.ndarray,
    feature_ids: np.ndarray,
    item_ids: np.ndarray,
    excluded_codes: np.ndarray,
) -> np.ndarray:
    """
    Produce one uniform 24-bit RGB code other than each excluded code.

    Output shape is (len(perturbation_ids), len(feature_ids)). The key contains
    the perturbation, feature, and item identity, so results do not depend on
    synthesis batch size or iteration order.
    """
    p = np.asarray(perturbation_ids, dtype=np.uint64).reshape(-1, 1)
    f = np.asarray(feature_ids, dtype=np.uint64).reshape(1, -1)
    item = np.asarray(item_ids, dtype=np.uint64).reshape(1, -1)
    excluded = np.asarray(excluded_codes, dtype=np.uint32).reshape(1, -1)
    with np.errstate(over="ignore"):
        key = (
            np.uint64(base_seed)
            ^ ((p + np.uint64(1)) * np.uint64(0xD2B74407B1CE6E93))
            ^ ((f + np.uint64(1)) * np.uint64(0xCA5A826395121157))
            ^ ((item + np.uint64(1)) * np.uint64(0x9E3779B185EBCA87))
        )
    draws = (splitmix64(key) % np.uint64(RANDOM_OTHER_COLOR_COUNT)).astype(np.uint32)
    return draws + (draws >= excluded).astype(np.uint32)


def prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> Path:
    resolved = path.expanduser().resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to use protected path as output directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if resume:
            return resolved
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {resolved}\n"
                "Pass --resume to continue it, choose another --output-dir, or "
                "explicitly pass --overwrite."
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
        outputs: list[np.ndarray] = []
        position = 0
        current_microbatch = min(self.microbatch, max(total, 1))
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
                    f"CUDA memory pressure: reducing model microbatch to {current_microbatch}",
                    flush=True,
                )
        return np.concatenate(outputs, axis=0)


def get_label_name(model: Any, class_id: int) -> str:
    id2label = getattr(model.config, "id2label", {}) or {}
    return str(id2label.get(class_id, id2label.get(str(class_id), class_id)))


def load_requested_records(
    args: argparse.Namespace,
    load_dataset: Callable[..., Any],
    processor: Any,
) -> list[ImageRecord]:
    token = os.environ.get("HF_TOKEN") or None
    kwargs: dict[str, Any] = {
        "split": args.dataset_split,
        "streaming": args.streaming,
    }
    if token:
        kwargs["token"] = token
    try:
        dataset = load_dataset(args.dataset_name, **kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        dataset = load_dataset(args.dataset_name, **kwargs)

    records: list[ImageRecord] = []
    stop = args.start_index + args.num_images
    if args.streaming:
        iterator: Iterable[tuple[int, Any]] = enumerate(dataset)
        selected = (
            (index, sample)
            for index, sample in iterator
            if args.start_index <= index < stop
        )
    else:
        dataset_length = len(dataset)
        if stop > dataset_length:
            raise IndexError(
                f"Requested through dataset index {stop - 1}, but split has "
                f"only {dataset_length} examples."
            )
        selected = ((index, dataset[index]) for index in range(args.start_index, stop))

    for position, (dataset_index, sample) in enumerate(selected, start=1):
        if position > args.num_images:
            break
        image_value = sample[args.image_column]
        if not isinstance(image_value, Image.Image):
            image_value = Image.fromarray(np.asarray(image_value))
        image = resize_to_model_space(image_value, processor)
        records.append(
            ImageRecord(
                image_position=position,
                dataset_index=int(dataset_index),
                label=int(sample[args.label_column]),
                image=image,
            )
        )
    if len(records) != args.num_images:
        raise RuntimeError(
            f"Only obtained {len(records)} examples; requested {args.num_images}."
        )
    return records


def build_color_features(
    image: np.ndarray,
    k_colors: int,
    image_seed: int,
    kmeans_n_init: int,
    KMeans: Any,
) -> ColorFeatureData:
    flat = image.reshape(-1, 3)
    unique_colors, pixel_to_unique, unique_counts = np.unique(
        flat, axis=0, return_inverse=True, return_counts=True
    )
    n_clusters = min(k_colors, len(unique_colors))
    if n_clusters < 2:
        raise ValueError("Color-LIME requires at least two distinct RGB colors.")
    estimator = KMeans(
        n_clusters=n_clusters,
        random_state=int(image_seed % (2**31 - 1)),
        n_init=kmeans_n_init,
        algorithm="lloyd",
    )
    estimator.fit(unique_colors.astype(np.float32), sample_weight=unique_counts)
    raw_unique_features = estimator.labels_.astype(np.int32, copy=False)
    _, unique_to_feature = np.unique(raw_unique_features, return_inverse=True)
    unique_to_feature = unique_to_feature.astype(np.int32, copy=False)
    segment_flat = unique_to_feature[pixel_to_unique]
    n_features = int(segment_flat.max()) + 1
    feature_pixel_counts = np.bincount(segment_flat, minlength=n_features).astype(np.int64)
    feature_sums = np.zeros((n_features, 3), dtype=np.float64)
    np.add.at(feature_sums, segment_flat, flat.astype(np.float64))
    feature_means = feature_sums / feature_pixel_counts[:, None]
    return ColorFeatureData(
        segments=segment_flat.reshape(image.shape[:2]).astype(np.int32, copy=False),
        unique_colors=unique_colors.astype(np.uint8, copy=False),
        pixel_to_unique=pixel_to_unique.astype(np.int32, copy=False),
        unique_to_feature=unique_to_feature,
        feature_means=feature_means.astype(np.float32),
        feature_pixel_counts=feature_pixel_counts,
    )


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


def fixed_structure_replacement(
    image: np.ndarray,
    color: ColorFeatureData,
    global_seed: int,
    image_index: int,
) -> np.ndarray:
    feature_ids = np.arange(color.n_features, dtype=np.uint64)
    mean_codes = rgb_to_codes(np.clip(np.rint(color.feature_means), 0, 255).astype(np.uint8))
    target_codes = stateless_other_color_codes(
        stable_seed(global_seed, image_index, "structure_fixed"),
        np.array([0], dtype=np.uint64),
        feature_ids,
        feature_ids,
        mean_codes,
    )[0]
    targets = codes_to_rgb(target_codes).astype(np.float32)
    deltas = targets - color.feature_means
    translated = image.astype(np.float32) + deltas[color.segments]
    return np.clip(np.rint(translated), 0, 255).astype(np.uint8)


def fixed_independent_replacement(
    color: ColorFeatureData,
    global_seed: int,
    image_index: int,
) -> np.ndarray:
    unique_codes = rgb_to_codes(color.unique_colors)
    target_codes = stateless_other_color_codes(
        stable_seed(global_seed, image_index, "independent_fixed"),
        np.array([0], dtype=np.uint64),
        color.unique_to_feature.astype(np.uint64),
        unique_codes.astype(np.uint64),
        unique_codes,
    )[0]
    target_unique = codes_to_rgb(target_codes)
    return target_unique[color.pixel_to_unique].reshape(*color.segments.shape, 3)


def synthesize_color_method(
    image: np.ndarray,
    color: ColorFeatureData,
    binary_chunk: np.ndarray,
    perturbation_ids: np.ndarray,
    method_key: str,
    global_seed: int,
    image_index: int,
    fixed_replacements: dict[str, np.ndarray],
) -> np.ndarray:
    on_pixels = binary_chunk[:, color.segments].astype(bool, copy=False)
    original = image[None, ...]
    if method_key in fixed_replacements:
        replacement = fixed_replacements[method_key][None, ...]
        return np.where(on_pixels[..., None], original, replacement).astype(
            np.uint8, copy=False
        )

    if method_key == "structure_fresh":
        feature_ids = np.arange(color.n_features, dtype=np.uint64)
        mean_codes = rgb_to_codes(
            np.clip(np.rint(color.feature_means), 0, 255).astype(np.uint8)
        )
        target_codes = stateless_other_color_codes(
            stable_seed(global_seed, image_index, method_key),
            perturbation_ids,
            feature_ids,
            feature_ids,
            mean_codes,
        )
        targets = codes_to_rgb(target_codes).astype(np.float32)
        deltas = targets - color.feature_means[None, ...]
        translated = image.astype(np.float32)[None, ...] + deltas[:, color.segments]
        replacement = np.clip(np.rint(translated), 0, 255).astype(np.uint8)
        return np.where(on_pixels[..., None], original, replacement).astype(
            np.uint8, copy=False
        )

    if method_key == "independent_fresh":
        unique_codes = rgb_to_codes(color.unique_colors)
        target_codes = stateless_other_color_codes(
            stable_seed(global_seed, image_index, method_key),
            perturbation_ids,
            color.unique_to_feature.astype(np.uint64),
            unique_codes.astype(np.uint64),
            unique_codes,
        )
        target_unique = codes_to_rgb(target_codes)
        replacement = target_unique[:, color.pixel_to_unique].reshape(
            len(perturbation_ids), *color.segments.shape, 3
        )
        return np.where(on_pixels[..., None], original, replacement).astype(
            np.uint8, copy=False
        )

    raise KeyError(f"Unknown Color-LIME method: {method_key}")


def run_color_perturbation_predictions(
    image: np.ndarray,
    color: ColorFeatureData,
    binary_data: np.ndarray,
    predictor: ViTPredictor,
    batch_size: int,
    global_seed: int,
    image_index: int,
) -> dict[str, np.ndarray]:
    fixed_replacements = {
        "color_black": np.zeros_like(image),
        "structure_fixed": fixed_structure_replacement(
            image, color, global_seed, image_index
        ),
        "independent_fixed": fixed_independent_replacement(
            color, global_seed, image_index
        ),
    }
    probability_chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in COLOR_METHOD_KEYS
    }
    for start in range(0, len(binary_data), batch_size):
        end = min(start + batch_size, len(binary_data))
        perturbation_ids = np.arange(start, end, dtype=np.uint64)
        chunk = binary_data[start:end]
        synthesized = [
            synthesize_color_method(
                image,
                color,
                chunk,
                perturbation_ids,
                method_key,
                global_seed,
                image_index,
                fixed_replacements,
            )
            for method_key in COLOR_METHOD_KEYS
        ]
        combined = np.concatenate(synthesized, axis=0)
        combined_probabilities = predictor.predict_proba(combined)
        chunk_length = end - start
        for method_number, method_key in enumerate(COLOR_METHOD_KEYS):
            left = method_number * chunk_length
            right = left + chunk_length
            probability_chunks[method_key].append(combined_probabilities[left:right])
        del synthesized, combined, combined_probabilities
    return {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in probability_chunks.items()
    }


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
    # progress_bar was added after older LIME releases. Inspecting the installed
    # signature keeps this runner compatible with both APIs.
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
        score_value = float(score_container.get(int(target_class), float("nan")))
    else:
        score_value = float(score_container)
    return ExplanationResult(
        method_key="default_lime",
        segments=np.asarray(explanation.segments, dtype=np.int32),
        weights=weights,
        surrogate_score=score_value,
        runtime_seconds=time.perf_counter() - started,
    )


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
    selected = positive[:selected_count]
    return np.isin(segments, selected), selected_count, len(positive)


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
        green = np.array([55.0, 220.0, 80.0], dtype=np.float32)
        output[mask] = np.clip(np.rint(0.68 * source + 0.32 * green), 0, 255).astype(
            np.uint8
        )
    return output


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Replace a small CSV atomically so an interruption cannot leave half a file."""
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
    """Load only complete six-method image groups from an interrupted run."""
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
    method_keys = set(METHOD_BY_KEY)
    complete_indices = {
        dataset_index
        for dataset_index in valid_dataset_indices
        if {
            method_key
            for (row_index, method_key) in latest
            if row_index == dataset_index
        }
        == method_keys
    }
    complete_rows = [
        latest[(dataset_index, method.key)]
        for dataset_index in sorted(complete_indices)
        for method in METHODS
    ]
    return complete_rows


def completed_dataset_indices(rows: Sequence[dict[str, Any]]) -> set[int]:
    grouped: dict[int, set[str]] = {}
    for row in rows:
        grouped.setdefault(int(row["dataset_index"]), set()).add(str(row["method_key"]))
    expected = set(METHOD_BY_KEY)
    return {dataset_index for dataset_index, keys in grouped.items() if keys == expected}


def top_mask_cache_path(output_dir: Path, record: ImageRecord) -> Path:
    return (
        output_dir
        / "mask_cache"
        / (
            f"image_{record.image_position:03d}_"
            f"dataset_{record.dataset_index:06d}_top_masks.npz"
        )
    )


def save_top_mask_cache(
    path: Path,
    masks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save 12 boolean masks packed to one bit each (about 75 KB at 224x224)."""
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
            unpacked[method_number * 2].astype(bool, copy=False),
            unpacked[method_number * 2 + 1].astype(bool, copy=False),
        )
        for method_number, method in enumerate(METHODS)
    }


def validate_resume_config(args: argparse.Namespace, output_dir: Path) -> None:
    config_path = output_dir / "run_config.json"
    if not args.resume or not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    compatibility_keys = (
        "num_images",
        "start_index",
        "num_samples",
        "k_colors",
        "critical_area_fraction",
        "seed",
        "model_id",
        "dataset_name",
        "dataset_split",
        "precision",
        "fast_gpu_preprocess",
    )
    current = vars(args)
    mismatches = [
        key
        for key in compatibility_keys
        if key in saved and saved[key] != current[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved[key]!r}, requested={current[key]!r}"
            for key in mismatches
        )
        raise ValueError(
            "Cannot resume because experiment-defining settings changed: " + details
        )


def save_lightweight_checkpoint(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    newly_completed: int,
    total_complete: int,
    total_correct: int,
    last_record: ImageRecord,
) -> None:
    summary_rows = summarize_results(rows)
    atomic_write_csv(output_dir / "summary.csv", summary_rows)
    atomic_write_json(
        output_dir / "checkpoint_state.json",
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "newly_completed_this_run": newly_completed,
            "complete_correct_images": total_complete,
            "total_correct_images": total_correct,
            "last_image_position": last_record.image_position,
            "last_dataset_index": last_record.dataset_index,
            "results_csv": str(output_dir / "results.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
        },
    )


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
        scores = np.asarray([float(row["surrogate_score"]) for row in selected])
        runtimes = [float(row["explanation_runtime_seconds"]) for row in selected]
        finite_scores = scores[np.isfinite(scores)]
        summary.append(
            {
                "method_key": method.key,
                "method": method.display_name,
                "n_correct_images": len(selected),
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
                "critical_area_fraction_median": float(np.median(area)),
                "surrogate_score_mean": float(np.mean(finite_scores))
                if finite_scores.size
                else float("nan"),
                "runtime_seconds_mean": float(np.mean(runtimes)),
                "runtime_seconds_total": float(np.sum(runtimes)),
            }
        )
    return summary


def save_boxplot(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [
        [float(row[value_key]) for row in rows if row["method_key"] == method.key]
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(14, 7))
    box = axis.boxplot(
        data,
        labels=[method.short_name for method in METHODS],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 5},
    )
    colors = ("#8DA0CB", "#66C2A5", "#FC8D62", "#E78AC3", "#A6D854", "#FFD92F")
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_title(title, fontsize=15)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    if ylim is not None:
        axis.set_ylim(*ylim)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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
    row_label_width = 230
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
    label_font = load_font(16, bold=True)
    column_font = load_font(14, bold=True)
    title = (
        f"Image {record.image_position:03d}  |  dataset index {record.dataset_index}  |  "
        f"confidence {record.predicted_confidence:.4f}"
    )
    draw.text((16, 12), title, fill="black", font=title_font)
    draw.text((16, 49), f"Ground truth: {record.label_name}", fill="black", font=regular_font)
    draw.text((16, 75), f"Prediction: {record.predicted_label}", fill="black", font=regular_font)
    for column_index, column_name in enumerate(columns):
        x = row_label_width + column_index * panel_width
        draw.rectangle((x, header_height, x + panel_width, header_height + column_header_height), fill=(238, 242, 247))
        draw.text((x + 8, header_height + 11), column_name, fill="black", font=column_font)
    for row_index, method in enumerate(METHODS):
        y = header_height + column_header_height + row_index * row_height
        draw.rectangle((0, y, row_label_width, y + panel_height), fill=(247, 247, 247))
        draw.multiline_text(
            (12, y + 76),
            method.display_name.replace(" — ", "\n"),
            fill="black",
            font=label_font,
            spacing=6,
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
    canvas.save(output_path, format="PNG", compress_level=1)


def save_config(args: argparse.Namespace, output_dir: Path, device: Any) -> None:
    payload = vars(args).copy()
    payload["output_dir"] = str(output_dir)
    payload["device_resolved"] = str(device)
    payload["methods"] = [asdict(method) for method in METHODS]
    atomic_write_json(output_dir / "run_config.json", payload)


def run_experiment(args: argparse.Namespace) -> None:
    deps = import_experiment_dependencies()
    torch = deps["torch"]
    if args.num_samples != 1000:
        print(
            f"Note: --num-samples is {args.num_samples}; the requested main experiment uses 1000.",
            flush=True,
        )
    if args.k_colors != 256:
        print(
            f"Note: --k-colors is {args.k_colors}; the requested main experiment uses 256.",
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
    predictor = ViTPredictor(
        torch=torch,
        processor=processor,
        model=model,
        device=device,
        microbatch=args.model_microbatch,
        precision=args.precision,
        fast_gpu_preprocess=args.fast_gpu_preprocess,
    )
    save_config(args, output_dir, device)

    print(f"Loading {args.num_images} ImageNet validation images ...", flush=True)
    records = load_requested_records(args, deps["load_dataset"], processor)
    original_probabilities = predictor.predict_proba([record.image for record in records])
    prediction_rows: list[dict[str, Any]] = []
    correct_records: list[ImageRecord] = []
    for record, probabilities in zip(records, original_probabilities):
        prediction = int(np.argmax(probabilities))
        record.predicted_class = prediction
        record.predicted_confidence = float(probabilities[prediction])
        record.correct = prediction == record.label
        record.label_name = get_label_name(model, record.label)
        record.predicted_label = get_label_name(model, prediction)
        if record.correct:
            correct_records.append(record)
        prediction_rows.append(
            {
                "image_position": record.image_position,
                "dataset_index": record.dataset_index,
                "ground_truth_class": record.label,
                "ground_truth_label": record.label_name,
                "predicted_class": prediction,
                "predicted_label": record.predicted_label,
                "predicted_confidence": record.predicted_confidence,
                "correct": int(record.correct),
            }
        )
    atomic_write_csv(output_dir / "prediction_summary.csv", prediction_rows)
    print(
        f"Correct predictions: {len(correct_records)}/{len(records)}. "
        "K-means and all explanations run only for these images.",
        flush=True,
    )

    valid_dataset_indices = {record.dataset_index for record in correct_records}
    all_result_rows = (
        load_result_rows(output_dir / "results.csv", valid_dataset_indices)
        if args.resume
        else []
    )
    completed_indices = completed_dataset_indices(all_result_rows)
    if all_result_rows:
        # This also removes duplicate or incomplete trailing groups from an older
        # interrupted write before new results are added.
        atomic_write_csv(output_dir / "results.csv", all_result_rows)
        print(
            f"Resume found {len(completed_indices)}/{len(correct_records)} "
            "complete six-method images.",
            flush=True,
        )
    newly_completed = 0
    for correct_number, record in enumerate(correct_records, start=1):
        cache_path = top_mask_cache_path(output_dir, record)
        resume_ready = record.dataset_index in completed_indices and (
            args.no_comparison_sheets or cache_path.exists()
        )
        if resume_ready:
            print(
                f"[{correct_number}/{len(correct_records)} correct] "
                f"image {record.image_position:03d}, dataset index "
                f"{record.dataset_index} — resumed, already complete",
                flush=True,
            )
            continue
        if record.dataset_index in completed_indices:
            # Numerical rows without their requested visual masks are not enough
            # to finish the comparison sheets, so redo this one image.
            all_result_rows = [
                row
                for row in all_result_rows
                if int(row["dataset_index"]) != record.dataset_index
            ]
            completed_indices.discard(record.dataset_index)
        print(
            f"[{correct_number}/{len(correct_records)} correct] "
            f"image {record.image_position:03d}, dataset index {record.dataset_index}",
            flush=True,
        )
        explanations: dict[str, ExplanationResult] = {}
        default_seed = stable_seed(args.seed, record.dataset_index, "default_lime", bits=32)
        default_result = explain_default_lime(
            record.image,
            record.predicted_class,
            args.num_samples,
            args.lime_batch_size,
            default_seed,
            predictor,
            deps["lime_image"],
        )
        explanations[default_result.method_key] = default_result

        color_started = time.perf_counter()
        color = build_color_features(
            record.image,
            args.k_colors,
            stable_seed(args.seed, record.dataset_index, "kmeans", bits=32),
            args.kmeans_n_init,
            deps["KMeans"],
        )
        binary_data = make_binary_perturbations(
            args.num_samples,
            color.n_features,
            stable_seed(args.seed, record.dataset_index, "shared_color_binary", bits=32),
        )
        color_probabilities = run_color_perturbation_predictions(
            record.image,
            color,
            binary_data,
            predictor,
            args.lime_batch_size,
            args.seed,
            record.dataset_index,
        )
        color_prediction_runtime = time.perf_counter() - color_started
        for method_key in COLOR_METHOD_KEYS:
            fit_started = time.perf_counter()
            weights, score = fit_lime_surrogate(
                binary_data,
                color_probabilities[method_key],
                record.predicted_class,
                stable_seed(args.seed, record.dataset_index, method_key, "surrogate", bits=32),
                deps["LimeBase"],
            )
            explanations[method_key] = ExplanationResult(
                method_key=method_key,
                segments=color.segments,
                weights=weights,
                surrogate_score=score,
                runtime_seconds=(color_prediction_runtime / len(COLOR_METHOD_KEYS))
                + (time.perf_counter() - fit_started),
            )

        deletion_images: list[np.ndarray] = []
        per_method_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]] = {}
        for method in METHODS:
            explanation = explanations[method.key]
            mask10, selected10, positive_total = mask_for_positive_fraction(
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
            per_method_masks[method.key] = (
                mask10,
                mask20,
                critical_mask,
                selected10,
                selected20,
                critical_selected,
            )
            deletion_images.append(black_delete(record.image, critical_mask))

        top_masks = {
            method.key: (
                per_method_masks[method.key][0],
                per_method_masks[method.key][1],
            )
            for method in METHODS
        }
        if not args.no_comparison_sheets:
            save_top_mask_cache(cache_path, top_masks)

        deletion_probabilities = predictor.predict_proba(deletion_images)
        image_rows: list[dict[str, Any]] = []
        original_target_confidence = record.predicted_confidence
        for method_number, method in enumerate(METHODS):
            explanation = explanations[method.key]
            mask10, mask20, critical_mask, selected10, selected20, critical_selected = (
                per_method_masks[method.key]
            )
            deleted_probabilities = deletion_probabilities[method_number]
            deleted_target_confidence = float(
                deleted_probabilities[record.predicted_class]
            )
            deleted_prediction = int(np.argmax(deleted_probabilities))
            positive_total = len(positive_feature_ids(explanation.weights))
            row = {
                "image_position": record.image_position,
                "dataset_index": record.dataset_index,
                "ground_truth_class": record.label,
                "ground_truth_label": record.label_name,
                "predicted_class": record.predicted_class,
                "predicted_label": record.predicted_label,
                "original_target_confidence": original_target_confidence,
                "method_key": method.key,
                "method": method.display_name,
                "num_samples": args.num_samples,
                "k_requested": args.k_colors if method.is_color_lime else "",
                "feature_count": int(len(np.unique(explanation.segments))),
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
                "cir": max(0.0, original_target_confidence - deleted_target_confidence),
                "dir": int(deleted_prediction != record.predicted_class),
                "surrogate_score": explanation.surrogate_score,
                "explanation_runtime_seconds": explanation.runtime_seconds,
            }
            image_rows.append(row)
        all_result_rows.extend(image_rows)
        completed_indices.add(record.dataset_index)
        newly_completed += 1
        # Six rows are tiny; replacing this CSV atomically after every completed
        # image gives useful recovery with negligible cost beside 6,000 ViT calls.
        atomic_write_csv(output_dir / "results.csv", all_result_rows)
        if (
            args.checkpoint_every > 0
            and newly_completed % args.checkpoint_every == 0
        ):
            save_lightweight_checkpoint(
                output_dir,
                all_result_rows,
                newly_completed,
                len(completed_indices),
                len(correct_records),
                record,
            )
            print(
                f"  checkpoint saved after {newly_completed} new correct images",
                flush=True,
            )
        del color_probabilities, binary_data, explanations, deletion_images
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_result_rows:
        print("No correctly predicted images; no CIR/DIR results were produced.")
        return

    atomic_write_csv(output_dir / "results.csv", all_result_rows)
    save_lightweight_checkpoint(
        output_dir,
        all_result_rows,
        newly_completed,
        len(completed_indices),
        len(correct_records),
        correct_records[-1],
    )
    summary_rows = summarize_results(all_result_rows)
    save_boxplot(
        all_result_rows,
        "cir",
        output_dir / "cir_boxplot.png",
        "Confidence Impact Ratio (black deletion for every method)",
        "Nonnegative confidence drop",
        (0.0, 1.0),
    )
    save_boxplot(
        all_result_rows,
        "dir",
        output_dir / "dir_boxplot.png",
        "Decision Impact Ratio (black deletion for every method)",
        "Decision changed (0 or 1)",
        (-0.05, 1.05),
    )

    if not args.no_comparison_sheets:
        print("Rendering deferred comparison sheets ...", flush=True)
        sheets_dir = output_dir / "comparison_sheets"
        renderable_records = [
            record
            for record in correct_records
            if record.dataset_index in completed_indices
            and top_mask_cache_path(output_dir, record).exists()
        ]
        missing_count = len(completed_indices) - len(renderable_records)
        if missing_count:
            print(
                f"Warning: {missing_count} complete images have no mask cache and "
                "cannot be rendered in this pass.",
                flush=True,
            )
        for sheet_number, record in enumerate(renderable_records, start=1):
            visual_record = VisualRecord(
                image_position=record.image_position,
                dataset_index=record.dataset_index,
                label_name=record.label_name,
                predicted_label=record.predicted_label,
                predicted_confidence=record.predicted_confidence,
                image=record.image,
                masks=load_top_mask_cache(top_mask_cache_path(output_dir, record)),
            )
            save_comparison_sheet(
                visual_record,
                sheets_dir
                / (
                    f"image_{visual_record.image_position:03d}_"
                    f"dataset_{visual_record.dataset_index:06d}_six_method_comparison.png"
                ),
            )
            if sheet_number % 10 == 0 or sheet_number == len(renderable_records):
                print(
                    f"  comparison sheets: {sheet_number}/{len(renderable_records)}",
                    flush=True,
                )
    print(f"Finished. Results are in: {output_dir}", flush=True)


def make_test_color_data(image: np.ndarray, segments: np.ndarray) -> ColorFeatureData:
    flat = image.reshape(-1, 3)
    unique_colors, pixel_to_unique = np.unique(flat, axis=0, return_inverse=True)
    segment_flat = segments.reshape(-1)
    unique_to_feature = np.empty(len(unique_colors), dtype=np.int32)
    for unique_id in range(len(unique_colors)):
        unique_to_feature[unique_id] = int(segment_flat[pixel_to_unique == unique_id][0])
    n_features = int(segments.max()) + 1
    counts = np.bincount(segment_flat, minlength=n_features)
    sums = np.zeros((n_features, 3), dtype=np.float64)
    np.add.at(sums, segment_flat, flat.astype(np.float64))
    return ColorFeatureData(
        segments=segments.astype(np.int32),
        unique_colors=unique_colors.astype(np.uint8),
        pixel_to_unique=pixel_to_unique.astype(np.int32),
        unique_to_feature=unique_to_feature,
        feature_means=(sums / counts[:, None]).astype(np.float32),
        feature_pixel_counts=counts,
    )


def run_self_test() -> None:
    image = np.array(
        [
            [[30, 50, 70], [40, 60, 80], [30, 50, 70]],
            [[120, 130, 140], [130, 140, 150], [120, 130, 140]],
        ],
        dtype=np.uint8,
    )
    segments = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    color = make_test_color_data(image, segments)
    data = np.array([[1, 1], [0, 1], [0, 1]], dtype=np.int8)
    pids = np.array([0, 1, 2], dtype=np.uint64)
    fixed = {
        "color_black": np.zeros_like(image),
        "structure_fixed": fixed_structure_replacement(image, color, 42, 7),
        "independent_fixed": fixed_independent_replacement(color, 42, 7),
    }
    outputs = {
        key: synthesize_color_method(
            image, color, data, pids, key, 42, 7, fixed
        )
        for key in COLOR_METHOD_KEYS
    }
    for key, values in outputs.items():
        assert np.array_equal(values[0], image), f"{key}: all-on row changed"
        assert np.array_equal(values[1][segments == 1], image[segments == 1]), (
            f"{key}: on feature changed"
        )
    assert np.all(outputs["color_black"][1][segments == 0] == 0)
    assert np.array_equal(
        outputs["structure_fixed"][1][segments == 0],
        outputs["structure_fixed"][2][segments == 0],
    )
    assert not np.array_equal(
        outputs["structure_fresh"][1][segments == 0],
        outputs["structure_fresh"][2][segments == 0],
    )
    assert np.array_equal(
        outputs["independent_fixed"][1][segments == 0],
        outputs["independent_fixed"][2][segments == 0],
    )
    assert not np.array_equal(
        outputs["independent_fresh"][1][segments == 0],
        outputs["independent_fresh"][2][segments == 0],
    )
    duplicate_positions = [(0, 0), (0, 2)]
    for key in ("independent_fixed", "independent_fresh"):
        values = outputs[key][1]
        assert np.array_equal(
            values[duplicate_positions[0]], values[duplicate_positions[1]]
        ), f"{key}: identical original RGB did not map consistently"
    mask = segments == 0
    cutout = white_cutout(image, mask)
    assert np.array_equal(cutout[mask], image[mask])
    assert np.all(cutout[~mask] == 255)
    cache_masks = {method.key: (mask, ~mask) for method in METHODS}
    with tempfile.TemporaryDirectory() as temporary_directory:
        cache_path = Path(temporary_directory) / "masks.npz"
        save_top_mask_cache(cache_path, cache_masks)
        restored_masks = load_top_mask_cache(cache_path)
        for method in METHODS:
            assert np.array_equal(restored_masks[method.key][0], mask)
            assert np.array_equal(restored_masks[method.key][1], ~mask)

    class OldLimeExplanation:
        def __init__(self) -> None:
            self.segments = segments
            self.local_exp = {3: [(0, 0.2), (1, -0.1)]}
            self.score = {3: 0.75}

    class OldLimeExplainer:
        def __init__(self, random_state: int) -> None:
            self.random_state = random_state

        # Deliberately has no progress_bar argument. This reproduces the API
        # that caused the reported server failure.
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

    compatibility_result = explain_default_lime(
        image=image,
        target_class=3,
        num_samples=1000,
        batch_size=32,
        image_seed=42,
        predictor=DummyPredictor(),  # type: ignore[arg-type]
        lime_image=OldLimeModule,
    )
    assert compatibility_result.surrogate_score == 0.75
    repeated = synthesize_color_method(
        image, color, data, pids, "structure_fresh", 42, 7, fixed
    )
    assert np.array_equal(repeated, outputs["structure_fresh"])
    print(
        "Self-test passed: old/new LIME compatibility, fixed/fresh mappings, "
        "packed checkpoints, black deletion, and RGB cutouts are correct."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
