"""Kernel SHAP over binary segment-presence features, in probability units."""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .lime_engine import ExplanationResult, select_critical_mask


_SHAP_RANDOM_LOCK = threading.Lock()


def run_shap(*, image: np.ndarray, predictor: Any, target_class_id: int,
             segments: np.ndarray, config: ExperimentConfig) -> ExplanationResult:
    import shap

    image = np.asarray(image, dtype=np.uint8)
    segments = np.asarray(segments, dtype=np.int32)
    if segments.shape != image.shape[:2]:
        raise ValueError("Segmentation shape does not match image shape")
    labels, inverse = np.unique(segments, return_inverse=True)
    feature_count = len(labels)
    if feature_count > config.shap_max_segments:
        raise ValueError(
            f"Kernel SHAP received {feature_count} segments; configured limit is "
            f"{config.shap_max_segments}. Choose a coarser segmentation or raise "
            "shap_max_segments explicitly."
        )
    feature_map = inverse.reshape(segments.shape)
    baseline = np.empty_like(image)
    if config.hide_color is None:
        for label in labels:
            region = segments == label
            baseline[region] = image[region].mean(axis=0)
    else:
        baseline[:] = config.hide_color

    def predict_masks(masks: np.ndarray) -> np.ndarray:
        outputs = []
        # Never materialize all perturbed RGB images at once.
        for start in range(0, len(masks), config.inference_batch_size):
            chunk = np.asarray(masks[start:start + config.inference_batch_size])
            visible = chunk[:, feature_map] > 0.5
            images = np.where(visible[..., None], image, baseline)
            outputs.append(np.asarray(predictor.predict_proba(images))[:, target_class_id])
        return np.concatenate(outputs)

    started = time.perf_counter()
    # KernelExplainer samples with NumPy's legacy global RNG. Serialize seeded
    # calls and restore state so concurrent Streamlit sessions stay reproducible.
    with _SHAP_RANDOM_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(config.random_seed)
            explainer = shap.KernelExplainer(
                predict_masks, np.zeros((1, feature_count)), link="identity"
            )
            weights = np.asarray(explainer.shap_values(
                np.ones((1, feature_count)), nsamples=config.shap_num_samples,
                l1_reg=f"num_features({min(10, feature_count)})", silent=True,
            )).reshape(-1)
        finally:
            np.random.set_state(state)
    if len(weights) != feature_count or not np.isfinite(weights).all():
        raise ValueError("Kernel SHAP returned invalid segment attributions")
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    target = float(predict_masks(np.ones((1, feature_count)))[0])
    positive = sorted(
        [(int(label), float(weight)) for label, weight in zip(labels, weights) if weight > 0],
        key=lambda item: item[1], reverse=True,
    )
    mask, selected, area = select_critical_mask(segments, positive, config.critical_area_fraction)
    return ExplanationResult(
        explanation=None, segments=segments, positive_features=positive,
        critical_mask=mask, selected_features=selected, actual_area_fraction=area,
        surrogate_score=None, lime_seconds=time.perf_counter() - started,
        explainer="shap", diagnostics={
            "algorithm": "kernel_shap", "link": "identity",
            "baseline_target_probability": expected,
            "explained_target_probability": target,
            "additivity_residual": float(target - expected - weights.sum()),
            "requested_samples": config.shap_num_samples,
            "l1_reg": f"num_features({min(10, feature_count)})",
            "hide_color": config.hide_color,
            "feature_weights": [[int(label), float(w)] for label, w in zip(labels, weights)],
        },
    )
