from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ExperimentConfig


@dataclass
class ExplanationResult:
    explanation: Any
    segments: np.ndarray
    positive_features: list[tuple[int, float]]
    critical_mask: np.ndarray
    selected_features: list[tuple[int, float]]
    actual_area_fraction: float
    surrogate_score: float | None
    lime_seconds: float
    explainer: str = "lime"
    diagnostics: dict[str, Any] = field(default_factory=dict)


# Preserve the existing Python API; runtime summaries use explanation_seconds.
LimeResult = ExplanationResult


def _score_for_label(explanation: Any, target_class_id: int) -> float:
    score = getattr(explanation, "score", float("nan"))
    if isinstance(score, dict):
        score = score.get(int(target_class_id), float("nan"))
    try:
        return float(np.asarray(score).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def ranked_positive_features(explanation: Any, target_class_id: int) -> list[tuple[int, float]]:
    local_exp = getattr(explanation, "local_exp", {}) or {}
    weights = local_exp.get(int(target_class_id), [])
    return sorted(
        [(int(feature_id), float(weight)) for feature_id, weight in weights if float(weight) > 0],
        key=lambda item: item[1],
        reverse=True,
    )


def select_critical_mask(
    segments: np.ndarray,
    positive_features: list[tuple[int, float]],
    target_area_fraction: float,
) -> tuple[np.ndarray, list[tuple[int, float]], float]:
    mask = np.zeros(segments.shape, dtype=bool)
    target_pixels = max(1, int(math.ceil(float(target_area_fraction) * mask.size)))
    selected: list[tuple[int, float]] = []
    for feature_id, weight in positive_features:
        mask |= segments == int(feature_id)
        selected.append((int(feature_id), float(weight)))
        if int(mask.sum()) >= target_pixels:
            break
    return mask, selected, float(mask.mean())


def run_lime(
    *,
    image: np.ndarray,
    predictor: Any,
    target_class_id: int,
    segments: np.ndarray,
    config: ExperimentConfig,
    variant: str = "lime",
) -> LimeResult:
    # Keep metric, policy, and segmentation tests lightweight. LIME is required
    # only when an explanation is actually executed.
    from lime import lime_image

    if variant not in {"lime", "lime_lasso"}:
        raise ValueError(f"Unknown LIME variant: {variant}")

    fixed_segments = np.asarray(segments, dtype=np.int32)
    if fixed_segments.shape != image.shape[:2]:
        raise ValueError(
            f"Segmentation shape {fixed_segments.shape} does not match image shape {image.shape[:2]}"
        )

    explainer = lime_image.LimeImageExplainer(random_state=int(config.random_seed))
    kwargs: dict[str, Any] = {
        "image": np.asarray(image, dtype=np.uint8),
        "classifier_fn": predictor.predict_proba,
        "labels": (int(target_class_id),),
        "top_labels": None,
        "hide_color": config.hide_color,
        "num_features": int(len(np.unique(fixed_segments))),
        "num_samples": int(config.lime_num_samples),
        "batch_size": int(config.lime_batch_size),
        "segmentation_fn": lambda _image: fixed_segments.copy(),
        "random_seed": int(config.random_seed),
    }
    if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
        kwargs["progress_bar"] = False
    if variant == "lime_lasso":
        from sklearn.linear_model import Lasso

        kwargs["model_regressor"] = Lasso(
            alpha=config.lime_lasso_alpha, max_iter=10000,
            random_state=config.random_seed,
        )

    started = time.perf_counter()
    explanation = explainer.explain_instance(**kwargs)
    lime_seconds = time.perf_counter() - started

    explanation_segments = np.asarray(explanation.segments, dtype=np.int32)
    positive = ranked_positive_features(explanation, target_class_id)
    mask, selected, area = select_critical_mask(
        explanation_segments,
        positive,
        config.critical_area_fraction,
    )
    return LimeResult(
        explanation=explanation,
        segments=explanation_segments,
        positive_features=positive,
        critical_mask=mask,
        selected_features=selected,
        actual_area_fraction=area,
        surrogate_score=_score_for_label(explanation, target_class_id),
        lime_seconds=float(lime_seconds),
        explainer=variant,
        diagnostics={
            "surrogate": "lasso" if variant == "lime_lasso" else "ridge",
            "lasso_alpha": config.lime_lasso_alpha if variant == "lime_lasso" else None,
            "num_samples": config.lime_num_samples,
            "hide_color": config.hide_color,
            "feature_weights": [
                [int(feature), float(weight)]
                for feature, weight in explanation.local_exp[int(target_class_id)]
            ],
        },
    )
