from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIRResult:
    original_target_probability: float
    omitted_target_probability: float
    cir: float
    relative_cir: float
    cir_per_removed_area: float
    decision_changed: bool
    omitted_top1_class_id: int
    omitted_top1_label: str
    omitted_image: np.ndarray


def omit_region(
    image: np.ndarray,
    mask: np.ndarray,
    omission_rgb: tuple[int, int, int],
) -> np.ndarray:
    omitted = np.asarray(image, dtype=np.uint8).copy()
    omitted[np.asarray(mask, dtype=bool)] = np.asarray(omission_rgb, dtype=np.uint8)
    return omitted


def calculate_cir(
    *,
    image: np.ndarray,
    critical_mask: np.ndarray,
    target_class_id: int,
    original_target_probability: float,
    predictor,
    omission_rgb: tuple[int, int, int],
) -> CIRResult:
    """Compute the project's confidence-impact metric for one explanation."""
    omitted = omit_region(image, critical_mask, omission_rgb)
    probabilities = predictor.predict_proba(omitted)[0]
    after = float(probabilities[int(target_class_id)])
    cir = max(float(original_target_probability) - after, 0.0)
    relative = cir / max(float(original_target_probability), 1e-12)
    area = float(np.asarray(critical_mask, dtype=bool).mean())
    omitted_top1 = int(np.argmax(probabilities))
    return CIRResult(
        original_target_probability=float(original_target_probability),
        omitted_target_probability=after,
        cir=float(cir),
        relative_cir=float(relative),
        cir_per_removed_area=float(cir / max(area, 1e-12)),
        decision_changed=bool(omitted_top1 != int(target_class_id)),
        omitted_top1_class_id=omitted_top1,
        omitted_top1_label=predictor.label(omitted_top1),
        omitted_image=omitted,
    )
