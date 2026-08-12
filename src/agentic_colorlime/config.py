from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """All non-secret numerical settings used by one explanation run.

    Tool-use policy is intentionally not configured here. The agent sees every
    registered explanation tool and decides how many of them to execute.
    """

    lime_num_samples: int = 500
    lime_batch_size: int = 32
    inference_batch_size: int = 16
    random_seed: int = 42

    # Keep the perturbation policy constant across segmentation methods so that
    # the experiment primarily isolates segmentation behaviour.
    hide_color: int | None = 0
    critical_area_fraction: float = 0.20
    omission_rgb: tuple[int, int, int] = (0, 0, 0)

    slic_n_segments: int = 100
    slic_compactness: float = 10.0
    slic_sigma: float = 1.0

    quickshift_kernel_size: int = 3
    quickshift_max_dist: int = 6
    quickshift_ratio: float = 0.5

    felzenszwalb_scale: float = 100.0
    felzenszwalb_sigma: float = 0.5
    felzenszwalb_min_size: int = 50

    watershed_markers: int = 100
    watershed_compactness: float = 0.001

    colorlime_k: int = 128
    colorlime_n_init: int = 3
    colorlime_max_iter: int = 200
    colorlime_tol: float = 1e-4

    def validate(self) -> None:
        positive_ints = {
            "lime_num_samples": self.lime_num_samples,
            "lime_batch_size": self.lime_batch_size,
            "inference_batch_size": self.inference_batch_size,
            "slic_n_segments": self.slic_n_segments,
            "quickshift_kernel_size": self.quickshift_kernel_size,
            "quickshift_max_dist": self.quickshift_max_dist,
            "felzenszwalb_min_size": self.felzenszwalb_min_size,
            "watershed_markers": self.watershed_markers,
            "colorlime_k": self.colorlime_k,
            "colorlime_n_init": self.colorlime_n_init,
            "colorlime_max_iter": self.colorlime_max_iter,
        }
        for name, value in positive_ints.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 < self.critical_area_fraction <= 1:
            raise ValueError("critical_area_fraction must be in (0, 1]")
        if len(self.omission_rgb) != 3 or any(not 0 <= int(v) <= 255 for v in self.omission_rgb):
            raise ValueError("omission_rgb must contain three values between 0 and 255")
        if self.colorlime_tol <= 0:
            raise ValueError("colorlime_tol must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
