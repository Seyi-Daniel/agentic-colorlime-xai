from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig


def load_profile(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load a named YAML profile and validate its ExperimentConfig section."""
    profile_path = Path(path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration profile must be a mapping: {profile_path}")

    experiment = payload.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("The 'experiment' section must be a mapping")

    allowed = {item.name for item in fields(ExperimentConfig)}
    unknown = sorted(set(experiment) - allowed)
    if unknown:
        raise ValueError(f"Unknown experiment setting(s): {', '.join(unknown)}")

    values = dict(experiment)
    if "omission_rgb" in values:
        values["omission_rgb"] = tuple(values["omission_rgb"])

    config = ExperimentConfig(**values)
    config.validate()
    payload["_profile_path"] = str(profile_path.resolve())
    return config, payload


def apply_overrides(config: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    """Return a validated config with only non-None command-line overrides applied."""
    updated = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    updated.validate()
    return updated

