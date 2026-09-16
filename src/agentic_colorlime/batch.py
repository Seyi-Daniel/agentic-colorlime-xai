"""Sequential, independent image runs sharing one loaded classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ExperimentConfig
from .image_io import load_image_from_path, load_image_from_upload
from .pipeline import ExperimentResult, _run_directory, run_experiment


@dataclass
class ImageInput:
    name: str
    source: str | Path | bytes | np.ndarray

    def load(self) -> np.ndarray:
        if isinstance(self.source, bytes):
            return load_image_from_upload(self.source)
        if isinstance(self.source, np.ndarray):
            return self.source
        return load_image_from_path(self.source)


@dataclass
class BatchResult:
    batch_dir: str
    items: list[dict[str, Any]] = field(default_factory=list)
    results: dict[int, ExperimentResult] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "batch_dir": self.batch_dir,
            "succeeded": sum(item["status"] == "completed" for item in self.items),
            "failed": sum(item["status"] == "failed" for item in self.items),
            "items": self.items,
        }


def run_batch(
    *, images: list[ImageInput], model_id_or_path: str, openai_api_key: str,
    openai_model: str = "gpt-5-mini", hf_token: str | None = None,
    device: str = "auto", output_root: str | Path = "outputs",
    send_visuals_to_agent: bool = True, config: ExperimentConfig | None = None,
    predictor: Any | None = None,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> BatchResult:
    if not images:
        raise ValueError("Provide at least one image.")
    config = config or ExperimentConfig()
    config.validate()
    directory = _run_directory(output_root)
    batch = BatchResult(batch_dir=str(directory.resolve()))
    batch.items = [{"index": i, "name": item.name, "status": "pending"}
                   for i, item in enumerate(images)]

    def save() -> None:
        temporary = directory / "batch_summary.tmp"
        temporary.write_text(json.dumps(batch.summary(), indent=2), encoding="utf-8")
        temporary.replace(directory / "batch_summary.json")

    save()
    if predictor is None:
        try:
            from .model_runner import HuggingFaceImageClassifier

            predictor = HuggingFaceImageClassifier(
                model_id_or_path, device=device,
                inference_batch_size=config.inference_batch_size, hf_token=hf_token,
            )
        except Exception as exc:
            for item in batch.items:
                item.update(status="failed", error=f"Classifier initialization failed: {type(exc).__name__}: {exc}")
            save()
            return batch

    for index, image_input in enumerate(images):
        item = batch.items[index]
        # Index-based folders avoid collisions and never treat user filenames as paths.
        image_dir = directory / f"image-{index + 1:04d}"
        image_dir.mkdir()
        item.update(status="running", output_dir=str(image_dir.resolve()))
        save()
        try:
            result = run_experiment(
                image=image_input.load(), model_id_or_path=model_id_or_path,
                openai_api_key=openai_api_key, openai_model=openai_model,
                hf_token=hf_token, device=device, output_root=image_dir,
                send_visuals_to_agent=send_visuals_to_agent, config=config,
                predictor=predictor,
            )
            batch.results[index] = result
            item.update(
                status="completed", run_dir=result.run_dir,
                target_class_label=result.target_class_label,
                target_probability=result.target_probability,
                selected_candidate=result.selected_candidate,
                decision=result.decision.raw,
            )
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        if on_progress is not None:
            on_progress(index + 1, len(images), dict(item))
    return batch
