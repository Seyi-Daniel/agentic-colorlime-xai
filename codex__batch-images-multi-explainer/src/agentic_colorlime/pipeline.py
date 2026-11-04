from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import ExperimentConfig
from .image_profile import ImageProfile, profile_image
if TYPE_CHECKING:
    from .model_runner import HuggingFaceImageClassifier
from .openai_agent import AgentDecision, OpenAIAdaptiveAgent
from .runtime import ToolRuntime


@dataclass
class ExperimentResult:
    run_dir: str
    target_class_id: int
    target_class_label: str
    target_probability: float
    image_profile: dict[str, Any]
    decision: AgentDecision
    selected_candidate: dict[str, Any]
    candidates: list[dict[str, Any]]


def _run_directory(output_root: str | Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = Path(output_root) / f"run-{stamp}-{uuid.uuid4().hex[:6]}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def run_experiment(
    *,
    image: np.ndarray,
    model_id_or_path: str,
    openai_api_key: str,
    openai_model: str = "gpt-5-mini",
    hf_token: str | None = None,
    device: str = "auto",
    output_root: str | Path = "outputs",
    send_visuals_to_agent: bool = True,
    config: ExperimentConfig | None = None,
    predictor: HuggingFaceImageClassifier | Any | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    config.validate()
    run_dir = _run_directory(output_root)

    image_array = np.asarray(image, dtype=np.uint8)
    profile: ImageProfile = profile_image(image_array)

    if predictor is None:
        from .model_runner import HuggingFaceImageClassifier

        predictor = HuggingFaceImageClassifier(
            model_id_or_path,
            device=device,
            inference_batch_size=config.inference_batch_size,
            hf_token=hf_token,
        )

    target_class_id, target_class_label, target_probability, probabilities = predictor.predict_one(image_array)

    runtime = ToolRuntime(
        image=image_array,
        image_profile=profile,
        predictor=predictor,
        target_class_id=target_class_id,
        target_class_label=target_class_label,
        original_target_probability=target_probability,
        config=config,
        output_dir=run_dir,
    )
    agent = OpenAIAdaptiveAgent(
        api_key=openai_api_key,
        model=openai_model,
        audit_root=run_dir / "llm_audit",
    )
    try:
        decision = agent.run(runtime, send_visuals_to_agent=send_visuals_to_agent)
    finally:
        # Preserve partial execution records if the agent or a remote request fails.
        runtime.write_outputs()

    selected_candidate = runtime.candidates_by_id[decision.final_candidate_id].public_summary()
    candidates = runtime.all_public_summaries()

    metadata = {
        "model_id_or_path": model_id_or_path,
        "openai_model": openai_model,
        "target_class_id": target_class_id,
        "target_class_label": target_class_label,
        "target_probability": target_probability,
        "top5_predictions": [
            {
                "class_id": int(index),
                "label": predictor.label(int(index)),
                "probability": float(probabilities[int(index)]),
            }
            for index in np.argsort(probabilities)[-5:][::-1]
        ],
        "image_profile": profile.to_dict(),
        "config": config.to_dict(),
        "decision": decision.raw,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExperimentResult(
        run_dir=str(run_dir.resolve()),
        target_class_id=target_class_id,
        target_class_label=target_class_label,
        target_probability=target_probability,
        image_profile=profile.to_dict(),
        decision=decision,
        selected_candidate=selected_candidate,
        candidates=candidates,
    )
