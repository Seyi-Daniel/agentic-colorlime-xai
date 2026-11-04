from __future__ import annotations

import json
import inspect
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cir import CIRResult, calculate_cir
from .config import ExperimentConfig
from .image_io import image_to_data_url
from .image_profile import ImageProfile
from .lime_engine import LimeResult, run_lime
from .shap_engine import run_shap
from .segmentations import SEGMENTATION_FUNCTIONS, SegmentationResult
from .tool_catalog import ToolCatalog
from .visualization import (
    explanation_overlay,
    mask_on_white,
    save_mask,
    save_rgb,
    segmentation_preview,
)


@dataclass
class Candidate:
    candidate_id: str
    method: str
    segmentation: SegmentationResult
    lime: LimeResult
    segmentation_seconds: float
    artifacts: dict[str, str]
    selection_rationale: str
    uncertainty_addressed: str
    expected_signal: str
    pre_call_confidence: float
    cir: CIRResult | None = None
    evidence_review: dict[str, Any] | None = None

    def public_summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "explainer": self.lime.explainer,
            "segmentation_method": self.segmentation.method,
            "selection_rationale": self.selection_rationale,
            "uncertainty_addressed": self.uncertainty_addressed,
            "expected_signal": self.expected_signal,
            "pre_call_confidence": float(self.pre_call_confidence),
            "parameters": self.segmentation.params,
            "segmentation_metadata": self.segmentation.metadata,
            "number_of_segments": int(len(np.unique(self.segmentation.labels))),
            "positive_feature_count": int(len(self.lime.positive_features)),
            "selected_feature_count": int(len(self.lime.selected_features)),
            "critical_area_fraction": float(self.lime.actual_area_fraction),
            "local_surrogate_score": self.lime.surrogate_score,
            "explanation_diagnostics": self.lime.diagnostics,
            "segmentation_seconds": float(self.segmentation_seconds),
            "explanation_seconds": float(self.lime.lime_seconds),
            "total_seconds": float(self.segmentation_seconds + self.lime.lime_seconds),
            "artifacts": dict(self.artifacts),
        }
        if self.lime.explainer in {"lime", "lime_lasso"}:
            payload["lime_local_surrogate_score"] = self.lime.surrogate_score
            payload["lime_seconds"] = float(self.lime.lime_seconds)
        if self.cir is not None:
            payload.update(
                {
                    "cir": float(self.cir.cir),
                    "relative_cir": float(self.cir.relative_cir),
                    "cir_per_removed_area": float(self.cir.cir_per_removed_area),
                    "original_target_probability": float(self.cir.original_target_probability),
                    "omitted_target_probability": float(self.cir.omitted_target_probability),
                    "decision_changed": bool(self.cir.decision_changed),
                    "omitted_top1_class_id": int(self.cir.omitted_top1_class_id),
                    "omitted_top1_label": self.cir.omitted_top1_label,
                }
            )
        else:
            payload["cir_status"] = "not_calculated"
        if self.evidence_review is not None:
            payload["evidence_review"] = dict(self.evidence_review)
        return payload


class ToolRuntime:
    """Local state while the remote LLM adaptively requests tools.

    No application-level minimum, maximum, shortlist, or expansion budget is
    imposed. Each registered explanation method can be attempted at most once
    in a run, and the agent decides when sufficient evidence exists to stop.
    """

    def __init__(
        self,
        *,
        image: np.ndarray,
        image_profile: ImageProfile,
        predictor: Any,
        target_class_id: int,
        target_class_label: str,
        original_target_probability: float,
        config: ExperimentConfig,
        output_dir: str | Path,
        catalog: ToolCatalog | None = None,
    ) -> None:
        self.image = np.asarray(image, dtype=np.uint8)
        self.image_profile = image_profile
        self.predictor = predictor
        self.target_class_id = int(target_class_id)
        self.target_class_label = str(target_class_label)
        self.original_target_probability = float(original_target_probability)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or ToolCatalog()

        registered_functions = set(SEGMENTATION_FUNCTIONS)
        registered_cards = {card.segmentation_method for card in self.catalog.all_cards()}

        if not registered_functions:
            raise RuntimeError(
                "SEGMENTATION_FUNCTIONS is empty. "
                "Register the executable segmentation functions at the bottom "
                "of agentic_colorlime/segmentations.py."
            )

        missing_cards = registered_functions - registered_cards
        if missing_cards:
            raise ValueError(
                "Executable segmentation methods are missing ToolCard metadata: "
                f"{sorted(missing_cards)}"
            )

        missing_functions = registered_cards - registered_functions
        if missing_functions:
            raise ValueError(
                "ToolCard entries do not have matching executable functions in "
                "SEGMENTATION_FUNCTIONS: "
                f"{sorted(missing_functions)}"
            )

        self.candidates_by_id: dict[str, Candidate] = {}
        self.candidate_id_by_method: dict[str, str] = {}
        self.failed_methods: dict[str, str] = {}
        self.failed_cir_methods: dict[str, str] = {}
        self.trace: list[dict[str, Any]] = []
        self.pending_cir_candidate_id: str | None = None
        self.pending_review_candidate_id: str | None = None
        self.inspected_methods: set[str] = set()
        self.finish_authorized = False
        self.last_evidence_review: dict[str, Any] | None = None
        self.selected_explainer: str | None = None

        save_rgb(self.image, self.output_dir / "input_image.png")

    @property
    def explanation_calls_used(self) -> int:
        return len(self.candidate_id_by_method) + len(self.failed_methods)

    @property
    def total_registered_tools(self) -> int:
        return len(self.available_methods)

    @property
    def available_methods(self) -> set[str]:
        return set(self.catalog.cards)

    def available_explainers(self) -> list[str]:
        return sorted({self.catalog.get(m).explainer for m in self.unattempted_methods()})

    def select_explainer(self, explainer: str, rationale: str) -> dict[str, Any]:
        if self.pending_cir_candidate_id or self.pending_review_candidate_id:
            raise RuntimeError("Complete the pending CIR and evidence review first.")
        if self.selected_explainer is not None:
            raise RuntimeError("Execute a segmentation for the selected explainer first.")
        if explainer not in self.available_explainers():
            raise ValueError(f"No unattempted candidates for explainer: {explainer}")
        if not str(rationale).strip():
            raise ValueError("An explainer selection rationale is required.")
        self.selected_explainer = explainer
        self.finish_authorized = False
        result = {"explainer": explainer, "rationale": str(rationale).strip()}
        self.trace.append({"event": "explainer_selection", "tool": "select_explainer", "result": result})
        return result

    def attempted_methods(self) -> set[str]:
        return set(self.candidate_id_by_method) | set(self.failed_methods)

    def generated_methods(self) -> list[str]:
        return list(self.candidate_id_by_method.keys())

    def successfully_evaluated_methods(self) -> list[str]:
        return [candidate.method for candidate in self.evaluated_candidates()]

    def executed_methods(self) -> list[str]:
        ordered = [*self.candidate_id_by_method.keys(), *self.failed_methods.keys()]
        return list(dict.fromkeys(ordered))

    def unattempted_methods(self) -> list[str]:
        return sorted(self.available_methods - self.attempted_methods())

    def uninspected_methods(self) -> list[str]:
        return sorted(self.available_methods - self.inspected_methods)

    def inspect_method_source(self, method: str) -> dict[str, Any]:
        """Return the exact registered wrapper source and active configuration."""
        if method not in self.catalog.cards:
            raise KeyError(f"Unknown method: {method}")
        if method in self.inspected_methods:
            raise RuntimeError(f"The source for {method} has already been inspected.")

        card = self.catalog.get(method)
        function = SEGMENTATION_FUNCTIONS[card.segmentation_method]
        result = {
            "method": method,
            "registered_function": f"{function.__module__}.{function.__name__}",
            "source_code": inspect.getsource(function),
            "explainer": card.explainer,
            "explainer_source_code": inspect.getsource(run_shap if card.explainer == "shap" else run_lime),
            "active_experiment_config": self.config.to_dict(),
            "note": (
                "This is the local executable wrapper registered for this run. "
                "Use the code and active settings as prior evidence only; actual "
                "performance is observed only after execution."
            ),
        }
        self.inspected_methods.add(method)
        self.trace.append(
            {
                "event": "source_inspection",
                "tool": "inspect_tool_source",
                "method": method,
                "result": result,
            }
        )
        return result

    def record_method_failure(self, method: str, error: str) -> None:
        if method not in self.available_methods:
            return
        if method in self.candidate_id_by_method:
            return
        self.failed_methods[method] = str(error)
        self.selected_explainer = None
        # Re-review existing evidence if the final remaining attempt fails.
        evaluated = self.evaluated_candidates()
        if not self.unattempted_methods() and evaluated:
            self.pending_review_candidate_id = evaluated[-1].candidate_id
        self.trace.append(
            {
                "event": "explanation_tool_failure",
                "tool": self.catalog.get(method).function_name,
                "method": method,
                "error": str(error),
            }
        )


    def record_cir_failure(self, candidate_id: str, error: str) -> None:
        candidate = self.candidates_by_id.get(candidate_id)
        if candidate is not None:
            self.failed_cir_methods[candidate.method] = str(error)
        self.pending_cir_candidate_id = None
        previously_evaluated = self.evaluated_candidates()
        self.pending_review_candidate_id = (
            previously_evaluated[-1].candidate_id
            if previously_evaluated
            else None
        )
        self.finish_authorized = False
        self.trace.append(
            {
                "event": "cir_tool_failure",
                "tool": "calculate_cir",
                "candidate_id": candidate_id,
                "method": None if candidate is None else candidate.method,
                "error": str(error),
            }
        )

    def run_method(
        self,
        method: str,
        *,
        selection_rationale: str,
        uncertainty_addressed: str,
        expected_signal: str,
        confidence: float,
    ) -> dict[str, Any]:
        if self.pending_cir_candidate_id is not None:
            raise RuntimeError("Calculate CIR for the pending candidate before running another explanation tool.")
        if self.pending_review_candidate_id is not None:
            raise RuntimeError("Review the pending candidate evidence before running another explanation tool.")
        if method not in self.catalog.cards:
            raise KeyError(f"Unknown method: {method}")
        card = self.catalog.get(method)
        if card.explainer != self.selected_explainer:
            raise RuntimeError("Select the candidate's explainer before its segmentation.")
        if method in self.attempted_methods():
            raise RuntimeError(f"{method} has already been attempted; select a different method or finish.")

        started = time.perf_counter()
        segmentation = SEGMENTATION_FUNCTIONS[card.segmentation_method](self.image, self.config)
        segmentation_seconds = time.perf_counter() - started
        engine = run_shap if card.explainer == "shap" else run_lime
        engine_options = {} if card.explainer == "shap" else {"variant": card.explainer}
        lime_result = engine(
            image=self.image,
            predictor=self.predictor,
            target_class_id=self.target_class_id,
            segments=segmentation.labels,
            config=self.config,
            **engine_options,
        )

        candidate_id = f"{method}-{uuid.uuid4().hex[:8]}"
        method_dir = self.output_dir / method
        artifacts = {
            "segmentation_preview": save_rgb(
                segmentation_preview(self.image, segmentation.labels),
                method_dir / "segmentation.png",
            ),
            "explanation_overlay": save_rgb(
                explanation_overlay(self.image, lime_result.critical_mask),
                method_dir / "explanation.png",
            ),
            "explanation_on_white": save_rgb(
                mask_on_white(self.image, lime_result.critical_mask),
                method_dir / "explanation_on_white.png",
            ),
            "critical_mask": save_mask(
                lime_result.critical_mask,
                method_dir / "critical_mask.png",
            ),
        }
        candidate = Candidate(
            candidate_id=candidate_id,
            method=method,
            segmentation=segmentation,
            lime=lime_result,
            segmentation_seconds=segmentation_seconds,
            artifacts=artifacts,
            selection_rationale=str(selection_rationale),
            uncertainty_addressed=str(uncertainty_addressed),
            expected_signal=str(expected_signal),
            pre_call_confidence=float(np.clip(confidence, 0.0, 1.0)),
        )
        self.candidates_by_id[candidate_id] = candidate
        self.candidate_id_by_method[method] = candidate_id
        self.pending_cir_candidate_id = candidate_id
        self.selected_explainer = None
        self.finish_authorized = False
        self.last_evidence_review = None

        result = candidate.public_summary()
        self.trace.append(
            {
                "event": "explanation_tool",
                "tool": self.catalog.get(method).function_name,
                "arguments": {
                    "selection_rationale": selection_rationale,
                    "uncertainty_addressed": uncertainty_addressed,
                    "expected_signal": expected_signal,
                    "confidence": confidence,
                },
                "result": result,
            }
        )
        return result

    def calculate_candidate_cir(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id not in self.candidates_by_id:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        if self.pending_cir_candidate_id is not None and candidate_id != self.pending_cir_candidate_id:
            raise RuntimeError(
                f"CIR must be calculated for pending candidate {self.pending_cir_candidate_id}, not {candidate_id}."
            )

        candidate = self.candidates_by_id[candidate_id]
        if candidate.cir is None:
            candidate.cir = calculate_cir(
                image=self.image,
                critical_mask=candidate.lime.critical_mask,
                target_class_id=self.target_class_id,
                original_target_probability=self.original_target_probability,
                predictor=self.predictor,
                omission_rgb=self.config.omission_rgb,
            )
            candidate.artifacts["omitted_image"] = save_rgb(
                candidate.cir.omitted_image,
                self.output_dir / candidate.method / "cir_omitted_image.png",
            )
        self.pending_cir_candidate_id = None
        self.pending_review_candidate_id = candidate_id
        self.finish_authorized = False
        result = candidate.public_summary()
        result["interpretation_note"] = (
            "CIR is corroborating evidence. A larger value means the selected region had a larger "
            "observed effect on the target probability, but it is not an automatic winner rule."
        )
        self.trace.append(
            {
                "event": "cir_tool",
                "tool": "calculate_cir",
                "arguments": {"candidate_id": candidate_id},
                "result": result,
            }
        )
        return result

    def review_candidate_evidence(
        self,
        candidate_id: str,
        *,
        evidence_summary: str,
        cir_assessment: str,
        counterevidence_considered: str,
        unresolved_uncertainty: str,
        next_action: str,
        action_reason: str,
    ) -> dict[str, Any]:
        """Record the required post-CIR reflection before stop/continue is possible."""
        if candidate_id not in self.candidates_by_id:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        if candidate_id != self.pending_review_candidate_id:
            raise RuntimeError(
                f"Evidence review is required for pending candidate "
                f"{self.pending_review_candidate_id}, not {candidate_id}."
            )
        candidate = self.candidates_by_id[candidate_id]
        if candidate.cir is None:
            raise RuntimeError("CIR must be calculated before evidence review.")
        if cir_assessment not in {
            "supports",
            "partially_supports",
            "contradicts",
            "inconclusive",
        }:
            raise ValueError(f"Unsupported CIR assessment: {cir_assessment}")
        if next_action not in {"ready_to_finish", "try_another_method"}:
            raise ValueError(f"Unsupported next action: {next_action}")
        if next_action == "try_another_method" and not self.unattempted_methods():
            raise RuntimeError(
                "No unattempted explanation method remains. Review the evidence "
                "again and choose ready_to_finish."
            )
        if next_action == "ready_to_finish" and str(unresolved_uncertainty).strip():
            raise ValueError(
                "ready_to_finish requires an empty unresolved_uncertainty. "
                "Choose try_another_method while a material uncertainty remains."
            )

        review = {
            "candidate_id": candidate_id,
            "method": candidate.method,
            "evidence_summary": str(evidence_summary).strip(),
            "cir_assessment": cir_assessment,
            "counterevidence_considered": str(counterevidence_considered).strip(),
            "unresolved_uncertainty": str(unresolved_uncertainty).strip(),
            "next_action": next_action,
            "action_reason": str(action_reason).strip(),
        }
        if not review["evidence_summary"]:
            raise ValueError("A non-empty evidence summary is required.")
        if not review["counterevidence_considered"]:
            raise ValueError("Counterevidence considered must be stated.")
        if not review["action_reason"]:
            raise ValueError("A non-empty action reason is required.")

        candidate.evidence_review = review
        self.last_evidence_review = review
        self.pending_review_candidate_id = None
        self.finish_authorized = next_action == "ready_to_finish"
        self.trace.append(
            {
                "event": "evidence_review",
                "tool": "review_candidate_evidence",
                "arguments": review,
                "result": {
                    "status": "recorded",
                    "finish_authorized": self.finish_authorized,
                    "unattempted_methods": self.unattempted_methods(),
                },
            }
        )
        return {
            "status": "recorded",
            "finish_authorized": self.finish_authorized,
            "review": review,
            "unattempted_methods": self.unattempted_methods(),
        }

    def candidate_overlay_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        return image_to_data_url(explanation_overlay(self.image, candidate.lime.critical_mask))

    def candidate_segmentation_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        return image_to_data_url(
            segmentation_preview(self.image, candidate.segmentation.labels)
        )

    def candidate_omitted_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        if candidate.cir is None:
            raise RuntimeError("The candidate has no CIR omitted image.")
        return image_to_data_url(candidate.cir.omitted_image)

    def evaluated_candidates(self) -> list[Candidate]:
        return [candidate for candidate in self.candidates_by_id.values() if candidate.cir is not None]

    def all_public_summaries(self) -> list[dict[str, Any]]:
        return [candidate.public_summary() for candidate in self.candidates_by_id.values()]

    def best_observed_cir_candidate(self) -> Candidate:
        valid = self.evaluated_candidates()
        if not valid:
            raise RuntimeError("No candidate has a CIR result")
        return max(valid, key=lambda candidate: float(candidate.cir.cir))

    def evidence_comparison(self, selected_candidate_id: str) -> dict[str, Any]:
        if selected_candidate_id not in self.candidates_by_id:
            raise KeyError(selected_candidate_id)
        selected = self.candidates_by_id[selected_candidate_id]
        if selected.cir is None:
            raise RuntimeError("Selected candidate has no CIR result")
        ordered = sorted(self.evaluated_candidates(), key=lambda item: item.cir.cir, reverse=True)
        best = ordered[0]
        rank = next(index for index, item in enumerate(ordered, start=1) if item.candidate_id == selected_candidate_id)
        return {
            "selected_cir": float(selected.cir.cir),
            "highest_observed_cir": float(best.cir.cir),
            "highest_observed_method": best.method,
            "selected_cir_rank": int(rank),
            "evaluated_candidate_count": len(ordered),
            "absolute_gap_from_highest": float(best.cir.cir - selected.cir.cir),
            "selected_is_highest_observed_cir": bool(best.candidate_id == selected_candidate_id),
        }

    def write_outputs(self) -> None:
        (self.output_dir / "candidate_results.json").write_text(
            json.dumps(self.all_public_summaries(), indent=2), encoding="utf-8"
        )
        (self.output_dir / "agent_tool_trace.json").write_text(
            json.dumps(self.trace, indent=2), encoding="utf-8"
        )
        timeline = []
        for index, event in enumerate(self.trace, start=1):
            event_type = str(event.get("event", "unknown"))
            compact: dict[str, Any] = {
                "step": index,
                "event": event_type,
                "tool": event.get("tool"),
            }
            if event_type == "source_inspection":
                compact.update(
                    {
                        "method": event.get("method"),
                        "decision": "Inspected local source before choosing.",
                    }
                )
            elif event_type == "explainer_selection":
                compact.update(event.get("result", {}))
            elif event_type == "explanation_tool":
                result = event.get("result", {})
                compact.update(
                    {
                        "method": result.get("method"),
                        "candidate_id": result.get("candidate_id"),
                        "reason": event.get("arguments", {}).get(
                            "selection_rationale"
                        ),
                    }
                )
            elif event_type == "cir_tool":
                result = event.get("result", {})
                compact.update(
                    {
                        "method": result.get("method"),
                        "candidate_id": result.get("candidate_id"),
                        "cir": result.get("cir"),
                        "relative_cir": result.get("relative_cir"),
                        "decision_changed": result.get("decision_changed"),
                    }
                )
            elif event_type == "evidence_review":
                arguments = event.get("arguments", {})
                compact.update(
                    {
                        "method": arguments.get("method"),
                        "candidate_id": arguments.get("candidate_id"),
                        "cir_assessment": arguments.get("cir_assessment"),
                        "next_action": arguments.get("next_action"),
                        "reason": arguments.get("action_reason"),
                    }
                )
            elif event_type == "final_selection":
                result = event.get("result", {})
                compact.update(
                    {
                        "method": result.get("selected_method"),
                        "candidate_id": result.get("selected_candidate_id"),
                        "reason": result.get("stopping_reason"),
                    }
                )
            else:
                compact.update(
                    {
                        "method": event.get("method"),
                        "reason": event.get("error"),
                    }
                )
            timeline.append(compact)
        (self.output_dir / "decision_timeline.json").write_text(
            json.dumps(timeline, indent=2), encoding="utf-8"
        )
