from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

try:
    from openai import OpenAI
except ImportError:  # Allows local unit tests that do not call the API.
    OpenAI = None  # type: ignore[assignment]

from .image_io import image_to_data_url

if TYPE_CHECKING:
    from .runtime import ToolRuntime
from .tool_catalog import EXPLAINERS, ToolCard


BASE_INSTRUCTIONS = """
You are the controller of an adaptive image-explanation experiment.

You have local source-inspection, explanation, CIR-evaluation, evidence-review,
and final-selection tools. Use one tool per step.

Rules:
- First select an explainer (LIME, sparse LIME with Lasso, or Kernel SHAP),
  then choose one of its available segmentations. Repeat that hierarchy for
  each comparison. The classifier and target class stay fixed for this image.
- A SHAP additivity residual checks accounting, not explanation quality.
  Do not compare SHAP diagnostics with LIME's surrogate R-squared as one score.
- Compare CIR with the actual removed area and masking baseline in mind.
- Tool names and inspected source code are prior evidence, not observed quality.
- Inspect source only when it would reduce uncertainty about a method.
- Before executing an explanation method, state the current reason for trying it
  and what result would support or weaken that choice.
- After every explanation, calculate CIR and then review the combined evidence.
- CIR is evidence, not an automatic winner rule.
- In evidence review, actively consider evidence against the current candidate.
- Continue only when a specific material uncertainty remains.
- Finish only after an evidence review authorizes finishing.
- Never claim to have evaluated an uncalled method.
- Never invent image content, measurements, source code, or tool results.
- Use short, plain language and concise evidence-based rationales.

There is no application-set minimum or maximum number of explanation methods.
You decide whether one, several, or all unattempted methods are needed.
""".strip()


MAX_CONSECUTIVE_PROTOCOL_ERRORS = 3


def _method_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selection_rationale": {
                "type": "string",
                "description": (
                    "A short reason for trying this method now, based only on "
                    "available image/model context, inspected source, and prior "
                    "executed results."
                ),
            },
            "uncertainty_addressed": {
                "type": "string",
                "description": (
                    "The specific uncertainty this method is intended to resolve. "
                    "For the first method, state what makes it a useful first test."
                ),
            },
            "expected_signal": {
                "type": "string",
                "description": (
                    "What observed result would support the choice, and what result "
                    "would make another method worth trying."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Pre-execution confidence that this is a suitable next "
                    "tool. This is not confidence that it will be the final "
                    "selected explanation."
                ),
            },
        },
        "required": [
            "selection_rationale",
            "uncertainty_addressed",
            "expected_signal",
            "confidence",
        ],
        "additionalProperties": False,
    }


def method_tool(card: ToolCard) -> dict[str, Any]:
    return {
        "type": "function",
        "name": card.function_name,
        "description": (
            f"Execute the '{card.segmentation_method}' segmentation function "
            f"and create one '{card.explainer}' explanation candidate."
        ),
        "parameters": _method_parameters(),
        "strict": True,
    }


def explainer_tool(explainers: list[str]) -> dict[str, Any]:
    return {
        "type": "function", "name": "select_explainer", "strict": True,
        "description": "Choose an explanation algorithm before choosing its segmentation. "
                       + json.dumps({name: EXPLAINERS[name] for name in explainers}),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "explainer": {"type": "string", "enum": explainers},
                "rationale": {"type": "string", "description": "Why try this explainer now?"},
            },
            "required": ["explainer", "rationale"],
        },
    }


def inspect_source_tool(methods: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "inspect_tool_source",
        "description": (
            "Read the exact local Python source of one registered segmentation "
            "function plus the active experiment settings. This does not execute "
            "the segmentation or create an explanation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": methods,
                    "description": "The registered method whose local source is needed.",
                },
                "inspection_reason": {
                    "type": "string",
                    "description": (
                        "A short statement of what uncertainty reading this source "
                        "is expected to resolve."
                    ),
                },
            },
            "required": ["method", "inspection_reason"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def cir_tool(candidate_id: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "calculate_cir",
        "description": (
            f"Calculate CIR for the newly created candidate {candidate_id}. This removes the strongest "
            "positive-attribution regions up to the area target and measures the target-class probability drop. "
            "It is evidence, not an automatic selection rule."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": f"Must be exactly {candidate_id}.",
                }
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def evidence_review_tool(candidate_id: str, method: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "review_candidate_evidence",
        "description": (
            f"Review the explanation and CIR evidence for candidate {candidate_id} "
            f"({method}). This required step decides whether evidence is sufficient "
            "to finish or whether another unattempted method should be tried."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": f"Must be exactly {candidate_id}.",
                },
                "evidence_summary": {
                    "type": "string",
                    "description": (
                        "A short summary of the observed explanation, explainer diagnostics, CIR, "
                        "runtime, and visual evidence that was actually supplied."
                    ),
                },
                "cir_assessment": {
                    "type": "string",
                    "enum": [
                        "supports",
                        "partially_supports",
                        "contradicts",
                        "inconclusive",
                    ],
                    "description": "How CIR affects confidence in this candidate.",
                },
                "counterevidence_considered": {
                    "type": "string",
                    "description": (
                        "State the strongest observed weakness, contradictory signal, "
                        "or reason the current evidence could be misleading. If none "
                        "was observed, say that explicitly."
                    ),
                },
                "unresolved_uncertainty": {
                    "type": "string",
                    "description": (
                        "A material question another method should resolve. Use an "
                        "empty string only when evidence is sufficient to finish."
                    ),
                },
                "next_action": {
                    "type": "string",
                    "enum": ["ready_to_finish", "try_another_method"],
                    "description": (
                        "Choose ready_to_finish only when no material uncertainty "
                        "remains; otherwise choose try_another_method."
                    ),
                },
                "action_reason": {
                    "type": "string",
                    "description": (
                        "A short reason why stopping or another execution is the "
                        "appropriate next action."
                    ),
                },
            },
            "required": [
                "candidate_id",
                "evidence_summary",
                "cir_assessment",
                "counterevidence_considered",
                "unresolved_uncertainty",
                "next_action",
                "action_reason",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def finish_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "finish_selection",
        "description": (
            "Record the final explanation after the required evidence review has "
            "authorized finishing. Select only from candidates with completed CIR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selected_candidate_id": {
                    "type": "string",
                    "description": "Candidate ID of an evaluated explanation that already has a CIR result.",
                },
                "decision_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the final selection based on observed evidence.",
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "Concise evidence-based justification for selecting this explanation, including "
                        "the relevant qualitative, explainer, CIR, and cost evidence that was actually observed."
                    ),
                },
                "cir_assessment": {
                    "type": "string",
                    "enum": ["supports", "partially_supports", "contradicts", "inconclusive"],
                    "description": "How the selected candidate's CIR relates to the rest of the judgement.",
                },
                "stopping_reason": {
                    "type": "string",
                    "description": (
                        "Specific reason no further explanation tool is needed. State what evidence is already "
                        "sufficient or why remaining tools are unlikely to resolve a meaningful uncertainty."
                    ),
                },
                "why_not_highest_cir": {
                    "type": "string",
                    "description": (
                        "If another evaluated candidate has higher CIR, explain why it was not selected. "
                        "Otherwise use an empty string."
                    ),
                },
                "alternative_xai_suggestion": {
                    "type": "string",
                    "description": "Optional unexecuted XAI alternative, or 'none'.",
                },
                "alternative_xai_reason": {
                    "type": "string",
                    "description": "Short reason for the optional alternative, or an empty string.",
                },
            },
            "required": [
                "selected_candidate_id",
                "decision_confidence",
                "rationale",
                "cir_assessment",
                "stopping_reason",
                "why_not_highest_cir",
                "alternative_xai_suggestion",
                "alternative_xai_reason",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


@dataclass
class AgentDecision:
    raw: dict[str, Any]
    final_candidate_id: str
    final_method: str
    trace: list[dict[str, Any]]
    evidence_comparison: dict[str, Any]
    tools_executed: list[str]
    tools_failed: dict[str, str]
    tools_not_executed: list[str]
    explanation_calls_used: int
    total_registered_tools: int
    audit_directory: str


class OpenAIAdaptiveAgent:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-mini",
        audit_root: str | Path = "llm_audit",
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required")

        if OpenAI is None:
            raise ImportError(
                "The openai package is required. "
                "Install the project requirements."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.audit_root = Path(audit_root)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Convert SDK objects and Python objects into JSON-safe values."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): OpenAIAdaptiveAgent._to_jsonable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                OpenAIAdaptiveAgent._to_jsonable(item)
                for item in value
            ]

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump(mode="json")
            except TypeError:
                return model_dump()

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return OpenAIAdaptiveAgent._to_jsonable(to_dict())

        return str(value)


    @classmethod
    def _write_audit_json(
        cls,
        path: Path,
        payload: Any,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(
                cls._to_jsonable(payload),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _append_audit_event(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _initial_prompt(
        runtime: ToolRuntime,
        *,
        visual_context_available: bool,
    ) -> str:
        registered = [card.to_dict() for card in runtime.catalog.all_cards()]
        visual_note = (
            "The original image is attached. Later rounds will attach generated "
            "segmentation, explanation, and omitted-image evidence."
            if visual_context_available
            else (
                "No image pixels are supplied to you. Do not make claims about "
                "objects, boundaries, highlighted regions, or visual coherence. "
                "Use only the supplied image profile and tool measurements."
            )
        )
        return (
            f"Task: explain the prediction '{runtime.target_class_label}' "
            f"(class {runtime.target_class_id}, probability {runtime.original_target_probability:.6f}).\n\n"
            f"Cheap image profile: {runtime.image_profile.qualitative_summary()}\n"
            f"Detailed profile: {json.dumps(runtime.image_profile.to_dict(), indent=2)}\n\n"
            f"All {runtime.total_registered_tools} registered explanation tools are available. "
            "There is no application-set tool-call budget. You decide how many methods are necessary.\n"
            f"Registered identifiers:\n{json.dumps(registered, indent=2)}\n\n"
            f"Visual context: {visual_note}\n\n"
            "Select an explainer first. Then inspect source if useful and execute a segmentation."
        )

    @staticmethod
    def _round_instructions(
        runtime: ToolRuntime,
        *,
        visual_context_available: bool,
    ) -> str:
        state = {
            "selected_explainer": runtime.selected_explainer,
            "explanation_tools_attempted": runtime.explanation_calls_used,
            "total_registered_explanation_tools": runtime.total_registered_tools,
            "pending_cir_candidate_id": runtime.pending_cir_candidate_id,
            "pending_review_candidate_id": runtime.pending_review_candidate_id,
            "finish_authorized_by_review": runtime.finish_authorized,
            "inspected_source_methods": sorted(runtime.inspected_methods),
            "generated_methods": runtime.generated_methods(),
            "successful_cir_methods": runtime.successfully_evaluated_methods(),
            "failed_methods": runtime.failed_methods,
            "failed_cir_methods": runtime.failed_cir_methods,
            "unattempted_methods": runtime.unattempted_methods(),
            "evaluated_candidates": runtime.all_public_summaries(),
            "last_evidence_review": runtime.last_evidence_review,
            "visual_context_available": visual_context_available,
        }
        visual_rule = (
            "Visual evidence is available; distinguish visual observations from numeric results."
            if visual_context_available
            else (
                "Visual evidence is unavailable. Do not assert anything about visible "
                "objects, boundaries, highlighted regions, or spatial coherence."
            )
        )
        return (
            BASE_INSTRUCTIONS
            + "\n\n"
            + visual_rule
            + "\n\nCURRENT STATE:\n"
            + json.dumps(state, indent=2)
        )

    @staticmethod
    def _dynamic_tools(runtime: ToolRuntime) -> tuple[list[dict[str, Any]], dict[str, str]]:
        method_by_function: dict[str, str] = {}

        if runtime.pending_cir_candidate_id is not None:
            return [cir_tool(runtime.pending_cir_candidate_id)], method_by_function

        if runtime.pending_review_candidate_id is not None:
            candidate = runtime.candidates_by_id[runtime.pending_review_candidate_id]
            return [
                evidence_review_tool(candidate.candidate_id, candidate.method)
            ], method_by_function

        tools: list[dict[str, Any]] = []
        if runtime.selected_explainer is None:
            explainers = runtime.available_explainers()
            if explainers:
                tools.append(explainer_tool(explainers))
            if runtime.finish_authorized:
                tools.append(finish_tool())
            if tools:
                return tools, method_by_function
        eligible = [m for m in runtime.unattempted_methods()
                    if runtime.catalog.get(m).explainer == runtime.selected_explainer]
        uninspected = [
            method
            for method in runtime.uninspected_methods()
            if method in eligible
        ]
        if uninspected:
            tools.append(inspect_source_tool(uninspected))

        for method in eligible:
            card = runtime.catalog.get(method)
            tools.append(method_tool(card))
            method_by_function[card.function_name] = card.method

        if runtime.finish_authorized:
            tools.append(finish_tool())

        if not tools:
            diagnostic = {
                "registered_function_methods": sorted(runtime.available_methods),
                "catalog_methods": sorted(runtime.catalog.cards),
                "attempted_methods": sorted(runtime.attempted_methods()),
                "unattempted_methods": runtime.unattempted_methods(),
                "pending_cir_candidate_id": runtime.pending_cir_candidate_id,
                "pending_review_candidate_id": runtime.pending_review_candidate_id,
                "finish_authorized": runtime.finish_authorized,
                "candidate_ids": sorted(runtime.candidates_by_id),
                "failed_methods": dict(runtime.failed_methods),
            }

            if runtime.candidates_by_id:
                raise RuntimeError(
                    "No candidate has a successful CIR result, and no unattempted "
                    "explanation tool remains.\n"
                    f"Diagnostic state: {json.dumps(diagnostic, indent=2)}"
                )

            raise RuntimeError(
                "No registered explanation tool is available.\n"
                f"Diagnostic state: {json.dumps(diagnostic, indent=2)}"
            )
        return tools, method_by_function

    @staticmethod
    def _validate_finish(arguments: dict[str, Any], runtime: ToolRuntime) -> tuple[bool, str]:
        if not runtime.finish_authorized:
            return False, (
                "Finishing has not been authorized by a completed evidence review."
            )
        if runtime.pending_cir_candidate_id is not None:
            return False, "Calculate the pending CIR before finishing."
        if runtime.pending_review_candidate_id is not None:
            return False, "Complete the pending evidence review before finishing."
        if (
            runtime.last_evidence_review is None
            or runtime.last_evidence_review.get("next_action") != "ready_to_finish"
        ):
            return False, "The most recent evidence review did not authorize finishing."
        candidate_id = str(arguments.get("selected_candidate_id", ""))
        candidate = runtime.candidates_by_id.get(candidate_id)
        if candidate is None:
            return False, f"Unknown selected_candidate_id: {candidate_id}"
        if candidate.cir is None:
            return False, "The selected candidate does not have a CIR result."
        if not str(arguments.get("rationale", "")).strip():
            return False, "A non-empty final selection rationale is required."
        if not str(arguments.get("stopping_reason", "")).strip():
            return False, "A non-empty stopping reason is required."

        comparison = runtime.evidence_comparison(candidate_id)
        if not comparison["selected_is_highest_observed_cir"] and not str(
            arguments.get("why_not_highest_cir", "")
        ).strip():
            return False, (
                "A different evaluated candidate has higher CIR. Provide an explicit "
                "why_not_highest_cir trade-off justification."
            )
        return True, "ok"

    def run(self, runtime: ToolRuntime, *, send_visuals_to_agent: bool = True) -> AgentDecision:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid4().hex[:8]
        )

        audit_directory = self.audit_root / run_id
        audit_directory.mkdir(parents=True, exist_ok=False)
        readable_timeline_path = audit_directory / "readable_timeline.jsonl"

        round_number = 0
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": self._initial_prompt(
                    runtime,
                    visual_context_available=send_visuals_to_agent,
                ),
            }
        ]
        if send_visuals_to_agent:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(runtime.image),
                    "detail": "low",
                }
            )

        input_items: list[Any] = [{"role": "user", "content": content}]
        final_arguments: dict[str, Any] | None = None
        consecutive_protocol_errors = 0

        while final_arguments is None:
            tools, method_by_function = self._dynamic_tools(runtime)
            round_number += 1

            request_payload = {
                "model": self.model,
                "instructions": self._round_instructions(
                    runtime,
                    visual_context_available=send_visuals_to_agent,
                ),
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "input": input_items,
                "store": False,
            }

            self._write_audit_json(
                audit_directory / f"round_{round_number:03d}_request.json",
                request_payload,
            )
            self._append_audit_event(
                readable_timeline_path,
                {
                    "round": round_number,
                    "event": "request",
                    "available_actions": [tool["name"] for tool in tools],
                    "pending_cir_candidate_id": runtime.pending_cir_candidate_id,
                    "pending_review_candidate_id": runtime.pending_review_candidate_id,
                    "finish_authorized": runtime.finish_authorized,
                },
            )

            response = self.client.responses.create(**request_payload)

            self._write_audit_json(
                audit_directory / f"round_{round_number:03d}_response.json",
                response,
            )
            input_items += list(response.output)
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                consecutive_protocol_errors += 1
                if consecutive_protocol_errors >= MAX_CONSECUTIVE_PROTOCOL_ERRORS:
                    raise RuntimeError("The agent repeatedly failed to issue a required function call.")
                input_items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Choose exactly one legal function tool now."}],
                    }
                )
                continue

            primary = calls[0]
            extras = calls[1:]
            action_made_progress = False
            selected_method_for_call = method_by_function.get(primary.name)
            legal_tool = primary.name in {tool["name"] for tool in tools}

            # Initialise this before parsing so it also exists when JSON parsing fails.
            arguments: dict[str, Any] = {}

            try:
                arguments = json.loads(primary.arguments or "{}")
            except json.JSONDecodeError as exc:
                result: dict[str, Any] = {
                    "status": "error",
                    "error": f"Invalid tool JSON: {exc}",
                }
            else:
                try:
                    if not legal_tool:
                        raise ValueError(f"Tool is not legal in this state: {primary.name}")
                    if primary.name == "select_explainer":
                        result = runtime.select_explainer(str(arguments["explainer"]), str(arguments["rationale"]))
                        action_made_progress = True

                    elif selected_method_for_call is not None:
                        result = runtime.run_method(
                            selected_method_for_call,
                            selection_rationale=str(
                                arguments["selection_rationale"]
                            ).strip(),
                            uncertainty_addressed=str(
                                arguments["uncertainty_addressed"]
                            ).strip(),
                            expected_signal=str(
                                arguments["expected_signal"]
                            ).strip(),
                            confidence=float(arguments["confidence"]),
                        )
                        action_made_progress = True

                    elif primary.name == "inspect_tool_source":
                        method = str(arguments["method"])
                        result = runtime.inspect_method_source(method)
                        result["inspection_reason"] = str(
                            arguments["inspection_reason"]
                        ).strip()
                        action_made_progress = True

                    elif primary.name == "calculate_cir":
                        result = runtime.calculate_candidate_cir(
                            str(arguments["candidate_id"])
                        )
                        action_made_progress = True

                    elif primary.name == "review_candidate_evidence":
                        result = runtime.review_candidate_evidence(
                            str(arguments["candidate_id"]),
                            evidence_summary=str(
                                arguments["evidence_summary"]
                            ),
                            cir_assessment=str(arguments["cir_assessment"]),
                            counterevidence_considered=str(
                                arguments["counterevidence_considered"]
                            ),
                            unresolved_uncertainty=str(
                                arguments["unresolved_uncertainty"]
                            ),
                            next_action=str(arguments["next_action"]),
                            action_reason=str(arguments["action_reason"]),
                        )
                        action_made_progress = True

                    elif primary.name == "finish_selection":
                        valid, message = self._validate_finish(
                            arguments,
                            runtime,
                        )

                        if valid:
                            final_arguments = dict(arguments)
                            result = {
                                "status": "accepted",
                                "message": "Final selection recorded.",
                            }
                            action_made_progress = True
                        else:
                            result = {
                                "status": "error",
                                "error": message,
                            }

                    else:
                        result = {
                            "status": "error",
                            "error": f"Unsupported tool: {primary.name}",
                        }

                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"

                    if selected_method_for_call is not None:
                        runtime.record_method_failure(
                            selected_method_for_call,
                            error_text,
                        )
                        action_made_progress = True

                    elif legal_tool and primary.name == "calculate_cir":
                        failed_candidate_id = str(
                            arguments.get(
                                "candidate_id",
                                runtime.pending_cir_candidate_id or "",
                            )
                        )
                        runtime.record_cir_failure(
                            failed_candidate_id,
                            error_text,
                        )
                        action_made_progress = True

                    result = {
                        "status": "error",
                        "tool": primary.name,
                        "error": error_text,
                        "method_marked_attempted": (
                            selected_method_for_call is not None
                        ),
                        "candidate_abandoned": (
                            legal_tool and primary.name == "calculate_cir"
                        ),
                    }

            # Add the executed tool's result to the next LLM request.
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": primary.call_id,
                    "output": json.dumps(result),
                }
            )

            # Save exactly which tool was requested, its arguments,
            # and the result returned by the local Python implementation.
            self._write_audit_json(
                audit_directory
                / f"round_{round_number:03d}_tool_execution.json",
                {
                    "selected_function": primary.name,
                    "function_call_id": primary.call_id,
                    "arguments": arguments,
                    "result": result,
                    "additional_calls_rejected": [
                        {
                            "name": extra.name,
                            "call_id": extra.call_id,
                            "reason": (
                                "Sequential policy permits only one tool call "
                                "per decision round."
                            ),
                        }
                        for extra in extras
                    ],
                },
            )
            self._append_audit_event(
                readable_timeline_path,
                {
                    "round": round_number,
                    "event": "tool_execution",
                    "selected_function": primary.name,
                    "arguments": arguments,
                    "status": result.get("status", "completed"),
                    "method": result.get("method")
                    or arguments.get("method"),
                    "candidate_id": result.get("candidate_id")
                    or arguments.get("candidate_id")
                    or arguments.get("selected_candidate_id"),
                    "cir": result.get("cir"),
                    "next_action": (
                        result.get("review", {}).get("next_action")
                        if isinstance(result.get("review"), dict)
                        else arguments.get("next_action")
                    ),
                    "error": result.get("error"),
                },
            )

            # Reject any extra calls if the model returned more than one.
            for extra in extras:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": extra.call_id,
                        "output": json.dumps(
                            {
                                "status": "error",
                                "error": (
                                    "Sequential policy: only one tool may be "
                                    "called per decision round."
                                ),
                            }
                        ),
                    }
                )
            if (
                send_visuals_to_agent
                and selected_method_for_call is not None
                and result.get("candidate_id")
            ):
                input_items.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Visual evidence for candidate {result['candidate_id']}. "
                                    "Image 1 shows segmentation boundaries. Image 2 "
                                    "shows the selected positive-attribution regions. Numeric "
                                    "measurements remain authoritative."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": runtime.candidate_segmentation_data_url(
                                    result["candidate_id"]
                                ),
                                "detail": "low",
                            },
                            {
                                "type": "input_image",
                                "image_url": runtime.candidate_overlay_data_url(result["candidate_id"]),
                                "detail": "low",
                            },
                        ],
                    }
                )

            if (
                send_visuals_to_agent
                and primary.name == "calculate_cir"
                and result.get("candidate_id")
                and result.get("cir") is not None
            ):
                input_items.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"CIR omission image for candidate "
                                    f"{result['candidate_id']}. The critical region "
                                    "used by CIR has been replaced with the configured "
                                    "omission colour."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": runtime.candidate_omitted_data_url(
                                    result["candidate_id"]
                                ),
                                "detail": "low",
                            },
                        ],
                    }
                )

            if action_made_progress:
                consecutive_protocol_errors = 0
            else:
                consecutive_protocol_errors += 1
                if consecutive_protocol_errors >= MAX_CONSECUTIVE_PROTOCOL_ERRORS:
                    raise RuntimeError(
                        "The agent repeatedly returned invalid actions or an invalid final decision."
                    )

        selected_id = str(final_arguments["selected_candidate_id"])
        selected = runtime.candidates_by_id[selected_id]
        evidence = runtime.evidence_comparison(selected_id)
        tools_executed = runtime.executed_methods()
        tools_failed = dict(runtime.failed_methods)
        tools_failed.update({method: f"CIR failure: {error}" for method, error in runtime.failed_cir_methods.items()})
        tools_not_executed = runtime.unattempted_methods()

        final_arguments.update(
            {
                "selected_method": selected.method,
                "selected_cir": float(selected.cir.cir),
                "selected_relative_cir": float(selected.cir.relative_cir),
                "tools_executed": tools_executed,
                "tools_failed": tools_failed,
                "tools_not_executed": tools_not_executed,
                "explanation_calls_used": runtime.explanation_calls_used,
                "total_registered_tools": runtime.total_registered_tools,
                "evidence_comparison": evidence,
                "inspected_source_methods": sorted(runtime.inspected_methods),
                "evidence_reviews": [
                    candidate.evidence_review
                    for candidate in runtime.candidates_by_id.values()
                    if candidate.evidence_review is not None
                ],
                "llm_audit_directory": str(audit_directory.resolve()),
            }
        )
        runtime.trace.append({"event": "final_selection", "result": final_arguments})
        runtime.write_outputs()

        self._write_audit_json(
            audit_directory / "final_decision.json",
            final_arguments,
        )

        self._write_audit_json(
            audit_directory / "complete_runtime_trace.json",
            runtime.trace,
        )
        self._append_audit_event(
            readable_timeline_path,
            {
                "round": round_number,
                "event": "final_decision",
                "selected_method": selected.method,
                "selected_candidate_id": selected_id,
                "stopping_reason": final_arguments.get("stopping_reason"),
                "explanation_calls_used": runtime.explanation_calls_used,
                "tools_not_executed": tools_not_executed,
            },
        )

        return AgentDecision(
            raw=final_arguments,
            final_candidate_id=selected_id,
            final_method=selected.method,
            trace=list(runtime.trace),
            evidence_comparison=evidence,
            tools_executed=tools_executed,
            tools_failed=tools_failed,
            tools_not_executed=tools_not_executed,
            explanation_calls_used=runtime.explanation_calls_used,
            total_registered_tools=runtime.total_registered_tools,
            audit_directory=str(audit_directory.resolve()),
        )
    
