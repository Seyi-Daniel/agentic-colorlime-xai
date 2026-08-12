import json
from types import SimpleNamespace

import numpy as np

from agentic_colorlime.config import ExperimentConfig
from agentic_colorlime.openai_agent import OpenAIAdaptiveAgent
from agentic_colorlime.segmentations import SEGMENTATION_FUNCTIONS
from agentic_colorlime.tool_catalog import ToolCatalog


class FakeRuntime:
    def __init__(self):
        self.catalog = ToolCatalog()
        self.pending_cir_candidate_id = None
        self.pending_review_candidate_id = None
        self.finish_authorized = False
        self._attempted: set[str] = set()
        self._evaluated: list[object] = []
        self.candidates_by_id: dict[str, object] = {}
        self.inspected_methods: set[str] = set()

    def unattempted_methods(self):
        return sorted(set(SEGMENTATION_FUNCTIONS) - self._attempted)

    def uninspected_methods(self):
        return sorted(set(SEGMENTATION_FUNCTIONS) - self.inspected_methods)

    def evaluated_candidates(self):
        return list(self._evaluated)

    def attempted_methods(self):
        return set(self._attempted)


def test_config_has_no_agent_tool_count_limits():
    config = ExperimentConfig()
    assert not hasattr(config, "max_explanation_calls")
    assert not hasattr(config, "shortlist_size")
    assert not hasattr(config, "max_shortlist_expansions")


def test_initial_round_exposes_source_inspection_and_every_method():
    runtime = FakeRuntime()
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    tool_names = {tool["name"] for tool in tools}

    assert set(mapping.values()) == set(SEGMENTATION_FUNCTIONS)
    assert "inspect_tool_source" in tool_names
    assert "finish_selection" not in tool_names
    assert "request_more_tools" not in tool_names


def test_source_inspection_is_optional_and_each_source_is_offered_once():
    runtime = FakeRuntime()
    runtime.inspected_methods.add("slic")
    tools, _ = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    inspect_tool = next(tool for tool in tools if tool["name"] == "inspect_tool_source")
    enum = inspect_tool["parameters"]["properties"]["method"]["enum"]

    assert "slic" not in enum
    assert set(enum) == set(SEGMENTATION_FUNCTIONS) - {"slic"}


def test_pending_candidate_forces_cir_before_any_other_action():
    runtime = FakeRuntime()
    runtime.pending_cir_candidate_id = "candidate-123"
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)

    assert mapping == {}
    assert [tool["name"] for tool in tools] == ["calculate_cir"]


def test_completed_cir_forces_evidence_review_before_finish_or_next_method():
    runtime = FakeRuntime()
    runtime.pending_review_candidate_id = "candidate-123"
    runtime.candidates_by_id["candidate-123"] = SimpleNamespace(
        candidate_id="candidate-123",
        method="slic",
    )
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)

    assert mapping == {}
    assert [tool["name"] for tool in tools] == ["review_candidate_evidence"]


def test_finish_is_exposed_only_after_review_authorizes_it():
    runtime = FakeRuntime()
    runtime._attempted.add("slic")
    runtime._evaluated.append(object())

    tools, _ = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    assert "finish_selection" not in {tool["name"] for tool in tools}

    runtime.finish_authorized = True
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    tool_names = {tool["name"] for tool in tools}
    assert "finish_selection" in tool_names
    assert set(mapping.values()) == set(SEGMENTATION_FUNCTIONS) - {"slic"}


def _finish_runtime(*, authorized=True):
    runtime = SimpleNamespace()
    runtime.finish_authorized = authorized
    runtime.pending_cir_candidate_id = None
    runtime.pending_review_candidate_id = None
    runtime.last_evidence_review = {
        "next_action": "ready_to_finish"
    } if authorized else None
    runtime.candidates_by_id = {
        "candidate-a": SimpleNamespace(cir=SimpleNamespace(cir=0.10)),
        "candidate-b": SimpleNamespace(cir=SimpleNamespace(cir=0.20)),
    }
    runtime.evidence_comparison = lambda candidate_id: {
        "selected_is_highest_observed_cir": candidate_id == "candidate-b"
    }
    return runtime


def test_finish_requires_prior_authorization_rationale_and_stopping_reason():
    base = {
        "selected_candidate_id": "candidate-b",
        "rationale": "Candidate B has sufficient observed evidence.",
        "stopping_reason": "The last review found no material uncertainty.",
        "why_not_highest_cir": "",
    }

    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        base, _finish_runtime(authorized=True)
    )
    assert valid is True

    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        base, _finish_runtime(authorized=False)
    )
    assert valid is False

    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        dict(base, rationale=""), _finish_runtime()
    )
    assert valid is False

    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        dict(base, stopping_reason=""), _finish_runtime()
    )
    assert valid is False


def test_lower_cir_selection_is_allowed_with_explicit_tradeoff():
    arguments = {
        "selected_candidate_id": "candidate-a",
        "rationale": "Candidate A is preferred on the observed non-CIR evidence.",
        "stopping_reason": "The trade-off was reviewed and no uncertainty remains.",
        "why_not_highest_cir": "Candidate B had a higher CIR but weaker observed evidence elsewhere.",
    }

    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        arguments, _finish_runtime()
    )
    assert valid is True

    arguments["why_not_highest_cir"] = ""
    valid, _ = OpenAIAdaptiveAgent._validate_finish(
        arguments, _finish_runtime()
    )
    assert valid is False


class _FakeResponses:
    def __init__(self):
        self.round = 0

    def create(self, **payload):
        expected = [
            "run_slic_lime",
            "calculate_cir",
            "review_candidate_evidence",
            "finish_selection",
        ][self.round]
        assert expected in {tool["name"] for tool in payload["tools"]}

        arguments = [
            {
                "selection_rationale": "SLIC is a useful first test.",
                "uncertainty_addressed": "Whether one candidate is sufficient.",
                "expected_signal": "CIR and LIME evidence will test the choice.",
                "confidence": 0.7,
            },
            {"candidate_id": "slic-123"},
            {
                "candidate_id": "slic-123",
                "evidence_summary": "The observed evidence is sufficient.",
                "cir_assessment": "supports",
                "counterevidence_considered": "No contradictory result was observed.",
                "unresolved_uncertainty": "",
                "next_action": "ready_to_finish",
                "action_reason": "Another method is not needed for this run.",
            },
            {
                "selected_candidate_id": "slic-123",
                "decision_confidence": 0.8,
                "rationale": "SLIC has sufficient observed evidence.",
                "cir_assessment": "supports",
                "stopping_reason": "The review found no material uncertainty.",
                "why_not_highest_cir": "",
                "alternative_xai_suggestion": "none",
                "alternative_xai_reason": "",
            },
        ][self.round]
        self.round += 1
        call = SimpleNamespace(
            type="function_call",
            name=expected,
            arguments=json.dumps(arguments),
            call_id=f"call-{self.round}",
        )
        return SimpleNamespace(output=[call])


class _LoopRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.image_profile = SimpleNamespace(
            qualitative_summary=lambda: "4x4 test image",
            to_dict=lambda: {"height": 4, "width": 4},
        )
        self.target_class_label = "test"
        self.target_class_id = 0
        self.original_target_probability = 0.8
        self.failed_methods = {}
        self.failed_cir_methods = {}
        self.last_evidence_review = None
        self.trace = []

    @property
    def explanation_calls_used(self):
        return len(self._attempted)

    @property
    def total_registered_tools(self):
        return len(SEGMENTATION_FUNCTIONS)

    @property
    def available_methods(self):
        return set(SEGMENTATION_FUNCTIONS)

    def generated_methods(self):
        return list(self._attempted)

    def successfully_evaluated_methods(self):
        return [
            candidate.method
            for candidate in self.candidates_by_id.values()
            if candidate.cir is not None
        ]

    def all_public_summaries(self):
        return [
            {
                "candidate_id": candidate.candidate_id,
                "method": candidate.method,
                "cir": None if candidate.cir is None else candidate.cir.cir,
            }
            for candidate in self.candidates_by_id.values()
        ]

    def run_method(self, method, **arguments):
        assert method == "slic"
        self._attempted.add(method)
        candidate = SimpleNamespace(
            candidate_id="slic-123",
            method="slic",
            cir=None,
            evidence_review=None,
        )
        self.candidates_by_id[candidate.candidate_id] = candidate
        self.pending_cir_candidate_id = candidate.candidate_id
        self.finish_authorized = False
        return {"candidate_id": candidate.candidate_id, "method": method}

    def calculate_candidate_cir(self, candidate_id):
        candidate = self.candidates_by_id[candidate_id]
        candidate.cir = SimpleNamespace(cir=0.2, relative_cir=0.25)
        self.pending_cir_candidate_id = None
        self.pending_review_candidate_id = candidate_id
        self._evaluated = [candidate]
        return {"candidate_id": candidate_id, "method": "slic", "cir": 0.2}

    def review_candidate_evidence(self, candidate_id, **arguments):
        review = {"candidate_id": candidate_id, "method": "slic", **arguments}
        self.candidates_by_id[candidate_id].evidence_review = review
        self.last_evidence_review = review
        self.pending_review_candidate_id = None
        self.finish_authorized = arguments["next_action"] == "ready_to_finish"
        return {
            "status": "recorded",
            "finish_authorized": self.finish_authorized,
            "review": review,
        }

    def evidence_comparison(self, candidate_id):
        return {
            "selected_is_highest_observed_cir": True,
            "selected_cir": 0.2,
            "highest_observed_cir": 0.2,
            "highest_observed_method": "slic",
            "selected_cir_rank": 1,
            "evaluated_candidate_count": 1,
            "absolute_gap_from_highest": 0.0,
        }

    def executed_methods(self):
        return list(self._attempted)

    def write_outputs(self):
        return None


def test_full_loop_requires_method_then_cir_then_review_then_finish(tmp_path):
    runtime = _LoopRuntime()
    agent = object.__new__(OpenAIAdaptiveAgent)
    agent.client = SimpleNamespace(responses=_FakeResponses())
    agent.model = "test-model"
    agent.audit_root = tmp_path

    decision = agent.run(runtime, send_visuals_to_agent=False)

    assert decision.final_method == "slic"
    assert decision.explanation_calls_used == 1
    assert decision.raw["evidence_reviews"][0]["next_action"] == "ready_to_finish"
    assert (tmp_path / next(tmp_path.iterdir()).name / "readable_timeline.jsonl").exists()
