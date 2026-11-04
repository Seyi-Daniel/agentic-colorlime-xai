import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from agentic_colorlime.config import ExperimentConfig

# The protocol tests do not run LIME. Provide a tiny import stub so they remain
# runnable in lightweight development environments where the optional runtime
# dependency has not yet been installed.
if "lime" not in sys.modules:
    lime_stub = ModuleType("lime")
    lime_stub.lime_image = ModuleType("lime.lime_image")
    sys.modules["lime"] = lime_stub
    sys.modules["lime.lime_image"] = lime_stub.lime_image

from agentic_colorlime.runtime import ToolRuntime


def test_source_inspection_returns_registered_code_and_active_config():
    runtime = SimpleNamespace(
        inspected_methods=set(),
        trace=[],
        config=ExperimentConfig(slic_n_segments=77),
    )

    result = ToolRuntime.inspect_method_source(runtime, "slic")

    assert "def segment_slic" in result["source_code"]
    assert result["active_experiment_config"]["slic_n_segments"] == 77
    assert runtime.inspected_methods == {"slic"}
    assert runtime.trace[0]["event"] == "source_inspection"


def _review_runtime(unattempted=("quickshift",)):
    candidate = SimpleNamespace(
        candidate_id="slic-123",
        method="slic",
        cir=SimpleNamespace(cir=0.2),
        evidence_review=None,
    )
    runtime = SimpleNamespace(
        candidates_by_id={"slic-123": candidate},
        pending_review_candidate_id="slic-123",
        finish_authorized=False,
        last_evidence_review=None,
        trace=[],
        unattempted_methods=lambda: list(unattempted),
    )
    return runtime


def test_review_can_authorize_finish_only_without_unresolved_uncertainty():
    runtime = _review_runtime()
    result = ToolRuntime.review_candidate_evidence(
        runtime,
        "slic-123",
        evidence_summary="The candidate has completed LIME and CIR evidence.",
        cir_assessment="supports",
        counterevidence_considered="No contradictory measured signal was observed.",
        unresolved_uncertainty="",
        next_action="ready_to_finish",
        action_reason="The available evidence is sufficient for this run.",
    )

    assert result["finish_authorized"] is True
    assert runtime.pending_review_candidate_id is None
    assert runtime.trace[0]["event"] == "evidence_review"


def test_review_cannot_finish_while_declaring_material_uncertainty():
    runtime = _review_runtime()

    with pytest.raises(ValueError, match="empty unresolved_uncertainty"):
        ToolRuntime.review_candidate_evidence(
            runtime,
            "slic-123",
            evidence_summary="Evidence exists.",
            cir_assessment="inconclusive",
            counterevidence_considered="The CIR result is inconclusive.",
            unresolved_uncertainty="A second segmentation may clarify the result.",
            next_action="ready_to_finish",
            action_reason="Stop anyway.",
        )


def test_review_cannot_request_another_method_when_none_remain():
    runtime = _review_runtime(unattempted=())

    with pytest.raises(RuntimeError, match="No unattempted"):
        ToolRuntime.review_candidate_evidence(
            runtime,
            "slic-123",
            evidence_summary="Evidence exists.",
            cir_assessment="contradicts",
            counterevidence_considered="CIR contradicts the initial expectation.",
            unresolved_uncertainty="Another method should be compared.",
            next_action="try_another_method",
            action_reason="A comparison is needed.",
        )
