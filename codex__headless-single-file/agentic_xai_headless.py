#!/usr/bin/env python3
"""Agentic Color-LIME XAI: fully commented command-line and file-output implementation.

Includes all current prediction, agent, LIME/Lasso-LIME/Kernel SHAP, segmentation,
CIR, multi-image processing, visual evidence, and audit logic in this one file.
There is no browser, window, widget, application session, or graphical entry point.
Images are saved as PNG files, never opened in a display window. Third-party
libraries are called normally; their internal implementations are not vendored.
Historical benchmark and subdivision experiments remain separate.

INSTALL DEPENDENCIES (Python 3.11+):
    python -m pip install 'numpy>=1.26' 'pillow>=10.3' 'python-dotenv>=1' \
        'PyYAML>=6' 'scikit-image>=0.25' 'scikit-learn>=1.4' 'lime>=0.2.0.1' \
        'shap>=0.46,<0.51' 'openai>=1.75' 'torch>=2.2' \
        'transformers>=4.45' 'accelerate>=1'
The project package is not required. Copying this file and installing the above
libraries is sufficient. Set OPENAI_API_KEY in your environment or a local .env.
HF_TOKEN is optional for gated/private classifier models. Live execution uses
remote agent calls and may download classifier weights.

RUN ONE IMAGE OR MANY (the output layout is the same):
    python agentic_xai_headless.py --image first.jpg --output-root outputs
    python agentic_xai_headless.py --image first.jpg second.jpg --output-root outputs
    python agentic_xai_headless.py --image first.jpg --no-send-visuals-to-agent
    python agentic_xai_headless.py --help
Optional --config loads a YAML profile; otherwise the settings are embedded.
--send-visuals-to-agent controls whether the remote agent receives evidence images;
it does not open a local display. Both settings still save evidence PNG files.

OUTPUTS:
    outputs/run-<batch-id>/
        batch_summary.json             Machine-readable outcomes, including errors
        batch_report.md                Readable overview and links to image reports
        image-0001/run-<image-id>/
            result.json                Complete serializable per-image result
            explanation.md             Prediction, rationale, comparisons, and image links
            run_summary.json           Classifier, target, settings, and top predictions
            input_image.png            Original input, normalized to RGB
            candidate_results.json     Measurements for each generated candidate
            agent_tool_trace.json      Detailed local actions and evidence
            decision_timeline.json     Compact ordered sequence of decisions
            <candidate-method>/*.png   Segmentation, selected regions, mask, omission
            llm_audit/...              Exact controller request/response and decision records
A failed image is listed in the batch summary/report; later images continue.
A failed run can retain partial records without a completed result.json.
Progress goes to stderr; stdout contains the final batch JSON, so it can be piped
into another program. Exit status is 0 on full success and 1 if any image failed.

USE FROM PYTHON:
    import os
    import agentic_xai_headless as xai
    batch = xai.run_batch(
        images=[xai.ImageInput("first.jpg", "first.jpg")],
        model_id_or_path="google/vit-base-patch16-224",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    # batch.results[index] is an ExperimentResult for each successful input.
    # result.selected_candidate, result.candidates, and result.decision hold evidence.
    # Call run_experiment(image=rgb_array, ...) directly for a single-image result.

READING MAP — search for SECTION 01, SECTION 02, etc.
    01  Settings and optional configuration loading
    02  Command-line inputs, terminal progress, and readable file reports
    03  Outer loop: multiple images, one shared classifier
    04  One-image pipeline: prediction -> runtime -> agent -> saved result
    05  Classifier adapter and cheap image measurements
    06  Agent tool schemas and the remote decision loop
    07  Tool registry and local execution/state machine
    08  Five ways to group pixels (segmentation)
    09  LIME, sparse LIME, Kernel SHAP, and critical-region selection
    10  CIR: omit selected regions and measure the probability drop
    11  Image loading, visual evidence, and saved image files
    12  Actual program start: call the command-line entry point

EXECUTION MAP:
    SECTION 12 -> cli_main (02): validate arguments and credentials
      -> run_batch (03): load one classifier
        -> for each image: run_experiment (04)
          -> profile_image + predictor.predict_one (05)
          -> OpenAIAdaptiveAgent.run (06), repeated decision rounds:
               select_explainer (07) -> optional source inspection (07)
               -> choose segmentation (08) + run explainer (09)
               -> select positive critical regions (09) -> calculate CIR (10)
               -> review evidence (07)
                    | uncertainty remains: choose another candidate
                    | sufficient evidence: validate and finish (06)
          -> return ExperimentResult and save JSON + explanation.md (02/04)
        -> checkpoint batch outcomes and batch_report.md; process next image (03)
      -> print batch JSON; exit with success/failure status (02)

HOW TO READ PYTHON HERE:
- The initial pass defines functions/classes and builds registries; function
  bodies execute only when called. The footer starts the command-line workflow.
- A class groups state and operations; self refers to that object. Dataclasses
  hold named settings or result values. Callers can inspect returned result objects.
- if/elif/else chooses a branch; for/while repeats work; return gives a result
  to the caller; raise reports failure; finally also runs when an error occurs.
- An image array is H x W x 3; segmentation is H x W integer IDs; a critical
  mask is H x W booleans; classifier probabilities have shape N x classes.
- The classifier predicts, the explainer attributes, and the agent chooses and
  reviews. No classifier retraining or agent fine-tuning happens during a run.

Derived from annotated snapshot 5856d8c6b444ba2caf3c91b40181a45321e8ac94.
The explanation/state-machine algorithms are retained. The command line now uses
one consistent batch path for any number of images, with reports replacing
interactive result presentation. This file contains all its application code;
future modular changes must be synchronized explicitly.
"""
from __future__ import annotations

# Standard-library tools: files/JSON, timing, CLI parsing, dataclasses, encoding,
# introspection for source inspection, and a lock for SHAP's random-number use.
import argparse
import base64
import contextlib
import inspect
import io
import json
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable
from uuid import uuid4

# Third-party numerical/image helpers. PyTorch/Transformers load only when a
# classifier is used; LIME and SHAP load inside their respective explanation engines.
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel
from skimage.segmentation import (
    felzenszwalb, find_boundaries, mark_boundaries, quickshift, slic, watershed,
)
from skimage.util import img_as_float, regular_grid

if TYPE_CHECKING:
    import torch



# ================================================================================================
# SECTION 01 — SETTINGS: choose parameters before running anything
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/config.py

# SETTINGS: one immutable collection of numeric choices shared by all candidates.
# These are user settings; the language model chooses algorithms, not these numbers.
@dataclass(frozen=True)
class ExperimentConfig:
    """All non-secret numerical settings used by one explanation run.

    Tool-use policy is intentionally not configured here. The agent sees every
    registered explanation tool and decides how many of them to execute.
    """

    lime_num_samples: int = 500
    lime_batch_size: int = 32
    lime_lasso_alpha: float = 0.001
    shap_num_samples: int = 512
    shap_max_segments: int = 256
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

    # Reject invalid sample counts, area fractions, colors, and penalties before expensive work.
    def validate(self) -> None:
        positive_ints = {
            "lime_num_samples": self.lime_num_samples,
            "lime_batch_size": self.lime_batch_size,
            "shap_num_samples": self.shap_num_samples,
            "shap_max_segments": self.shap_max_segments,
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
        if self.shap_num_samples < 2:
            raise ValueError("shap_num_samples must be at least 2")
        if not 0 < self.lime_lasso_alpha < float("inf"):
            raise ValueError("lime_lasso_alpha must be finite and positive")

    # Convert settings to an ordinary dictionary for audit records and tool-source inspection.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ORIGINAL COMPONENT: src/agentic_colorlime/config_io.py

# INPUT: an optional YAML path. OUTPUT: validated experiment settings and application defaults.
# With no path, this single-file edition uses the embedded agent-demo defaults.
def load_profile(path: str | Path | None) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load a named YAML profile and validate its ExperimentConfig section."""
    if path is None:
        config = ExperimentConfig()
        config.validate()
        return config, {
            "name": "agent-demo (embedded)",
            "application": {
                "model_id": "google/vit-base-patch16-224",
                "openai_model": "gpt-5-mini",
                "device": "auto",
                "send_visuals_to_agent": True,
                "output_root": "outputs",
            },
            "experiment": config.to_dict(),
        }
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


# Replace only settings explicitly supplied by the caller; then validate the resulting copy.
def apply_overrides(config: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    """Return a validated config with only non-None command-line overrides applied."""
    updated = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    updated.validate()
    return updated



# ================================================================================================
# SECTION 02 — COMMAND LINE AND REPORTS: receive inputs, run work, save results
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/cli.py

# The single-file edition needs no adjacent configs directory. An explicit --config still works.
def default_profile_path() -> None:
    return None


# Define the command-line inputs and return the values the user supplied.
# --image accepts one path or many; numeric overrides default to None so profile values survive.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the code-grounded agentic Color-LIME selector."
    )
    parser.add_argument("--image", required=True, nargs="+", action="extend",
                        help="One or more local image paths; --image may be repeated")
    parser.add_argument("--config", type=Path, default=default_profile_path())
    parser.add_argument("--model", help="Override the profile's Hugging Face model")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--lime-samples", type=int, default=None)
    parser.add_argument("--lime-batch-size", type=int, default=None)
    parser.add_argument("--lime-lasso-alpha", type=float, default=None)
    parser.add_argument("--shap-samples", type=int, default=None)
    parser.add_argument("--shap-max-segments", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--critical-area", type=float, default=None)
    parser.add_argument("--colorlime-k", type=int, default=None)
    parser.add_argument(
        "--send-visuals-to-agent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether evidence images are sent to the agent.",
    )
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


# CLI ENTRY: load settings and credentials, then run all paths through SECTION 03.
# A one-image invocation is a batch of one, so saved output has a consistent layout.
def cli_main() -> None:
    load_dotenv()
    args = parse_args()
    config, profile = load_profile(args.config)
    application = profile.get("application", {})

    config = apply_overrides(
        config,
        lime_num_samples=args.lime_samples,
        lime_batch_size=args.lime_batch_size,
        lime_lasso_alpha=args.lime_lasso_alpha,
        shap_num_samples=args.shap_samples,
        shap_max_segments=args.shap_max_segments,
        inference_batch_size=args.inference_batch_size,
        critical_area_fraction=args.critical_area,
        colorlime_k=args.colorlime_k,
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    # Stop early rather than downloading a classifier when the remote controller cannot run.
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env or export it locally.")

    model_id = args.model or application.get("model_id", "google/vit-base-patch16-224")
    openai_model = (
        args.openai_model
        or os.getenv("OPENAI_MODEL")
        or application.get("openai_model", "gpt-5-mini")
    )
    device = args.device or application.get("device", "auto")
    output_root = args.output_root or application.get("output_root", "outputs")
    send_visuals = (
        args.send_visuals_to_agent
        if args.send_visuals_to_agent is not None
        else bool(application.get("send_visuals_to_agent", True))
    )


    # Progress and any library console messages go to stderr. stdout remains JSON.
    print(f"Processing {len(args.image)} image(s)...", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        batch = run_batch(
            images=[ImageInput(name=Path(path).name, source=path) for path in args.image],
            model_id_or_path=str(model_id), openai_api_key=api_key,
            openai_model=str(openai_model), hf_token=os.getenv("HF_TOKEN") or None,
            device=str(device), output_root=str(output_root),
            send_visuals_to_agent=send_visuals, config=config,
            on_progress=print_image_progress,
        )
    summary = batch.summary()
    print(f"Completed: {summary['succeeded']}; failed: {summary['failed']}.", file=sys.stderr)
    print(f"Readable report: {Path(batch.batch_dir) / 'batch_report.md'}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    # Partial success still reports a nonzero exit code for scripts that call us.
    if summary["failed"]:
        raise SystemExit(1)


# Terminal progress callback: called after each image outcome has been saved.
# This is an optional ordinary Python callback; it holds no application session state.
def print_image_progress(done: int, total: int, item: dict[str, Any]) -> None:
    if item["status"] == "completed":
        selected = item["selected_candidate"]
        detail = (f"{item['target_class_label']} | {selected['explainer']} / "
                  f"{selected['segmentation_method']} | CIR={selected['cir']:.4f}")
    else:
        detail = item.get("error", item["status"])
    print(f"[{done}/{total}] {item['name']}: {item['status']} — {detail}",
          file=sys.stderr, flush=True)


# Save reports through a temporary file; replace the old file only when writing succeeds.
def write_report_file(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# Use relative links so reports and images remain together when the output folder moves.
def report_link(label: str, target: str | Path, report_dir: Path) -> str:
    from urllib.parse import quote

    relative = Path(os.path.relpath(target, report_dir)).as_posix()
    return f"[{label}]({quote(relative, safe='/')})"


# PRESENTATION FOR ONE IMAGE: save the returned object as JSON and a readable explanation.
# PNG generation still happens in the runtime. Here we link those measured artifacts;
# we do not recompute an explanation, call the model again, or synthesize new findings.
def write_result_files(result: ExperimentResult) -> None:
    directory = Path(result.run_dir)
    write_report_file(directory / "result.json", json.dumps(asdict(result), indent=2))
    selected = result.selected_candidate
    decision = result.decision.raw
    lines = [
        "# Image explanation", "",
        f"Predicted class: **{result.target_class_label}**",
        f"Original target probability: {result.target_probability:.6f}",
        f"Selected explainer: **{selected['explainer']}**",
        f"Selected segmentation: **{selected['segmentation_method']}**", "",
        "## Findings and selection", "",
        str(decision.get("rationale", "No rationale recorded.")), "",
        "### Why the agent stopped", "",
        str(decision.get("stopping_reason", "No stopping reason recorded.")), "",
        f"CIR assessment: {decision.get('cir_assessment', 'not recorded')}", "",
    ]
    if decision.get("why_not_highest_cir"):
        lines += ["### CIR trade-off", "", str(decision["why_not_highest_cir"]), ""]
    lines += [
        "## Measured evidence", "",
        f"- CIR (absolute probability drop): {selected['cir']:.6f}",
        f"- Relative CIR: {selected['relative_cir']:.6f}",
        f"- Actual removed area fraction: {selected['critical_area_fraction']:.6f}",
        "- CIR is omission evidence, not proof that an explanation is correct.",
        "- Whole regions can exceed the requested area; masking choices affect results.", "",
    ]
    if selected.get("local_surrogate_score") is not None:
        lines += [f"- LIME surrogate score: {selected['local_surrogate_score']:.6f}", ""]
    elif selected['explainer'] == 'shap':
        residual = selected['explanation_diagnostics']['additivity_residual']
        lines += [f"- SHAP additivity residual: {residual:.6g} (an accounting check, not a quality score).", ""]
    lines += ["## Saved images", ""]
    for name, path in selected["artifacts"].items():
        # A link opens an already saved file only when a reader chooses to follow it.
        lines.append("- " + report_link(name.replace("_", " "), path, directory))
    lines += ["", "## Candidate comparisons", "",
              "| Explainer | Segmentation | CIR | Removed area | Seconds |",
              "|---|---|---:|---:|---:|"]
    for candidate in result.candidates:
        cir = f"{candidate['cir']:.6f}" if 'cir' in candidate else 'not calculated'
        lines.append(f"| {candidate['explainer']} | {candidate['segmentation_method']} | "
                     f"{cir} | {candidate['critical_area_fraction']:.6f} | {candidate['total_seconds']:.3f} |")
    lines += ["", "## Evidence reviews", ""]
    for review in decision.get("evidence_reviews", []):
        lines += [f"### {review['method']}", "",
                  str(review.get("evidence_summary", "")), "",
                  "Counterevidence: " + str(review.get("counterevidence_considered", "")), "",
                  "Remaining uncertainty: " + (str(review.get("unresolved_uncertainty", "")) or "None declared."), "",
                  "Next action: " + str(review.get("next_action", "")), "",
                  "Reason: " + str(review.get("action_reason", "")), ""]
    lines += ["## Execution record", "",
              "Attempted pairs: " + (", ".join(result.decision.tools_executed) or "none"), "",
              "Unattempted pairs: " + (", ".join(result.decision.tools_not_executed) or "none"), ""]
    for method, error in result.decision.tools_failed.items():
        lines.append(f"- Failed {method}: {error}")
    for filename in ("result.json", "run_summary.json", "candidate_results.json",
                     "agent_tool_trace.json", "decision_timeline.json"):
        lines.append("- " + report_link(filename, directory / filename, directory))
    lines.append("- " + report_link("Exact agent audit", result.decision.audit_directory, directory))
    suggestion = decision.get("alternative_xai_suggestion", "none")
    if suggestion != "none":
        lines += ["", "## Suggested follow-up", "", str(suggestion), "",
                  str(decision.get("alternative_xai_reason", ""))]
    write_report_file(directory / "explanation.md", "\n".join(lines) + "\n")


# PRESENTATION FOR THE BATCH: update a human-readable index alongside the JSON manifest.
# Pending/running/failed items stay visible; successful items link to their full reports.
def write_batch_report(batch: BatchResult) -> None:
    directory = Path(batch.batch_dir)
    summary = batch.summary()
    lines = ["# Image explanation batch", "",
             f"Completed: {summary['succeeded']} · Failed: {summary['failed']}", ""]
    for item in batch.items:
        lines += [f"## {item['index'] + 1}. {item['name']}", "", f"Status: {item['status']}", ""]
        if item['status'] == 'completed':
            selected = item['selected_candidate']
            lines += [f"Prediction: {item['target_class_label']} ({item['target_probability']:.6f})", "",
                      f"Method: {selected['explainer']} / {selected['segmentation_method']}", "",
                      report_link("Read explanation", Path(item['run_dir']) / 'explanation.md', directory), ""]
        elif item.get('error'):
            lines += ["Error: " + item['error'], ""]
        if item.get('output_dir'):
            lines += [report_link("Image output folder", item['output_dir'], directory), ""]
    lines += [report_link("Machine-readable batch summary", directory / 'batch_summary.json', directory), ""]
    write_report_file(directory / "batch_report.md", "\n".join(lines))


# ================================================================================================
# SECTION 03 — BATCH ORCHESTRATION: the outer image loop
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/batch.py

# An image name plus its source: a path, encoded bytes, or an already decoded RGB array.
@dataclass
class ImageInput:
    name: str
    source: str | Path | bytes | np.ndarray

    # Decode this one image only when its turn arrives; normalize file/byte inputs via SECTION 11.
    def load(self) -> np.ndarray:
        if isinstance(self.source, bytes):
            return load_image_from_bytes(self.source)
        if isinstance(self.source, np.ndarray):
            return self.source
        return load_image_from_path(self.source)


# Batch-level manifest plus successful per-image result objects, indexed by input position.
@dataclass
class BatchResult:
    batch_dir: str
    items: list[dict[str, Any]] = field(default_factory=list)
    results: dict[int, ExperimentResult] = field(default_factory=dict)

    # Create a serializable report of successes, failures, and individual image records.
    def summary(self) -> dict[str, Any]:
        return {
            "batch_dir": self.batch_dir,
            "succeeded": sum(item["status"] == "completed" for item in self.items),
            "failed": sum(item["status"] == "failed" for item in self.items),
            "items": self.items,
        }


# BATCH ORCHESTRATOR: load one classifier, then loop through independent image explanations.
# Each iteration calls run_experiment; one failed image is recorded and does not stop later images.
def run_batch(
    *, images: list[ImageInput], model_id_or_path: str, openai_api_key: str,
    openai_model: str = "gpt-5-mini", hf_token: str | None = None,
    device: str = "auto", output_root: str | Path = "outputs",
    send_visuals_to_agent: bool = True, config: ExperimentConfig | None = None,
    predictor: Any | None = None,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> BatchResult:
    # An empty batch is a caller error; no output folder or model is created.
    if not images:
        raise ValueError("Provide at least one image.")
    config = config or ExperimentConfig()
    config.validate()
    directory = _run_directory(output_root)
    batch = BatchResult(batch_dir=str(directory.resolve()))
    batch.items = [{"index": i, "name": item.name, "status": "pending"}
                   for i, item in enumerate(images)]

    # Write the manifest to a temporary file, then atomically replace the prior manifest.
    # A reader therefore sees a complete old or new JSON document, not a half-written update.
    def save() -> None:
        temporary = directory / "batch_summary.tmp"
        temporary.write_text(json.dumps(batch.summary(), indent=2), encoding="utf-8")
        temporary.replace(directory / "batch_summary.json")
        write_batch_report(batch)

    # Checkpoint the manifest so completed work remains visible if later work fails.
    save()
    # Reuse a supplied classifier object; otherwise create one for the whole batch.
    if predictor is None:
        try:

            predictor = HuggingFaceImageClassifier(
                model_id_or_path, device=device,
                inference_batch_size=config.inference_batch_size, hf_token=hf_token,
            )
        # FAILURE BRANCH: record what failed; do not manufacture an explanation.
        except Exception as exc:
            for item in batch.items:
                item.update(status="failed", error=f"Classifier initialization failed: {type(exc).__name__}: {exc}")
            # Checkpoint the manifest so completed work remains visible if later work fails.
            save()
            return batch

    # OUTER LOOP: process inputs sequentially. The index also makes duplicate filenames safe.
    for index, image_input in enumerate(images):
        item = batch.items[index]
        # Index-based folders avoid collisions and never treat user filenames as paths.
        image_dir = directory / f"image-{index + 1:04d}"
        image_dir.mkdir()
        item.update(status="running", output_dir=str(image_dir.resolve()))
        # Checkpoint the manifest so completed work remains visible if later work fails.
        save()
        try:
            # Hand this image to SECTION 04. The returned decision belongs only to this
            # iteration.
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
        # FAILURE BRANCH: record what failed; do not manufacture an explanation.
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        # Checkpoint the manifest so completed work remains visible if later work fails.
        save()
        # Notify the caller after saving this outcome, then proceed to the next image.
        if on_progress is not None:
            on_progress(index + 1, len(images), dict(item))
    return batch



# ================================================================================================
# SECTION 04 — ONE-IMAGE PIPELINE: connect prediction, runtime, and agent
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/pipeline.py

# The final result for ONE image: original prediction, agent decision, and candidate summaries.
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


# Make a timestamped, randomly suffixed output folder so separate runs do not overwrite each other.
def _run_directory(output_root: str | Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = Path(output_root) / f"run-{stamp}-{uuid.uuid4().hex[:6]}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


# ONE-IMAGE ORCHESTRATOR: profile -> predict -> create runtime -> run agent -> save result.
# The original winning class remains the target for every candidate and every CIR calculation.
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
    # Measure the input once; these cheap statistics become initial agent context.
    profile: ImageProfile = profile_image(image_array)

    # A direct single-image call loads a classifier here; a batch passes its existing one.
    if predictor is None:

        predictor = HuggingFaceImageClassifier(
            model_id_or_path,
            device=device,
            inference_batch_size=config.inference_batch_size,
            hf_token=hf_token,
        )

    target_class_id, target_class_label, target_probability, probabilities = predictor.predict_one(image_array)

    # Create fresh candidate lists, pending-state flags, and output records for this image.
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
    # Create a fresh controller; conversation history is not carried over from another image.
    agent = OpenAIAdaptiveAgent(
        api_key=openai_api_key,
        model=openai_model,
        audit_root=run_dir / "llm_audit",
    )
    try:
        # Enter the agent decision loop in SECTION 06; return only after a valid final
        # selection.
        decision = agent.run(runtime, send_visuals_to_agent=send_visuals_to_agent)
    # Save partial runtime evidence even if the remote call or controller fails.
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

    result = ExperimentResult(
        run_dir=str(run_dir.resolve()),
        target_class_id=target_class_id,
        target_class_label=target_class_label,
        target_probability=target_probability,
        image_profile=profile.to_dict(),
        decision=decision,
        selected_candidate=selected_candidate,
        candidates=candidates,
    )
    write_result_files(result)
    return result



# ================================================================================================
# SECTION 05 — PREDICTION AND PROFILE: what the image contains and cheap measurements
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/model_runner.py

# Resolve an explicit CPU/CUDA/MPS choice or choose the available accelerator automatically.
# An unavailable explicitly requested device raises an error instead of silently changing the request.
def choose_device(requested: str = "auto") -> torch.device:
    import torch
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CLASSIFIER ADAPTER: give all explainers the same predict_proba interface.
# The processor prepares model inputs; the model is frozen in evaluation mode and is never retrained.
class HuggingFaceImageClassifier:
    """Adapter that makes a Hugging Face image classifier usable by LIME."""

    # Load the processor, pretrained classification model, and class-name mapping once.
    # The selected device and inference microbatch size apply to every later prediction.
    def __init__(
        self,
        model_id_or_path: str,
        *,
        device: str = "auto",
        inference_batch_size: int = 16,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.model_id_or_path = model_id_or_path
        self.device = choose_device(device)
        self.inference_batch_size = max(1, int(inference_batch_size))

        shared: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if hf_token:
            shared["token"] = hf_token

        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path, **shared)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id_or_path,
            low_cpu_mem_usage=True,
            **shared,
        )
        self.model.eval().to(self.device)

        raw_mapping = getattr(self.model.config, "id2label", {}) or {}
        self.id2label = {
            int(key): str(value)
            for key, value in raw_mapping.items()
            if str(key).lstrip("-").isdigit()
        }

    # Turn a numeric class ID into a readable label, falling back to the number if unmapped.
    def label(self, class_id: int) -> str:
        return self.id2label.get(int(class_id), str(int(class_id)))

    # Use a supported lower-precision CUDA context; CPU and MPS use a no-op context here.
    def _autocast(self):
        import torch
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    # Predict ONE inference microbatch: normalize pixels, preprocess, move tensors, run the model.
    # Softmax converts logits into probabilities; returned NumPy rows correspond to input images.
    def _predict_chunk(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        cleaned = [np.clip(np.asarray(image), 0, 255).astype(np.uint8, copy=False) for image in images]
        encoded = self.processor(images=cleaned, return_tensors="pt")
        encoded = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        # Disable gradient recording; this is inference only.
        with torch.inference_mode(), self._autocast():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
        return probabilities.detach().cpu().numpy()

    # Accept one RGB image or many; repeatedly predict manageable chunks in their original order.
    # A CUDA out-of-memory error halves the microbatch and retries the same cursor position.
    def predict_proba(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        import torch
        if isinstance(images, np.ndarray) and images.ndim == 3:
            items: list[np.ndarray] = [images]
        else:
            items = list(images)
        if not items:
            labels = int(getattr(self.model.config, "num_labels", 0))
            return np.empty((0, labels), dtype=np.float32)

        batch_size = min(self.inference_batch_size, len(items))
        outputs: list[np.ndarray] = []
        cursor = 0
        # INFERENCE LOOP: cursor advances only after a chunk succeeds.
        while cursor < len(items):
            end = min(cursor + batch_size, len(items))
            try:
                outputs.append(self._predict_chunk(items[cursor:end]))
                cursor = end
            # Retry a smaller CUDA microbatch; do not silently drop the images that failed.
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                self.inference_batch_size = min(self.inference_batch_size, batch_size)
        return np.concatenate(outputs, axis=0)

    # Predict the original image, select its maximum-probability class, and retain all class probabilities.
    def predict_one(self, image: np.ndarray) -> tuple[int, str, float, np.ndarray]:
        probabilities = self.predict_proba(image)[0]
        class_id = int(np.argmax(probabilities))
        return class_id, self.label(class_id), float(probabilities[class_id]), probabilities


# ORIGINAL COMPONENT: src/agentic_colorlime/image_profile.py

# Cheap measurements of the input image that the agent can inspect before running an explainer.
@dataclass(frozen=True)
class ImageProfile:
    """Cheap image observations available before any expensive LIME call."""

    height: int
    width: int
    aspect_ratio: float
    unique_color_ratio: float
    colorfulness: float
    saturation_mean: float
    luminance_entropy: float
    edge_density: float
    texture_variance: float

    # Serialize the profile measurements for the prompt and saved run summary.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Translate fixed measurement thresholds into a short low/moderate/high description.
    # These words describe image statistics; they are not measured explanation quality.
    def qualitative_summary(self) -> str:
        edge = "high" if self.edge_density >= 0.20 else "moderate" if self.edge_density >= 0.10 else "low"
        color = "high" if self.colorfulness >= 0.25 else "moderate" if self.colorfulness >= 0.12 else "low"
        texture = "high" if self.texture_variance >= 0.012 else "moderate" if self.texture_variance >= 0.004 else "low"
        return (
            f"{self.width}x{self.height} RGB image; {edge} edge density; "
            f"{texture} texture variation; {color} colorfulness; "
            f"mean saturation {self.saturation_mean:.3f}; luminance entropy {self.luminance_entropy:.3f}."
        )


# Bin brightness values, normalize bin counts to probabilities, and calculate normalized entropy.
def _entropy(values: np.ndarray, bins: int = 64) -> float:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities /= total
    probabilities = probabilities[probabilities > 0]
    raw = -float(np.sum(probabilities * np.log2(probabilities)))
    return raw / np.log2(float(bins))


# Measure shape, color variation, saturation, edges, brightness entropy, and local texture.
# This profile guides initial choices but does not replace execution evidence.
def profile_image(image: np.ndarray) -> ImageProfile:
    rgb_uint8 = np.asarray(image, dtype=np.uint8)
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb_uint8.shape}")

    height, width, _ = rgb_uint8.shape
    rgb = img_as_float(rgb_uint8)
    gray = rgb2gray(rgb)
    hsv = rgb2hsv(rgb)
    gradient = sobel(gray)

    # Sobel values are normalized for float images. A fixed threshold makes the
    # proportion informative across images instead of forcing a constant quantile.
    edge_density = float(np.mean(gradient >= 0.08))

    # Hasler/Süsstrunk-inspired normalized colourfulness proxy.
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    # Cap unique-colour work on very large images by deterministic striding.
    flat = rgb_uint8.reshape(-1, 3)
    if len(flat) > 200_000:
        stride = max(1, len(flat) // 200_000)
        flat = flat[::stride]
    unique_color_ratio = float(len(np.unique(flat, axis=0)) / max(len(flat), 1))

    local_dx = np.diff(gray, axis=1)
    local_dy = np.diff(gray, axis=0)
    texture_variance = float(0.5 * (np.var(local_dx) + np.var(local_dy)))

    return ImageProfile(
        height=int(height),
        width=int(width),
        aspect_ratio=float(width / max(height, 1)),
        unique_color_ratio=unique_color_ratio,
        colorfulness=colorfulness,
        saturation_mean=float(np.mean(hsv[..., 1])),
        luminance_entropy=_entropy(gray),
        edge_density=edge_density,
        texture_variance=texture_variance,
    )



# ================================================================================================
# SECTION 06 — THE AGENT: tool schemas, legal actions, and the decision loop
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/openai_agent.py

try:
    from openai import OpenAI
except ImportError:  # Allows local unit tests that do not call the API.
    OpenAI = None  # type: ignore[assignment]




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


# SCHEMA: before running a candidate, require a rationale, uncertainty, expected signal, and confidence.
# This describes arguments the LLM must supply; it does not execute an explainer.
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


# SCHEMA: expose one specific explainer/segmentation pair as a callable function tool.
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


# SCHEMA: offer only explanation families that still have unattempted candidates.
# This is the first choice; segmentation tools appear in the following round.
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


# SCHEMA: let the agent request a registered implementation and explain why inspection would help.
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


# SCHEMA: restrict the next CIR call to the candidate currently waiting for evaluation.
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


# SCHEMA: force a review of evidence, counterevidence, and unresolved uncertainty after CIR.
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


# SCHEMA: collect the selected candidate, final rationale, stopping reason, and any CIR trade-off.
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


# Final agent decision plus the trace, comparison, failed attempts, and audit-directory location.
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


# REMOTE CONTROLLER: ask the language model for one action, execute it locally, and return the evidence.
# ToolRuntime (SECTION 07) enforces the legal state transitions even if the model asks for something else.
class OpenAIAdaptiveAgent:
    # Validate the API credential and create the client. The key is not included in saved request payloads.
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

    # Recursively convert SDK responses, containers, and paths into values that JSON can store.
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


    # Persist an audit snapshot through a temporary file so interruption does not leave partial JSON.
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

    # Append one compact event per line to the human-readable audit timeline.
    @staticmethod
    def _append_audit_event(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Introduce this image, its fixed target class, measurements, and registered tools.
    # When images are not sent, explicitly prohibit unsupported visual claims.
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

    # Rebuild the controller instructions with the CURRENT runtime state at every round.
    # The agent sees pending obligations, attempted pairs, failed tools, and observed candidate results.
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

    # LEGAL-ACTION GATE: pending CIR takes priority, then pending review, then family/segmentation choice.
    # Finishing becomes available only after a review authorizes it; unavailable tools are never offered.
    @staticmethod
    def _dynamic_tools(runtime: ToolRuntime) -> tuple[list[dict[str, Any]], dict[str, str]]:
        method_by_function: dict[str, str] = {}

        # PRIORITY 1: a generated explanation cannot bypass its omission test.
        if runtime.pending_cir_candidate_id is not None:
            return [cir_tool(runtime.pending_cir_candidate_id)], method_by_function

        # PRIORITY 2: numeric results must be reviewed before another candidate or a final
        # choice.
        if runtime.pending_review_candidate_id is not None:
            candidate = runtime.candidates_by_id[runtime.pending_review_candidate_id]
            return [
                evidence_review_tool(candidate.candidate_id, candidate.method)
            ], method_by_function

        tools: list[dict[str, Any]] = []
        # No family is committed: expose family selection, and finish only if already
        # authorized.
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

        # Only unattempted pairs from the committed family become executable tools this round.
        for method in eligible:
            card = runtime.catalog.get(method)
            tools.append(method_tool(card))
            method_by_function[card.function_name] = card.method

        if runtime.finish_authorized:
            tools.append(finish_tool())

        # No legal action remains: report the state instead of entering an endless empty-tool
        # loop.
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

    # Check the proposed final choice against actual runtime evidence.
    # It must be reviewed, have CIR, include reasons, and justify selecting a candidate below the highest observed CIR.
    @staticmethod
    def _validate_finish(arguments: dict[str, Any], runtime: ToolRuntime) -> tuple[bool, str]:
        if not runtime.finish_authorized:
            return False, (
                "Finishing has not been authorized by a completed evidence review."
            )
        # PRIORITY 1: a generated explanation cannot bypass its omission test.
        if runtime.pending_cir_candidate_id is not None:
            return False, "Calculate the pending CIR before finishing."
        # PRIORITY 2: numeric results must be reviewed before another candidate or a final
        # choice.
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

    # MAIN AGENT LOOP: prompt -> one tool request -> local execution -> evidence -> next prompt.
    # The loop exits only when final_arguments is accepted; repeated invalid actions raise an error.
    # The model can stop after sufficient evidence; it does not have to exhaust all 15 candidate pairs.
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

        # DECISION LOOP: this is where choosing, executing, reflecting, and trying again happen.
        while final_arguments is None:
            tools, method_by_function = self._dynamic_tools(runtime)
            round_number += 1

            # Send the current legal tools and accumulated conversation; require one function
            # call.
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

            # REMOTE BOUNDARY: the language model proposes an action; it does not run local
            # Python itself.
            response = self.client.responses.create(**request_payload)

            self._write_audit_json(
                audit_directory / f"round_{round_number:03d}_response.json",
                response,
            )
            input_items += list(response.output)
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            # Plain text alone is invalid here. Ask again, but abort after repeated protocol
            # errors.
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
                # Return to the top of the decision loop without executing a candidate in this
                # round.
                continue

            # Execute at most the first proposed call; additional calls receive rejection
            # feedback below.
            primary = calls[0]
            extras = calls[1:]
            action_made_progress = False
            selected_method_for_call = method_by_function.get(primary.name)
            legal_tool = primary.name in {tool["name"] for tool in tools}

            # Initialise this before parsing so it also exists when JSON parsing fails.
            arguments: dict[str, Any] = {}

            try:
                # Decode the proposed tool arguments as data. No generated Python is evaluated.
                arguments = json.loads(primary.arguments or "{}")
            except json.JSONDecodeError as exc:
                result: dict[str, Any] = {
                    "status": "error",
                    "error": f"Invalid tool JSON: {exc}",
                }
            else:
                try:
                    # Reject a tool that was not offered in this state, even if its name is
                    # otherwise known.
                    if not legal_tool:
                        raise ValueError(f"Tool is not legal in this state: {primary.name}")
                    # DISPATCH A: save the family choice; segmentation choice follows in the
                    # next round.
                    if primary.name == "select_explainer":
                        result = runtime.select_explainer(str(arguments["explainer"]), str(arguments["rationale"]))
                        action_made_progress = True

                    # DISPATCH B: run the chosen segmentation and explainer locally (SECTION
                    # 07).
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

                    # DISPATCH C: return actual function source; this performs no explanation or
                    # CIR test.
                    elif primary.name == "inspect_tool_source":
                        method = str(arguments["method"])
                        result = runtime.inspect_method_source(method)
                        result["inspection_reason"] = str(
                            arguments["inspection_reason"]
                        ).strip()
                        action_made_progress = True

                    # DISPATCH D: measure the impact of omitting the candidate's selected
                    # regions.
                    elif primary.name == "calculate_cir":
                        result = runtime.calculate_candidate_cir(
                            str(arguments["candidate_id"])
                        )
                        action_made_progress = True

                    # DISPATCH E: validate and record the agent's stop/continue assessment.
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

                    # DISPATCH F: validate the final candidate and reasons before allowing the
                    # loop to exit.
                    elif primary.name == "finish_selection":
                        valid, message = self._validate_finish(
                            arguments,
                            runtime,
                        )

                        # Set the sentinel checked by the while loop; this is the successful
                        # stopping path.
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

                # A tool error becomes feedback to the controller, with failures tracked by the
                # runtime.
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
            # Return one rejection per extra call so the conversation has a matching output for
            # each call ID.
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

            # Reset the protocol-error counter only when a valid action or tracked execution
            # failure advanced state.
            if action_made_progress:
                consecutive_protocol_errors = 0
            else:
                consecutive_protocol_errors += 1
                if consecutive_protocol_errors >= MAX_CONSECUTIVE_PROTOCOL_ERRORS:
                    raise RuntimeError(
                        "The agent repeatedly returned invalid actions or an invalid final decision."
                    )

        # AFTER THE LOOP: retrieve measured state for the accepted candidate and write the final
        # audit.
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



# ================================================================================================
# SECTION 07 — LOCAL EXECUTION: registered tools, candidates, and enforced transitions
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/tool_catalog.py

# A small record linking a candidate ID, exposed tool name, explainer family, and segmentation.
@dataclass(frozen=True)
class ToolCard:
    method: str
    function_name: str
    explainer: str = "lime"
    segmentation: str = ""

    # Resolve the segmentation name; original LIME IDs double as their segmentation names.
    @property
    def segmentation_method(self) -> str:
        return self.segmentation or self.method

    # Expose the registered identifiers as plain metadata in the initial agent prompt.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOL_CARDS: dict[str, ToolCard] = {
    "slic": ToolCard(
        method="slic",
        function_name="run_slic_lime",
    ),
    "quickshift": ToolCard(
        method="quickshift",
        function_name="run_quickshift_lime",
    ),
    "felzenszwalb": ToolCard(
        method="felzenszwalb",
        function_name="run_felzenszwalb_lime",
    ),
    "watershed": ToolCard(
        method="watershed",
        function_name="run_watershed_lime",
    ),
    "colorlime": ToolCard(
        method="colorlime",
        function_name="run_colorlime",
    ),
}

EXPLAINERS = {
    "lime": "Image LIME with its default weighted Ridge surrogate.",
    "lime_lasso": "Image LIME with a Lasso surrogate and configured positive alpha; a sparse LIME variant, not LEMON.",
    "shap": "Kernel SHAP over segment-presence features, using an all-hidden background and probability units. At most 10 nonzero features are fitted; inspect diagnostics and segment count.",
}

for _family in ("lime_lasso", "shap"):
    for _segmentation in tuple(TOOL_CARDS):
        if TOOL_CARDS[_segmentation].explainer != "lime":
            continue
        _method = f"{_family}_{_segmentation}"
        TOOL_CARDS[_method] = ToolCard(
            method=_method, function_name=f"run_{_method}",
            explainer=_family, segmentation=_segmentation,
        )


# Registry of available explainer/segmentation pairs. It provides identifiers, not measured rankings.
class ToolCatalog:
    """Identifier catalogue for every registered explanation tool.

    There is deliberately no shortlist or retrieval policy in this version.
    Every unattempted registered tool is exposed to the agent at each decision
    point. Tool behaviour is available through local source inspection rather
    than hand-written strengths, limitations, or recommendations.
    """

    # Copy the supplied registry, or use all 15 default pairs.
    def __init__(self, cards: dict[str, ToolCard] | None = None) -> None:
        self.cards = dict(cards or TOOL_CARDS)

    # Return every tool card in a stable order for reproducible prompts.
    def all_cards(self) -> list[ToolCard]:
        return [self.cards[name] for name in sorted(self.cards)]

    # Resolve one candidate ID to its registered tool metadata; unknown IDs raise KeyError.
    def get(self, method: str) -> ToolCard:
        return self.cards[method]

    # Return only requested cards, ordered consistently.
    def cards_for_methods(self, methods: list[str] | set[str]) -> list[ToolCard]:
        return [self.cards[name] for name in sorted(methods)]


# ORIGINAL COMPONENT: src/agentic_colorlime/runtime.py

# ONE ATTEMPT: segmentation, explanation result, selection rationale, artifacts, and optional CIR/review.
# The legacy field name lime holds the common ExplanationResult for LIME, Lasso-LIME, or SHAP.
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

    # Build the measured evidence for the agent and saved reports without serializing image arrays.
    # SHAP receives its own diagnostics; a LIME surrogate-fit score is not invented for SHAP.
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


# LOCAL STATE MACHINE for one image. This object owns candidates, pending work, failures, and files.
# It contains no remote model decision-making: the controller calls these methods as tools.
class ToolRuntime:
    """Local state while the remote LLM adaptively requests tools.

    No application-level minimum, maximum, shortlist, or expansion budget is
    imposed. Each registered explanation method can be attempted at most once
    in a run, and the agent decides when sufficient evidence exists to stop.
    """

    # Bind the image and fixed target to this run; verify registry coverage; start with empty state.
    # Save the original image so the evidence can later be inspected alongside its explanations.
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

    # Count generated candidates and explanation failures, not inspections, CIR calls, or reviews.
    @property
    def explanation_calls_used(self) -> int:
        return len(self.candidate_id_by_method) + len(self.failed_methods)

    # Report the number of candidate pairs in this run, normally three explainers times five segmentations.
    @property
    def total_registered_tools(self) -> int:
        return len(self.available_methods)

    # Expose the full set of registered pair IDs; availability here does not mean already evaluated.
    @property
    def available_methods(self) -> set[str]:
        return set(self.catalog.cards)

    # Derive the families that still contain at least one unattempted pair.
    def available_explainers(self) -> list[str]:
        return sorted({self.catalog.get(m).explainer for m in self.unattempted_methods()})

    # First decision: commit the next candidate to a family and record the reason.
    # Reject switching families while a selection, CIR, or review is already pending.
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

    # Combine generated and failed pair IDs so neither can be retried within the same image run.
    def attempted_methods(self) -> set[str]:
        return set(self.candidate_id_by_method) | set(self.failed_methods)

    # List pairs that produced an explanation; their CIR may still be pending or may have failed.
    def generated_methods(self) -> list[str]:
        return list(self.candidate_id_by_method.keys())

    # List only candidates that also have a successful CIR result.
    def successfully_evaluated_methods(self) -> list[str]:
        return [candidate.method for candidate in self.evaluated_candidates()]

    # Report unique generated and failed pair IDs for the final execution summary.
    def executed_methods(self) -> list[str]:
        ordered = [*self.candidate_id_by_method.keys(), *self.failed_methods.keys()]
        return list(dict.fromkeys(ordered))

    # Subtract attempted pair IDs from the registry to obtain remaining choices.
    def unattempted_methods(self) -> list[str]:
        return sorted(self.available_methods - self.attempted_methods())

    # Allow each source-inspection request once per pair to prevent repetitive inspection loops.
    def uninspected_methods(self) -> list[str]:
        return sorted(self.available_methods - self.inspected_methods)

    # Return the actual local segmentation and explainer function source plus active settings.
    # This gives prior information about implementation, not evidence about this image until the tool runs.
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

    # Mark an unsuccessful pair as attempted and reopen family selection.
    # If no pair remains, reopen review of an earlier evaluated candidate so it can still be selected.
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


    # Abandon the failed CIR obligation without inventing a score; reopen earlier evidence for review if available.
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

    # EXECUTE A CANDIDATE: validate the selected family, segment the image, run the chosen explainer.
    # Save visual artifacts, register the candidate, and make CIR the mandatory next step.
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
        # The selected family is enforced locally, not merely suggested in the prompt.
        if card.explainer != self.selected_explainer:
            raise RuntimeError("Select the candidate's explainer before its segmentation.")
        # A pair gets one attempt per image; failures also consume that attempt.
        if method in self.attempted_methods():
            raise RuntimeError(f"{method} has already been attempted; select a different method or finish.")

        started = time.perf_counter()
        # Look up the real segmentation function in SECTION 08 and obtain its integer label map.
        segmentation = SEGMENTATION_FUNCTIONS[card.segmentation_method](self.image, self.config)
        segmentation_seconds = time.perf_counter() - started
        # ALGORITHM BRANCH: SHAP has its own engine; the two LIME variants share one engine with
        # a variant flag.
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

        # Give this result a unique identity; all later CIR/review calls refer to this
        # candidate.
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
        # STATE TRANSITION: this candidate now blocks further choices until its CIR obligation
        # is handled.
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

    # Run the shared omission test on this candidate and save the omitted image.
    # On success, clear pending CIR and require an evidence review before any further choice.
    def calculate_candidate_cir(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id not in self.candidates_by_id:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        if self.pending_cir_candidate_id is not None and candidate_id != self.pending_cir_candidate_id:
            raise RuntimeError(
                f"CIR must be calculated for pending candidate {self.pending_cir_candidate_id}, not {candidate_id}."
            )

        candidate = self.candidates_by_id[candidate_id]
        # Calculate omission evidence once for a candidate; reuse an existing result if
        # requested again.
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
        # STATE TRANSITION: after CIR, the next legal step is evidence review.
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

    # Validate and record the agent's evidence review.
    # ready_to_finish requires no declared unresolved uncertainty; try_another_method requires a remaining pair.
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
        # Calculate omission evidence once for a candidate; reuse an existing result if
        # requested again.
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
        # The agent cannot claim both material uncertainty and readiness to finish.
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
        # This single flag controls whether the final-selection tool is exposed in the next
        # round.
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

    # Encode the positive-region overlay for the optional visual evidence sent to the agent.
    def candidate_overlay_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        return image_to_data_url(explanation_overlay(self.image, candidate.lime.critical_mask))

    # Encode segmentation boundaries so the agent can inspect the grouping it actually executed.
    def candidate_segmentation_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        return image_to_data_url(
            segmentation_preview(self.image, candidate.segmentation.labels)
        )

    # Encode the actual image used by CIR; reject the request if CIR has not completed.
    def candidate_omitted_data_url(self, candidate_id: str) -> str:
        candidate = self.candidates_by_id[candidate_id]
        # Calculate omission evidence once for a candidate; reuse an existing result if
        # requested again.
        if candidate.cir is None:
            raise RuntimeError("The candidate has no CIR omitted image.")
        return image_to_data_url(candidate.cir.omitted_image)

    # Filter generated candidates to those with measured CIR; failures are excluded from comparison.
    def evaluated_candidates(self) -> list[Candidate]:
        return [candidate for candidate in self.candidates_by_id.values() if candidate.cir is not None]

    # Serialize every generated candidate for prompts, display, and saved records.
    def all_public_summaries(self) -> list[dict[str, Any]]:
        return [candidate.public_summary() for candidate in self.candidates_by_id.values()]

    # Find the largest measured CIR among evaluated candidates.
    # This is a comparison reference, not an automatic final-selection rule.
    def best_observed_cir_candidate(self) -> Candidate:
        valid = self.evaluated_candidates()
        if not valid:
            raise RuntimeError("No candidate has a CIR result")
        return max(valid, key=lambda candidate: float(candidate.cir.cir))

    # Compute the selected candidate's CIR rank and gap from the best observed candidate.
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

    # Write detailed candidate results, the tool trace, and a compact step-by-step decision timeline.
    def write_outputs(self) -> None:
        (self.output_dir / "candidate_results.json").write_text(
            json.dumps(self.all_public_summaries(), indent=2), encoding="utf-8"
        )
        (self.output_dir / "agent_tool_trace.json").write_text(
            json.dumps(self.trace, indent=2), encoding="utf-8"
        )
        timeline = []
        # Convert each detailed event into one compact timeline row, preserving its execution
        # order.
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



# ================================================================================================
# SECTION 08 — SEGMENTATION: turn pixels into interpretable groups
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/segmentations.py

# A height-by-width label map plus the exact segmentation settings and method-specific metadata.
@dataclass(frozen=True)
class SegmentationResult:
    method: str
    labels: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any]


# Renumber region IDs as consecutive integers starting at zero; preserve which pixels belong together.
def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    # np.unique(..., return_inverse=True) is compatible across scikit-image
    # versions and gives contiguous feature IDs starting at zero.
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


# Group nearby pixels into compact spatial superpixels, balancing color similarity and position.
def segment_slic(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "n_segments": config.slic_n_segments,
        "compactness": config.slic_compactness,
        "sigma": config.slic_sigma,
    }
    labels = slic(
        image,
        n_segments=config.slic_n_segments,
        compactness=config.slic_compactness,
        sigma=config.slic_sigma,
        start_label=0,
        channel_axis=-1,
    )
    return SegmentationResult("slic", _normalize_labels(labels), params, {})


# Group pixels by a local density-based procedure; adapt the seed argument to the installed library API.
def segment_quickshift(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "kernel_size": config.quickshift_kernel_size,
        "max_dist": config.quickshift_max_dist,
        "ratio": config.quickshift_ratio,
    }
    kwargs = {
        "kernel_size": config.quickshift_kernel_size,
        "max_dist": config.quickshift_max_dist,
        "ratio": config.quickshift_ratio,
        "convert2lab": True,
        "channel_axis": -1,
    }
    # scikit-image renamed random_seed to rng. Support both APIs.
    import inspect

    if "rng" in inspect.signature(quickshift).parameters:
        kwargs["rng"] = config.random_seed
    else:
        kwargs["random_seed"] = config.random_seed
    labels = quickshift(image, **kwargs)
    return SegmentationResult("quickshift", _normalize_labels(labels), params, {})


# Use graph-based image segmentation; scale and minimum size control how regions are merged.
def segment_felzenszwalb(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    params = {
        "scale": config.felzenszwalb_scale,
        "sigma": config.felzenszwalb_sigma,
        "min_size": config.felzenszwalb_min_size,
    }
    labels = felzenszwalb(
        image,
        scale=config.felzenszwalb_scale,
        sigma=config.felzenszwalb_sigma,
        min_size=config.felzenszwalb_min_size,
        channel_axis=-1,
    )
    return SegmentationResult("felzenszwalb", _normalize_labels(labels), params, {})


# Place grid seeds, compute a grayscale edge surface, and grow regions over that surface.
# A center seed is used if the requested grid produces no coordinates.
def segment_watershed(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    gray = rgb2gray(img_as_float(image))
    gradient = sobel(gray)
    markers = np.zeros(gray.shape, dtype=np.int32)
    seed_slices = regular_grid(gray.shape, n_points=config.watershed_markers)
    seed_mask = np.zeros(gray.shape, dtype=bool)
    seed_mask[seed_slices] = True
    coordinates = np.argwhere(seed_mask)
    if len(coordinates) == 0:
        coordinates = np.array([[gray.shape[0] // 2, gray.shape[1] // 2]])
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1, dtype=np.int32)
    labels = watershed(
        gradient,
        markers=markers,
        compactness=config.watershed_compactness,
    )
    params = {
        "requested_markers": config.watershed_markers,
        "actual_markers": int(len(coordinates)),
        "compactness": config.watershed_compactness,
    }
    return SegmentationResult("watershed", _normalize_labels(labels), params, {})


# Group pixels by weighted RGB color clustering, even when matching colors occur far apart.
# Cluster unique colors weighted by pixel frequency, then map each pixel back to its color cluster.
def segment_colorlime(image: np.ndarray, config: ExperimentConfig) -> SegmentationResult:
    """Weighted K-means Color-LIME segmentation used by the benchmark."""
    rgb = np.asarray(image, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, received shape {rgb.shape}")

    flat_pixels = rgb.reshape(-1, 3)
    unique_colors, inverse_indices, color_counts = np.unique(
        flat_pixels,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    original_unique_colors = int(len(unique_colors))
    requested_k = int(config.colorlime_k)

    # No clustering is needed when every unique color can have its own group.
    if requested_k >= original_unique_colors:
        labels = inverse_indices.reshape(height, width).astype(np.int32)
        metadata = {
            "requested_k": requested_k,
            "actual_color_features": original_unique_colors,
            "original_unique_colors": original_unique_colors,
            "kmeans_inertia": 0.0,
            "kmeans_iterations": 0,
        }
        return SegmentationResult(
            "colorlime",
            _normalize_labels(labels),
            {"k": requested_k},
            metadata,
        )

    kmeans = KMeans(
        n_clusters=requested_k,
        init="k-means++",
        n_init=int(config.colorlime_n_init),
        max_iter=int(config.colorlime_max_iter),
        tol=float(config.colorlime_tol),
        random_state=int(config.random_seed),
        algorithm="lloyd",
        copy_x=False,
    )
    # Pixel frequencies weight the unique colors; repeated colors retain their influence without
    # duplicating data.
    kmeans.fit(
        unique_colors.astype(np.float32, copy=False),
        sample_weight=color_counts.astype(np.float64, copy=False),
    )

    # Preserve the benchmark's canonical feature definition: every unique RGB
    # value inherits its weighted K-means cluster and every pixel inherits the
    # cluster of its original RGB value.
    # Map cluster IDs back through unique colors to pixels; disconnected locations can share a
    # feature.
    unique_color_features = kmeans.labels_.astype(np.int32, copy=False)
    labels = unique_color_features[inverse_indices].reshape(height, width)

    metadata = {
        "requested_k": requested_k,
        "actual_color_features": int(len(np.unique(labels))),
        "original_unique_colors": original_unique_colors,
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_iterations": int(kmeans.n_iter_),
    }
    return SegmentationResult(
        "colorlime",
        _normalize_labels(labels),
        {"k": requested_k},
        metadata,
    )


SEGMENTATION_FUNCTIONS: dict[str, Callable[[np.ndarray, ExperimentConfig], SegmentationResult]] = {
    "slic": segment_slic,
    "quickshift": segment_quickshift,
    "felzenszwalb": segment_felzenszwalb,
    "watershed": segment_watershed,
    "colorlime": segment_colorlime,
}



# ================================================================================================
# SECTION 09 — EXPLANATION: attribute the target prediction and select critical regions
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/lime_engine.py

# Common explainer output: signed-weight source, positive features, selected mask, fit score, and diagnostics.
# For compatibility, lime_seconds stores explanation time even when the algorithm is SHAP.
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


# Read the target's surrogate-fit score across LIME versions that expose a dictionary or scalar.
def _score_for_label(explanation: Any, target_class_id: int) -> float:
    score = getattr(explanation, "score", float("nan"))
    if isinstance(score, dict):
        score = score.get(int(target_class_id), float("nan"))
    try:
        return float(np.asarray(score).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


# Keep only positive target-class coefficients, ordered from largest to smallest.
# Negative attributions can exist but do not enter the saved critical-region mask.
def ranked_positive_features(explanation: Any, target_class_id: int) -> list[tuple[int, float]]:
    local_exp = getattr(explanation, "local_exp", {}) or {}
    weights = local_exp.get(int(target_class_id), [])
    return sorted(
        [(int(feature_id), float(weight)) for feature_id, weight in weights if float(weight) > 0],
        key=lambda item: item[1],
        reverse=True,
    )


# Accumulate WHOLE positive regions until the requested pixel-area target is reached.
# The mask can overshoot the target; it can also be smaller or empty if too few positive regions exist.
def select_critical_mask(
    segments: np.ndarray,
    positive_features: list[tuple[int, float]],
    target_area_fraction: float,
) -> tuple[np.ndarray, list[tuple[int, float]], float]:
    mask = np.zeros(segments.shape, dtype=bool)
    target_pixels = max(1, int(math.ceil(float(target_area_fraction) * mask.size)))
    selected: list[tuple[int, float]] = []
    # AREA LOOP: visit positively weighted regions in descending importance.
    for feature_id, weight in positive_features:
        mask |= segments == int(feature_id)
        selected.append((int(feature_id), float(weight)))
        # Stop adding regions once their combined area meets the target; never split a region.
        if int(mask.sum()) >= target_pixels:
            break
    return mask, selected, float(mask.mean())


# Run image LIME on the supplied fixed label map, with either Ridge or the Lasso surrogate.
# The library samples hidden/visible region combinations, queries the classifier, and fits weighted regression.
# Return the ranked positives, area-selected mask, surrogate score, elapsed time, and all signed coefficients.
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
    # SURROGATE BRANCH: replace default Ridge with Lasso; sampling and segmentation remain the
    # same.
    if variant == "lime_lasso":
        from sklearn.linear_model import Lasso

        kwargs["model_regressor"] = Lasso(
            alpha=config.lime_lasso_alpha, max_iter=10000,
            random_state=config.random_seed,
        )

    started = time.perf_counter()
    # LIBRARY BOUNDARY: LIME perturbs the image, calls predict_proba, and fits the local
    # surrogate.
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


# ORIGINAL COMPONENT: src/agentic_colorlime/shap_engine.py

_SHAP_RANDOM_LOCK = threading.Lock()


# Run Kernel SHAP with segment-presence vectors: zero means hidden, one means original pixels.
# Explain the original target probability relative to the all-hidden image; no gradients or retraining are used.
def run_shap(*, image: np.ndarray, predictor: Any, target_class_id: int,
             segments: np.ndarray, config: ExperimentConfig) -> ExplanationResult:
    import shap

    image = np.asarray(image, dtype=np.uint8)
    segments = np.asarray(segments, dtype=np.int32)
    if segments.shape != image.shape[:2]:
        raise ValueError("Segmentation shape does not match image shape")
    labels, inverse = np.unique(segments, return_inverse=True)
    feature_count = len(labels)
    # Fail this candidate before allocating large SHAP coalition arrays; the agent can choose a
    # coarser grouping.
    if feature_count > config.shap_max_segments:
        raise ValueError(
            f"Kernel SHAP received {feature_count} segments; configured limit is "
            f"{config.shap_max_segments}. Choose a coarser segmentation or raise "
            "shap_max_segments explicitly."
        )
    feature_map = inverse.reshape(segments.shape)
    baseline = np.empty_like(image)
    # BASELINE BRANCH: use each region's mean RGB instead of a constant replacement color.
    if config.hide_color is None:
        for label in labels:
            region = segments == label
            baseline[region] = image[region].mean(axis=0)
    else:
        baseline[:] = config.hide_color

    # SHAP callback: turn each binary feature vector into a masked RGB image, then obtain target probabilities.
    # Materialize only one inference microbatch at a time so RGB perturbations do not all occupy memory together.
    def predict_masks(masks: np.ndarray) -> np.ndarray:
        outputs = []
        # Never materialize all perturbed RGB images at once.
        # PERTURBATION LOOP: decode only this chunk of feature vectors into RGB images.
        for start in range(0, len(masks), config.inference_batch_size):
            chunk = np.asarray(masks[start:start + config.inference_batch_size])
            visible = chunk[:, feature_map] > 0.5
            images = np.where(visible[..., None], image, baseline)
            outputs.append(np.asarray(predictor.predict_proba(images))[:, target_class_id])
        return np.concatenate(outputs)

    started = time.perf_counter()
    # KernelExplainer samples with NumPy's legacy global RNG. Serialize seeded
    # calls and restore state so concurrent callers stay reproducible.
    # Serialize SHAP's seeded global-RNG section; restore the previous random state afterwards.
    with _SHAP_RANDOM_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(config.random_seed)
            # Background = all segments hidden. Explained instance = all segments visible.
            explainer = shap.KernelExplainer(
                predict_masks, np.zeros((1, feature_count)), link="identity"
            )
            # Estimate segment contributions; the explicit feature-selection setting fits at
            # most ten nonzero features.
            weights = np.asarray(explainer.shap_values(
                np.ones((1, feature_count)), nsamples=config.shap_num_samples,
                l1_reg=f"num_features({min(10, feature_count)})", silent=True,
            )).reshape(-1)
        # Restore random state even if SHAP or classifier inference raises an error.
        finally:
            np.random.set_state(state)
    if len(weights) != feature_count or not np.isfinite(weights).all():
        raise ValueError("Kernel SHAP returned invalid segment attributions")
    # Additivity compares target probability with baseline + sum(weights); it is not a quality
    # score.
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    target = float(predict_masks(np.ones((1, feature_count)))[0])
    positive = sorted(
        [(int(label), float(weight)) for label, weight in zip(labels, weights) if weight > 0],
        key=lambda item: item[1], reverse=True,
    )
    mask, selected, area = select_critical_mask(segments, positive, config.critical_area_fraction)
    return ExplanationResult(
        explanation=None, segments=segments, positive_features=positive,
        critical_mask=mask, selected_features=selected, actual_area_fraction=area,
        surrogate_score=None, lime_seconds=time.perf_counter() - started,
        explainer="shap", diagnostics={
            "algorithm": "kernel_shap", "link": "identity",
            "baseline_target_probability": expected,
            "explained_target_probability": target,
            "additivity_residual": float(target - expected - weights.sum()),
            "requested_samples": config.shap_num_samples,
            "l1_reg": f"num_features({min(10, feature_count)})",
            "hide_color": config.hide_color,
            "feature_weights": [[int(label), float(w)] for label, w in zip(labels, weights)],
        },
    )



# ================================================================================================
# SECTION 10 — CIR EVALUATION: test the selected regions by omission
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/cir.py

# Measured omission evidence: target probabilities before/after, absolute and relative drop, and class change.
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


# Copy the original image and replace only the selected pixels with the configured omission RGB value.
def omit_region(
    image: np.ndarray,
    mask: np.ndarray,
    omission_rgb: tuple[int, int, int],
) -> np.ndarray:
    omitted = np.asarray(image, dtype=np.uint8).copy()
    omitted[np.asarray(mask, dtype=bool)] = np.asarray(omission_rgb, dtype=np.uint8)
    return omitted


# CIR = max(original target probability - omitted target probability, 0).
# Relative CIR divides that drop by the original probability; CIR per area divides by actual removed area.
# Keep the ORIGINAL target class fixed even if a different class wins after omission.
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
    # Run the SAME classifier on the modified image, without changing its weights.
    probabilities = predictor.predict_proba(omitted)[0]
    # Read the original target class's new probability, not the new winning class's probability.
    after = float(probabilities[int(target_class_id)])
    # Clip confidence increases to zero. This code's CIR is an absolute probability drop despite
    # its historical name.
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



# ================================================================================================
# SECTION 11 — IMAGE UTILITIES: load, visualize, encode, and save evidence
# ================================================================================================


# ORIGINAL COMPONENT: src/agentic_colorlime/image_io.py

# Honor the image orientation metadata, convert to RGB, and return an unsigned-byte H x W x 3 array.
def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.load()
    return np.asarray(image, dtype=np.uint8)


# Resolve and check a local file, open it safely, then normalize its orientation and color channels.
def load_image_from_path(path: str | Path) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    with Image.open(resolved) as image:
        return pil_to_rgb_array(image)


# Open encoded bytes or a file-like object and return the same RGB format as local-path loading.
def load_image_from_bytes(encoded: BinaryIO | bytes) -> np.ndarray:
    if isinstance(encoded, bytes):
        stream = io.BytesIO(encoded)
    else:
        stream = encoded
    with Image.open(stream) as image:
        return pil_to_rgb_array(image)


# Create a downscaled JPEG for the LLM only; the original classifier/explainer image is unchanged.
def image_to_data_url(image: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    """Create a compact JPEG data URL for an optional multimodal agent input."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ORIGINAL COMPONENT: src/agentic_colorlime/visualization.py

# Keep critical-region pixels vivid, desaturate other pixels, and draw a white region boundary.
# This is a visualization of the selected mask, not a new attribution computation.
def explanation_overlay(image: np.ndarray, critical_mask: np.ndarray) -> np.ndarray:
    image_float = np.asarray(image, dtype=np.float32) / 255.0
    mask = np.asarray(critical_mask, dtype=bool)
    overlay = image_float.copy()
    gray = np.mean(overlay, axis=2, keepdims=True)
    overlay[~mask] = 0.25 * overlay[~mask] + 0.75 * gray[~mask]
    boundaries = find_boundaries(mask, mode="thick")
    overlay[boundaries] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(np.rint(overlay * 255), 0, 255).astype(np.uint8)


# Draw the boundaries of the actual label map over the original image.
def segmentation_preview(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    preview = mark_boundaries(
        np.asarray(image, dtype=np.float32) / 255.0,
        np.asarray(segments, dtype=np.int32),
        mode="thick",
    )
    return np.clip(np.rint(preview * 255), 0, 255).astype(np.uint8)


# Show the selected pixels against white for inspection; this is distinct from the black CIR omission test.
def mask_on_white(image: np.ndarray, critical_mask: np.ndarray) -> np.ndarray:
    output = np.full_like(np.asarray(image, dtype=np.uint8), 255)
    mask = np.asarray(critical_mask, dtype=bool)
    output[mask] = np.asarray(image, dtype=np.uint8)[mask]
    return output


# Create parent folders, save an RGB image, and return its absolute output path.
def save_rgb(array: np.ndarray, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(destination)
    return str(destination.resolve())


# Save a boolean critical mask as a black/white grayscale image for later inspection.
def save_mask(mask: np.ndarray, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(destination)
    return str(destination.resolve())



# ================================================================================================
# SECTION 12 — ACTUAL PROGRAM START: all definitions above now exist
# ================================================================================================
# Importing this file defines the functions and result classes without starting
# model inference or remote requests. A direct Python launch calls cli_main.
if __name__ == "__main__":
    cli_main()  # SECTION 02 -> batch loop -> per-image agent -> saved reports.
