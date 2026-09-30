# Agentic LIME notebook

Open **agentic_lime_decisions.ipynb** for a clean, editable notebook, or
**agentic_lime_demo_executed.ipynb** to read an already executed example with
cell outputs, tables, image panels, and final decisions.

## Install and open

From the repository root, with your preferred Python 3.11/3.12 environment active:

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name agentic-lime-notebook --display-name "Agentic LIME"
python -m jupyter lab notebooks/agentic_lime_decisions.ipynb
```

Select the **Agentic LIME** kernel. Start at the top and run cells in order.
The notebook contains its own implementation: it does not import the legacy
package or single-file application, and does not require `pip install -e .`.

## Follow one image step by step

Sections 1–16 define the components in commented, grouped cells. Sections 17–29
run the workflow:

1. Load inputs and one classifier.
2. Inspect the prediction and image profile.
3. Let the controller select and lock a target class (A).
4. Select a segmentation method and parameters (B–C).
5. Select all LIME options (D–L).
6. Generate the candidate, inspect images and feature counts.
7. Measure CIR and class change.
8. Review evidence, then either try a new configuration or validate a selection.
9. Inspect the report and timeline; process the remaining images; inspect DIR.

Each walkthrough call to `run_agent_step(session)` advances **one action**.
Re-running it advances again; it does not replay the earlier decision.
`run_agent_to_completion(session)` finishes the same session within its budgets.
Re-running the session-creation cell starts a fresh image run.

## Live agent and ViT

```bash
python -m pip install 'torch>=2.2' 'transformers>=4.45,<5'
```

Set `OPENAI_API_KEY` in your environment or a local `.env` in the notebook or
repository folder. Never commit it. In section 2 set:

```python
MODE = "live"
IMAGE_PATHS = ["../images/first.jpg", "../images/second.jpg"]
AGENT_MODEL = "gpt-5-mini"  # Or another compatible Responses tool-calling model.
```

Paths resolve from the notebook's working directory. Restart the kernel and run
from the top. Live mode downloads the selected Hugging Face classifier if it is
not cached and makes paid OpenAI API calls. `SEND_VISUALS=False` sends predictions,
measurements and settings; `True` also sends original and evidence images.

The notebook's default **demo** uses two generated images, a synthetic color-based
classifier, a scripted controller, and actual LIME computations. It is explicitly
labeled throughout and makes no API calls. Its scores are illustrative software
outputs, not evidence of research performance or autonomous LLM reasoning.

## Scope of agent control

All A–L categories are implemented: target class, segmentation method, segmentation
settings, sample count, hide baseline, distance metric, kernel width, kernel
function, feature cap, feature selection, surrogate model, and regularization.
The options are bounded/registered; the agent cannot submit Python code.

- Target: one class per image, selected once and locked. `REQUESTED_CLASS_ID`
  can constrain it. This does not generate five independent target explanations
  in one call as the stock `top_labels=5` convenience setting does.
- Surrogates: Ridge or Lasso inside the same image-LIME engine. No separate
  explainer-family selection is involved.
- Kernels: original exponential, plus registered Laplacian and inverse quadratic.
- Reattempts: the same segmentation may be retried with different settings;
  identical configurations are rejected. Each attempted execution consumes budget.
- Policies: fixed seed, classifier, removal color and target area; candidate,
  round, pixel, sample, segment, and forward-selection limits bound work.

## Evaluation and outputs

Per candidate: training weighted R², segment/fitted/nonzero/positive/selected
feature counts, actual removed area, target probability before/after, CIR,
relative CIR, CIR per area, and original top-1 decision change. CIR tracks the
chosen target even when it is not the original winning class. Decision change
always compares the winning classes before and after omission.

R² is training fidelity, not held-out validation. Different neighborhoods can make
scores difficult to compare directly. Whole regions may exceed the 20% target.
CIR does not prove semantic correctness; no metric alone automatically selects a
live result. Failed, inconclusive and budget-limited runs are explicit.

Each run writes `result.json`, `timeline.json`, `explanation.md`, and per-candidate
PNG/NPZ evidence under `outputs/lime-notebook/`. Live requests/responses are in
`audit/`. Batch DIR uses one selected candidate per completed image; the report
includes the denominator, coverage, errors and excluded session statuses.
The walkthrough plus the remaining batch are combined in
`combined_batch_summary.json`; the first image is not processed twice.

## Verification

```bash
python -m pytest tests/test_lime_notebook.py
python scripts/execute_lime_notebook.py --output /tmp/lime-demo.ipynb
```

The dedicated notebook CI installs no classifier runtime and runs the offline
workflow in a real Jupyter kernel. The older application's tests remain separate.
Only the explicitly synthetic example notebook contains committed outputs; clear
live outputs before sharing a notebook because they can contain your images and
model responses.
