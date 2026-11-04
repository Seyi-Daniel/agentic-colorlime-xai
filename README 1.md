# Agentic XAI live notebook

`agentic_lime_decisions.ipynb` is a self-contained **live** one-image research
workflow. It uses cached pretrained ViT locally, sends bounded image requests
to the OpenAI agent for method selection and review, calculates LIME, GradCAM,
and GradCAM++ locally, and saves every result in a numbered image folder.
The default input is `imagenet_sample_100/n01616318_vulture.JPEG`.

## Run in VS Code

1. Open this project folder and select the `agentic-lime-notebook` Conda kernel.
   If an older editor tab is open, reopen the notebook from disk and restart
   the kernel so external edits are not overwritten.
2. Keep `OPENAI_API_KEY` in this folder's `.env`; never paste it into a cell.
3. In the Run settings cell, change only `IMAGE_PATHS` to a list containing one
   local image path when you want to test another image.
4. Select **Run All**. This notebook runs live by default and makes real API
   requests. Each Run All allows at most three candidates and six agent
   requests. SDK retries are disabled, output tokens and images are bounded,
   and the transport enforces a conservative US$1 estimated-cost guard for
   `gpt-5-mini`. Each new Run All is a new experiment with its own request cap.

The image is classified locally before the first API request. The target class
is ViT's top prediction unless a valid `REQUESTED_CLASS_ID` is set. Check the
target printed by the notebook; the filename alone does not set the class.
The model weights load from the local cache, and CPU execution can take time.

If the Conda environment does not exist yet, create it from `environment.yml`
and select it as the VS Code notebook kernel:

```powershell
conda env create -f environment.yml
conda activate agentic-lime-notebook
python -m ipykernel install --user --name agentic-lime-notebook --display-name "Agentic XAI"
```

## Saved results

For an input named `n01616318_vulture.JPEG`, the notebook saves to
`outputs/live-image-runs/vulture/run-XX/`. A different filename gets its own
image folder. Each numbered run contains `result.json`, `timeline.json`,
`explanation.md`, `request_manifest.json`, candidate images and arrays,
`batch_results.csv`, `batch_summary.json`, and `workflow_comparison.csv`.
The root `outputs/live-image-runs/` contains image folders only; the notebook
no longer writes duplicate batch files there. `result.json` and the table
include the **best observed** candidate even if the agent stops inconclusive;
that descriptive field does not claim the explanation was validated.

The notebook itself does not import `agentic_xai.py`. That file loads its tagged
core cells for historical command-line scripts and deterministic tests; the
legacy scripted helpers live there, outside the live notebook. The earlier
notebook is preserved in `backups/agentic_lime_decisions.pre-live-only-2026-11-03.ipynb`.

## Agent workflow

1. Run local classification once. Use top-1 as the target unless a valid manual
   override is supplied. Target selection requires zero agent calls.
2. Build a compact multimodal selection request with the original image,
   target, top-five predictions, remaining budget, and previous candidate
   summaries. One strict tool call supplies method and all parameters.
3. Validate the tool arguments and reject unsupported, malformed, or repeated
   configurations before local computation.
4. Run LIME, GradCAM, or GradCAM++ locally. Validate the mask/heatmap, save
   visuals, omit the selected region using the configured RGB baseline, and
   classify the omitted image locally.
5. Build a review request with the candidate explanation, omitted image,
   metrics, and compact prior results. The agent can retry, stop inconclusive,
   or select any measured candidate, including an earlier one. A candidate ID
   is required for acceptance; an ID supplied with retry or inconclusive is
   ignored because those decisions do not select a candidate.
6. Stop at the candidate or agent-call budget and save the decision rationale.

Selection requests include the original image when a fresh method choice is
needed. Review requests do **not** resend the original. Requests are stateless:
the agent receives compact structured summaries instead of an expanding full
conversation history. API images are bounded JPEG data URLs; the initial image
uses `detail="low"`, while explanation reviews use `detail="high"`. Full-size
PNG artifacts remain saved locally. These input forms are described in the
[OpenAI image input guide](https://developers.openai.com/api/docs/guides/images-vision).
Tool schemas use strict function calling with required fields and no additional
object properties, following the [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).

## Explanation methods and metrics

- LIME retains the five segmenters and configurable perturbation, kernel,
  feature-selection, and surrogate options from the earlier notebook.
- GradCAM uses class-score gradients of patch tokens at the **last ViT block's
  pre-attention LayerNorm**. The class token is excluded and the patch grid is
  derived from processed image dimensions and patch size. Hooking after the
  final attention block would leave final patch-token gradients uninformative
  for the class-token classifier. See the
  [ViT CAM adaptation note](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md).
- GradCAM++ uses the same activations and class target with the established
  gradient-square/cube approximation to the method's weighting formula. It
  does **not** compute exact second/third derivatives. This matches the
  [pytorch-grad-cam implementation](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py)
  of the [original GradCAM++ method](https://arxiv.org/abs/1710.11063).
  The ViT adaptation is experimental and requires further localization review.

The fixed omission metric uses the same original image, target class,
classifier, and configured black RGB baseline for all three methods:

- **Confidence drop** = `p(target | original) − p(target | omitted)`; it remains
  signed so confidence increases are visible.
- **CIR** = `max(confidence drop, 0)` for continuity with earlier results.
- **Relative CIR** = `CIR / p(target | original)`.
- **Removed area** = fraction of input pixels omitted.
- **DIR** = fraction of selected, completed images whose top-1 class changed
  after omission.
- **LIME local fit R²** is a weighted training fit on LIME perturbations; it is
  not held-out fidelity and does not apply to CAM methods.

LIME removes whole positive-weight segments until the area threshold is met;
CAMs remove the hottest pixel fraction. Actual removed area is reported because
the mask geometry and area can differ. Confidence drop alone is not proof of
semantic correctness or localization. The old workflow call count in the
comparison table is reconstructed from the preserved state machine; the new
mock count is distinct from the real OpenAI request count saved with each live
result.

## Historical research results

The separate four-image scripted study on goose, garden spider, peacock, and
frilled lizard remains under `outputs/presentation-experiments/` and in the
presentation gallery. It is **not** part of the live notebook's Run All.
Historical experiment scripts and their deterministic tests remain available
through `agentic_xai.py`; they do not represent a live LLM validation. The old
`check_offline_notebook.py` and `verify_jupyter_kernel.py` helpers target the
previous offline notebook layout and are not the way to run this live notebook.

A prior live goose run is saved at `outputs/live-image-runs/goose/run-01/`:
GradCAM++ was accepted after a 0.891777 target-probability drop with 20% of
pixels omitted. A live vulture run is saved at
`outputs/live-image-runs/vulture/run-01/`: LIME was accepted, but its mask
covered 51.3% of the image and should be presented with that limitation.
Goldfish runs under `outputs/live-goldfish/` stopped inconclusive because the
fixed-area masks produced very small confidence changes. Agent acceptance
and omission scores are exploratory evidence, not proof of semantic
localization.

For the earlier interactive workflow and four-image gallery, see
`presentation/agentic_xai_interactive_flow.html` and
`presentation/four_image_experiment_gallery.html`. Implementation history and
validation details are in `DEVELOPMENT_LOG.md`.
