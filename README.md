# Agentic XAI notebook

The self-contained research notebook compares **LIME**, **ViT GradCAM**, and
**ViT GradCAM++** for one locally fixed image-class target. It is currently
configured for a **live OpenAI agent run on one ImageNet goldfish image** with
the locally cached pretrained ViT. The same notebook can still run the
offline scripted demonstration by changing three flags in Run settings.

All classifier, explanation, agent, and reporting code needed to run the
notebook is in its labeled cells. The prior versions are preserved in `backups/`.

## Research meeting presentation

Open `presentation/agentic_xai_interactive_flow.html` directly in Chrome or
Edge for an offline, clickable flow diagram. `presentation/PROJECT_EXPLAINED.md`
traces the data from image to decision; `presentation/PRESENTATION_GUIDE.md`
provides two- and five-minute scripts and research-team questions. A static
overview is saved as `presentation/agentic_xai_high_level.svg`. The diagram
uses saved experiment images and makes no model or API calls.

## Files

- `agentic_lime_decisions.ipynb`: complete implementation, walkthrough,
  optional four-image reproduction procedure, and results table. Run this file
  alone in the configured environment.
- `agentic_xai.py`: small compatibility loader for existing scripts and tests.
  The notebook does not import or need this file.
- `tests/test_agentic_xai.py`: deterministic offline tests.
- `scripts/check_offline_notebook.py`: executes all code cells in source order
  without a Jupyter network socket.
- `scripts/verify_jupyter_kernel.py`: executes and saves notebook outputs in
  a real Jupyter kernel after checking both offline safety flags.
- `scripts/validate_cached_vit.py`: uses locally cached ViT weights to compare
  GradCAM and GradCAM++ on a goldfish image and a labeled two-goldfish composite.
- `scripts/validate_cached_vit_lime.py`: runs one bounded, 100-sample LIME
  candidate on the cached ViT and goldfish image.
- `scripts/compare_baseline_xai.py`: executes the preserved notebook's first
  offline LIME candidate and compares its target, mask, and metrics with the
  revised implementation on the same synthetic image.
- `DEVELOPMENT_LOG.md`: implementation and validation record.

## Environment

From this folder in Anaconda Prompt or a configured VS Code terminal:

```powershell
conda env create -f environment.yml
conda activate agentic-lime-notebook
python -m ipykernel install --user --name agentic-lime-notebook --display-name "Agentic XAI"
jupyter lab agentic_lime_decisions.ipynb
```

If the environment already exists, use `conda activate agentic-lime-notebook`
without recreating it. In VS Code, choose that environment as the notebook
kernel. Run the notebook from this project folder so relative paths resolve.

## Which folders are needed?

For a live run of one image, use the notebook, a valid `.env` API key, the
image named in `IMAGE_PATHS`, the configured Python environment, and locally
cached ViT weights. The notebook creates `outputs/live-goldfish/` as needed.
It does not need `scripts/`, `tests/`, `backups/`, or earlier `outputs/` folders
to complete the live workflow. If the saved four-image CSV is absent, the last
cell prints a message rather than failing.

Keep `outputs/presentation-experiments/` with
`presentation/four_image_experiment_gallery.html` if you want the existing
four-image visual gallery: the HTML links to images in that output folder.
The notebook can regenerate these experiment results with
`RECOMPUTE_FOUR_IMAGES=True`, using the four source images and manifest in
`imagenet_sample_100/`. The other output folders are prior runs or validation
records and can be archived after any evidence you need for reports is saved.
`scripts/add_experiment_notebook_section.py` is a historical notebook editor;
do not run it on the current self-contained notebook.

## Current one-image live run

The notebook Run settings cell is currently set to:

```python
OFFLINE_MODE = False
ALLOW_REAL_OPENAI = True
USE_CACHED_VIT = True
IMAGE_PATHS = ["imagenet_sample_100/n01443537_goldfish.JPEG"]
```

Place `OPENAI_API_KEY` in the project `.env` file, select the
`agentic-lime-notebook` kernel, and run cells from top to bottom. The first
live request occurs in section 7, after ViT has classified the image locally.
The workflow permits up to nine agent decision steps across at most three
candidate attempts. The OpenAI SDK may retry a failed HTTP request internally,
so this limit is not an exact network-request or billing limit. The decisions
and actual usage are unknown until the run completes.
Saved results appear in `outputs/live-goldfish/goldfish/run-XX/`. The final
four-image section is a separate offline study and does not use the API key.

## Offline alternative

To run the earlier synthetic demonstration without OpenAI requests, change
the same Run settings cell to `OFFLINE_MODE=True`,
`ALLOW_REAL_OPENAI=False`, and `USE_CACHED_VIT=False`. This demonstrates all
three methods on two synthetic images. The selection request includes the
original image and the review request includes the explanation and omitted
image, but the scripted controller makes the decisions.

To use the **locally cached pretrained ViT** with offline scripted decisions,
keep the two offline flags and set `USE_CACHED_VIT=True`.
`ViTClassifier` loads model and processor with
`local_files_only=True`; it will fail clearly if weights are not cached rather
than download them. In this mode the notebook's scripted agent compares the two
CAM methods on the goldfish image. LIME on ViT is supported by the notebook
but can be slow on CPU; to include it in the same notebook run, change the
offline action script to `demo_actions(CONFIG)` and allow at least three
candidates.

The optional `REQUESTED_CLASS_ID` integer overrides local top-1 selection. It
must be in the classifier's actual class mapping. The chosen class stays fixed
for all methods in that image's session.

## Offline validation commands

These notebook execution helpers require the three offline settings above;
they intentionally refuse the current live configuration. The unit tests and
other validation scripts do not make OpenAI requests.

```powershell
python -m unittest discover -s tests -v
python scripts\check_offline_notebook.py
python scripts\verify_jupyter_kernel.py
python scripts\compare_baseline_xai.py
python scripts\validate_cached_vit.py
python scripts\validate_cached_vit_lime.py
```

The two pretrained validation commands require locally cached
`google/vit-base-patch16-224` weights. It creates a two-goldfish **composite
test fixture** from the supplied goldfish image. This tests same-target ViT
execution and visual alignment; it is not a natural multi-instance benchmark.
None of these commands uses OpenAI. `OpenAITransport` refuses to initialize
while `offline=True`; tests use `ScriptedTransport` and `FailClosedTransport`.

Results are saved under `outputs/meeting-notebook/` and
`outputs/pretrained-validation/`; the historical LIME check is under
`outputs/baseline-comparison/`. Each image has a numbered run directory with
`input.png`, candidate explanation/heatmap/mask/omission images, `arrays.npz`,
`result.json`, `timeline.json`, and `request_manifest.json`. Batch files include
`batch_results.csv`, `batch_summary.json`, and `workflow_comparison.csv`.
The manifest records request sizes and image counts, not encoded image content
or credentials.

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
   or select any measured candidate, including an earlier one.
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
count is instrumented with a mock, not billed OpenAI usage.

## Four-image offline research experiments

The latest saved experiment uses four real photographs: goose, garden spider,
peacock, and frilled lizard. The cached pretrained ViT performs inference,
while LIME, GradCAM, and GradCAM++ run locally. A deterministic scripted policy
handles the separate agent attempts. It makes **zero OpenAI API calls**.

To reproduce the four real-image calculations **inside the notebook**, set
`RECOMPUTE_FOUR_IMAGES = True` in its last experiment section and run that cell,
then the following results cell. The default is `False` so Run All displays
the saved results without a lengthy CPU recomputation.

The existing command-line helpers remain available. Activate
`agentic-lime-notebook` and run:

```powershell
python scripts/run_four_image_experiments.py
python scripts/validate_four_image_experiments.py
```

The first command creates a new numbered run for each image under
`outputs/presentation-experiments/`, then updates `experiment_index.json`,
`candidate_results.csv`, `candidate_results.json`, and `method_summary.csv`.
LIME uses 200 perturbations, 50 requested SLIC segments, seed 42, and a black
perturbation baseline. All methods use the local top prediction as target and
black RGB omission for the impact score. The binary CAM masks cover 20% of
pixels; LIME can exceed 20% because it removes complete superpixels. The
records distinguish explanation generation time from total candidate
evaluation time.

The notebook has a final cell that displays the saved result table.
Open `presentation/four_image_experiment_gallery.html` for an offline visual
viewer with method switching and image enlargement. The updated PowerPoint is
`presentation/Agentic_Image_XAI_Notebook_Four_Image_Results.pptx`; the separate
interactive workflow diagram remains in
`presentation/agentic_xai_interactive_flow.html` and links to the new gallery.

The sample manifest provides class labels, but these are not independently
audited localization or ground truth annotations. See
`outputs/presentation-experiments/RESEARCH_SUMMARY.md` for findings and limits.

## Live API note

The notebook is ready for a user-run live experiment, but a real multimodal
request has **not** been sent or validated in this implementation session.
The transport reads `OPENAI_API_KEY` from `.env` first, then the process
environment, and does not print the value. A present key does not establish
that OpenAI will accept it or that the project can access `gpt-5-mini`.
