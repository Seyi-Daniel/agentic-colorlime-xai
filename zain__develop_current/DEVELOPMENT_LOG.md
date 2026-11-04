# Development log — 2026-10-20 meeting implementation

## Scope and preservation

The project remains a Jupyter notebook workflow. No GUI or teaching-example
dataset was added. The previous notebook and README were copied to `backups/`
before modification because this folder is not a Git repository.

The new `agentic_lime_decisions.ipynb` began as a 16-cell research walkthrough backed
by `agentic_xai.py`. The module preserves the earlier five LIME segmenters and
configurable kernel, perturbation, feature selection, and surrogate choices.
It adds shared evaluation, ViT GradCAM++, structured agent decisions, request
builders, offline transports, failure handling, and concise reporting.

## Architecture and meeting requirements

1. `RunSession` classifies the original RGB image locally once and fixes the
   target to top-1 or a validated manual class override. No agent call chooses
   the target.
2. `build_agent_request` constructs a stateless Responses API request. A
   selection request contains the original image, target, top-five predictions,
   model and evaluation context, and compact previous attempts. A review
   request contains the current explanation, omitted image, optional heatmap,
   locally computed metrics, and prior candidate summaries. It omits the
   original image. API images are bounded JPEG data URLs. Request construction
   is tested without network access.
3. One strict tool call chooses a method and its complete parameters. Five
   LIME tools cover the segmenters; two CAM tools cover GradCAM and GradCAM++.
   Local JSON-schema and cross-field validation reject malformed, unsupported,
   invalid, or repeated configurations before explanation work.
4. Local computation generates a candidate, mask, omitted image, probabilities,
   metrics, and saved visuals. The next agent call reviews the evidence. It may
   retry, stop inconclusive, or select any measured candidate, including an
   earlier one. Candidate and call budgets bound the loop.
5. Results preserve signed confidence drop, CIR, actual removed area, decision
   change, local LIME fit R², warnings/failures, execution time, decisions,
   reasons, and simulated call counts. `batch_results.csv` and notebook tables
   show concise per-candidate records.
6. `RunConfig.offline=True` is the default. `OpenAITransport` refuses to
   initialize unless both `offline=False` and `allow_real_api=True` are set.
   `AgentController` also refuses a network-capable transport for any offline
   session before it constructs or sends a request.
   Offline scripts use `ScriptedTransport`; `FailClosedTransport` is available
   for tests. No real OpenAI API call was made.

## GradCAM++ method

`ViTClassifier.cam_heatmap` hooks the last ViT encoder block's
`layernorm_before`, before its final attention operation. This is intentional:
the classifier head uses the class token, and patch tokens after the final
attention cannot influence that class token. The implementation drops the
class token, checks the processed image dimensions against `patch_size`,
reshapes patches to the actual spatial grid, and resizes the CAM to the
original image. Both CAM methods use the same activations, target class, and
target layer.

GradCAM uses spatially averaged class-score gradients as channel weights.
GradCAM++ uses the established first-gradient approximation to its higher
derivative weighting equation: squared/cubed gradients, spatially summed
activations, positive gradients, and guarded division. It does **not** compute
exact second or third derivatives. This choice follows the
[pytorch-grad-cam GradCAM++ source](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py),
adapted to ViT patch tokens following its
[ViT guidance](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md).
The [original GradCAM++ paper](https://arxiv.org/abs/1710.11063) established
the method for CNN features; its claimed multi-instance advantage is not
assumed to transfer to this ViT.

## Tests actually executed

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -v` | 14 offline tests passed, including a central offline transport guard and SDK request serialization through an `httpx.MockTransport` with an intentionally nonresolving URL. |
| `scripts/check_offline_notebook.py` | All notebook code cells executed in source order in a fresh Python process. |
| `scripts/verify_jupyter_kernel.py` | All 16 notebook cells executed in an actual Jupyter kernel; outputs were saved in the notebook. The sandboxed kernel launcher initially stalled, so this verification ran outside that sandbox. Both offline safety flags were checked before launch. |
| `scripts/compare_baseline_xai.py` | The old and new first LIME candidates used target 0 and produced **identical masks**, area `0.2021484375`, CIR `0.2285995327`, and training R² `0.9986913918` on the same synthetic image. The old workflow used 6 controller rounds through impact; the new mock used 1 selection call with local execution and impact. |
| `scripts/validate_cached_vit.py` | Cached pretrained `google/vit-base-patch16-224` loaded locally. GradCAM and GradCAM++ produced finite, nonconstant, correctly sized heatmaps on the original goldfish and a two-goldfish composite, always using goldfish class ID 1. Four scripted calls per image; no OpenAI calls. |
| `scripts/validate_cached_vit_lime.py` | Cached ViT completed one 100-sample LIME candidate on goldfish: target class 1, area `0.265219`, signed confidence drop/CIR `0.285882`, and two scripted calls. |

For the pretrained CAM comparison, mean absolute heatmap differences were
`0.1620` on the original goldfish and `0.1704` on the two-goldfish composite.
The original-image GradCAM heatmap also changed when the target changed to
class ID 2 (mean absolute difference about `0.2709`). Repeated GradCAM++
inference on the same input/target was checked for numerical determinism.

On the composite, fixed-area black omission gave CIR about `0.9660` for
GradCAM and `0.2242` for GradCAM++; on the original image it gave about
`0.0018` and `0.0001`, respectively. The scripted final choice selected the
earlier GradCAM candidate after both were measured. These numbers describe
this omission test only; they do not establish that GradCAM localizes better.

## Workflow comparison and context

For two CAM candidates, the preserved old state machine implies about 11
agent calls (one target choice, two four-action CAM cycles, and two decisions)
if all steps succeed. The revised local test observed 4 mocked calls: one
combined selection and one review per candidate. For one LIME candidate,
the fair same-image test observed 6 old rounds through impact versus 1 revised
selection call through local impact.

The old goldfish `run-05` audit contains 15 requests ranging from 61,888 to
965,418 serialized bytes as full history accumulated. A revised goldfish CAM
comparison request manifest recorded four JPEG multimodal payloads of roughly
46,000–84,000 serialized characters, with text state of roughly 1,000–2,300
characters. These are **illustrative different runs**, not a controlled token
or billing comparison. Visual inputs add image tokens; only a real API run can
measure billed usage. The new request does not send all 1,000 class labels or
replay the entire conversation.

## Bugs and design problems fixed

- Agent-driven top-class selection made an unnecessary API round and allowed a
  mismatch with the original classifier prediction. Target selection is local.
- The earlier controller repeatedly sent a large label map and growing history.
  Requests now send top-five predictions and compact candidate summaries.
- The default old agent did not receive image pixels. New selection and review
  requests include the appropriate images, without resending the original in
  review rounds.
- Separate method, segmentation, LIME settings, execution, and measurement
  calls caused redundant agent rounds. Selection is combined; execution and
  metrics are local.
- Zero-area LIME masks and constant/non-finite CAMs could be treated as useful
  results. They are now local failures with explicit reasons.
- The first revised review path could only accept its latest candidate. It now
  accepts an earlier measured candidate, allowing a direct CAM comparison.
- An offline session could have been paired with a separately constructed live
  transport. The controller now refuses any network-capable transport before
  constructing or sending a request; a spy test verifies zero sends.
- A sandboxed Jupyter-kernel launcher stalled on local socket setup. Running
  the offline-guarded kernel verification outside that sandbox completed.

## Remaining research and integration limits

- Real OpenAI multimodal responses, schema acceptance by the server, latency,
  and billing were **not** tested, per the explicit no-API requirement.
- Scripted decisions prove application control flow, not real agent judgment.
- The two-goldfish image is a composite fixture, not a natural multi-instance
  benchmark. No instance masks or human localization judgments were available.
- GradCAM++ is a first-gradient approximation applied to ViT patch features.
  A broader benchmark should compare it with ordinary GradCAM, including
  natural multi-instance images and localization annotations.
- LIME and CAM masks use different geometry and may cover different actual
  areas. CIR alone cannot rank semantic explanation quality. LIME R² is a
  training-fit measure, not held-out faithfulness.
- CPU tests covered one pretrained ViT and a small number of images. Other
  architectures, devices, seeds, and larger batches remain unverified.

## Acceptance status

All software requirements that can be checked offline passed: local top-1 and
override validation, zero-call target choice, multimodal request construction,
one-call method/parameter selection, all three local methods, shared omission
evaluation, invalid-result handling, retry/switch/stop behavior, concise batch
table, old/new LIME comparison, mocked call accounting, and clean notebook
execution. Natural-image localization quality and real OpenAI integration
remain unverified by design.

## Four additional image experiments and presentation update — 2026-10-20

The notebook now has 18 cells, including a final offline section that reads
the four-image results. `scripts/run_four_image_experiments.py` executes the
same module used by the notebook against goose, garden spider, peacock, and
frilled lizard images from `imagenet_sample_100`. The source manifest SHA256
was checked for each. The locally cached pretrained
`google/vit-base-patch16-224` performed all predictions. The sample manifest
labels were recorded as source labels, not independently audited ground truth.

The fixed comparison ran LIME, GradCAM, and GradCAM++ on every image (12 valid
results). A separate offline scripted policy made eight actual explanation
attempts: GradCAM was accepted first for goose and spider; peacock exhausted
three attempts and remained inconclusive; frilled lizard accepted LIME on its
third attempt. The policy used a preset threshold of signed drop at least 20%
of baseline target probability and coverage at most 25%. It made **zero real
OpenAI calls**. LIME used 200 samples, SLIC with 50 requested regions, seed 42,
ridge surrogate, and black perturbation; all methods used black RGB omission.
Candidate generation now separately records explanation time and total
candidate evaluation time. CAM candidates also save a continuous
`heatmap_overlay.png`, distinctly labeled from the binary region selection.

The previously created nine-slide PowerPoint was imported and extended to 17
slides. Its workflow diagrams and visual style were retained. The goldfish
slide's saved pictures were verified as selected-region overlays. Its second
historical LIME setup had seven regions, 20.9% coverage, and a signed drop
of about 0.9412; the first had an empty selection. Four image slides, a
four-image gallery, an editable method table, and findings were added. The
separate interactive HTML workflow remains available and now links to an
offline experiment gallery with image and method navigation, zoom, omission
views, and a return link.

Validation completed after the final timed run:

- `python -m unittest discover -s tests -q`: 14 tests passed.
- `scripts/check_offline_notebook.py`: all 18 notebook cells executed in
  source order with offline flags; the last cell displayed 20 saved records.
- `scripts/validate_four_image_experiments.py`: all 20 saved binary masks,
  omitted images, target classes, and probability drops reproduced with the
  cached pretrained ViT.
- Artifact Tool rendered all 17 slides. The package integrity and layout
  validators reported zero findings and zero layout warnings, including an
  editable native table on slide 16.
- Local Edge browser automation loaded all 4 × 3 gallery views, all image
  assets, zoom, and navigation links with no page errors.

The presentation finalizer's file-link step returned Windows `EPERM` in both
the OneDrive checkout and a second writable workspace. The draft passed its
separate integrity and layout validators, was copied to the final presentation
path, and the final file's SHA256 matched the validated draft. The actual
PowerPoint application was not used for this QA; visual inspection used the
Artifact Tool slide renders. A first headless browser attempt lacked bundled
Chromium; the installed local Edge executable completed browser QA.

Research limits remain: four images are exploratory, there are no object masks
for quantitative localization, LIME superpixels sometimes exceed the area
target (notably peacock at 54.0%), and black omission can itself affect the
classifier. The scripted agent tests control flow only. Real LLM decisions,
API behavior, billing, and broader method rankings remain untested.

## 2026-10-23 — self-contained notebook requested by user

The previous 18-cell compact notebook was backed up as
`backups/agentic_lime_decisions.pre-inline-2026-10-23.ipynb`, and the full
implementation module was backed up as
`backups/agentic_xai.pre-inline-2026-10-23.py`. The current notebook has 41
cells. Ten labeled, tagged code cells now contain the classifier, LIME,
GradCAM++, agent request/decision loop, metrics, and reporting functions.
The demonstration cells follow those definitions. An optional final cell
contains the full four-image offline reproduction procedure; by default it
loads saved results instead of recomputing the pretrained ViT on every Run All.

The notebook no longer imports `agentic_xai.py`. That file is now a small
compatibility loader that executes only the notebook cells tagged
`agentic_core` for existing command-line scripts and tests. Thus the notebook
is the single source of implementation code, while established scripts remain
usable. The obsolete compact-notebook builder is retired.

Verification: all 41 notebook cells executed in order offline; the notebook
also executed and saved outputs in a real Jupyter kernel. All 14 existing
offline unit tests passed after converting `agentic_xai.py` to the loader.
The optional four-image recomputation was not rerun during this migration;
its saved 20 candidate records remain available in the final results cell.

## 2026-10-26 — one-image live run preparation

The user requested a live API run with one ImageNet example. The notebook Run
settings now set `OFFLINE_MODE=False`, `ALLOW_REAL_OPENAI=True`, and
`USE_CACHED_VIT=True`, with only `n01443537_goldfish.JPEG` in `IMAGE_PATHS`.
Results are separated under `outputs/live-goldfish/`. A non-secret preflight
checks image existence and API-key presence. Old synthetic outputs were cleared
from the saved notebook to avoid confusing them with live evidence. The
four-image experiment section remains explicitly offline.

The local preflight loaded cached `google/vit-base-patch16-224`, predicted
goldfish as class ID 1 with probability about 0.9975, built a multimodal
selection request offering seven methods/tools, and made zero OpenAI requests.
All 14 offline unit tests passed. The live API response, key validity, model
access, latency, and billed usage remain for the user's run to determine.

## 2026-11-03 — live review response repair

The user's live goldfish `run-04` reached OpenAI and selected GradCAM. Local
evaluation measured a 0.001799 confidence drop after masking 20.0% of the
image; the top class stayed goldfish. The agent then chose `retry` three times
but included `candidate-001` in the review action. The prior validator required
`candidate_id=null` for retry, so all three responses were rejected and the
run ended with `protocol_error` before testing another method.

The review tool description and agent instructions now state the intended ID
rule. After schema validation, non-accept decisions normalize an unused
candidate ID to null while retaining the raw response in `timeline.json`.
Acceptance still requires a measured candidate ID. The notebook's review cell
also handles sessions that ended earlier, avoiding a secondary stage error.
The pre-fix notebook is saved in
`backups/agentic_lime_decisions.pre-review-fix-2026-11-03.ipynb`.

Verification was offline: a regression test replayed the agent's extra-ID
retry pattern and continued to GradCAM++ acceptance. All 15 unit tests passed,
all 19 notebook code cells compiled, and the guarded review cell completed
after a simulated stopped session. No live API request was made during the
repair. A new live run is still needed to observe the real agent's final
choice and metrics.

## 2026-11-03 — complete review and bounded live validation

The user's newly saved `run-05` notebook had overwritten the earlier repair
with the still-open VS Code editor copy. Its first review again failed in
cell 13's `validate_selection`: the real agent chose `retry` but also supplied
`candidate-001`, while the validator demanded null. Three validation failures
ended the run as `protocol_error`; no second method was tried. The notebook
also executed an unconditional review preview cell, which could raise a
secondary stage error after the main run had already ended. The saved notebook
had no Jupyter exception output, so syntax-only or `output_type=error` checks
were insufficient. A complete backup is at
`backups/agentic_lime_decisions.pre-full-validation-2026-11-03.ipynb`.

The notebook is again the single source of implementation. `agentic_xai.py`
loads its ten tagged core cells solely for scripts and tests; the notebook
itself never imports it. Restored the review rule: acceptance still requires a
measured candidate ID; retry and inconclusive normalize an unused ID to null
while retaining the raw reply in the timeline. Guarded the review preview when
the session is no longer in review. Added `Run status` to per-candidate result
rows, so measured but unaccepted candidates no longer look like successful
completed runs. The notebook now defaults to `LIVE_MODE=False`, cached ViT,
one goldfish image, three candidate attempts, and at most six agent requests.
Offline and live artifacts have separate roots. A single `LIVE_MODE=True`
switch opts into real calls.

The live transport disables SDK retries and enforces at most six actual
Responses sends, `max_output_tokens=4096`, JPEG images up to 640 pixels per
side, no more than three image parts, and a cumulative US$1 conservative
request-cost estimate for `gpt-5-mini`. It logs token usage and request count
without key material. OpenAI exception text is sanitized to exception class,
HTTP status, and error code. The request-cost formula uses the published
standard rates (US$0.25 per million input tokens and US$2 per million output
tokens) and the documented 1.2 patch multiplier for `gpt-5-mini` image input.
This is a spending guard and estimate, not a billing receipt.

Offline validation passed: 17 unit tests, including retry-with-redundant-ID,
budget-guarded mock transport, and failed-run table visibility; all notebook
cells executed source-order with the cached pretrained ViT and zero API calls;
and the 41-cell notebook executed in a fresh Jupyter kernel. Actual pretrained
ViT checks ran LIME (100 perturbations), GradCAM, and GradCAM++ on the goldfish;
the CAM check also used the labeled two-goldfish composite. GradCAM and
GradCAM++ produced finite, distinct, repeatable maps. The composite is only
an alignment/software fixture; it has no natural-image localization ground
truth. The notebook environment has CPU PyTorch and no CUDA; `torchvision` is
not installed or required by the declared notebook dependencies.

One controlled fresh-kernel live notebook execution was saved as
`outputs/live-validation-2026-11-03.ipynb`, with run artifacts at
`outputs/live-goldfish/goldfish/run-06/`. It made exactly six real API requests
and received valid Responses outputs on all six: select GradCAM, retry; select
LIME Felzenszwalb, retry; select GradCAM++, stop inconclusive. No provider,
schema, or unhandled notebook errors occurred. The model's original goldfish
target probability was 0.997514725. The candidate confidence drops after
black omission were 0.001799047 (GradCAM, 20.0% area), 0.001476645 (LIME,
13.23% area, six positive segments, local-fit R² 0.51838), and 0.000110924
(GradCAM++, 20.0% area). Top-1 stayed goldfish in every case. The agent
accepted no explanation, so the batch records one attempted image, zero
completed images, coverage 0, and DIR null. The six calls used 12,997 input
and 4,736 output tokens, an estimated US$0.01272125 at published rates; the
transport's cumulative conservative ceiling was US$0.06392325. No more live
requests were made after the six-call limit.

An independent audit loaded every `arrays.npz` and saved PNG, confirmed mask
shape, black omitted pixels, unchanged unmasked pixels, reported area, signed
probability differences, CIR, and unchanged top-1. The live selection reason
for GradCAM++ mentioned tests unavailable in the current tool menu. After
the six-call run, agent instructions were clarified to name the fixed black
omission baseline, distinguish LIME's `hide_color` from evaluation omission,
state that CAM has no R², and prohibit promising unoffered tests. This final
prompt adjustment has offline checks only; its impact on real agent wording
remains unverified. Low confidence drops do not establish which method best
localizes the fish. A natural multi-instance image, localization annotations,
and omission-baseline sensitivity remain research tasks.

## 2026-11-03 — live default requested by user

At the user's explicit request, the notebook Run settings now save
`LIVE_MODE=True`, which derives `OFFLINE_MODE=False` and
`ALLOW_REAL_OPENAI=True`. The notebook's opening text and README now state
that Run All makes live requests by default. The six-request per-run limit,
three-candidate limit, disabled SDK retries, bounded image inputs, and
conservative US$1 estimated-cost guard remain in place. The saved outputs from
the earlier offline verification were cleared from the main notebook so they
cannot be mistaken for a new live execution. The full pre-change notebook is
backed up at `backups/agentic_lime_decisions.pre-live-default-2026-11-03.ipynb`.
This setting change was verified statically; it made no additional API calls.
The one-image session no longer hardcodes `image_id="goldfish"`; it derives the
folder name from the selected input filename, so another image gets its own
run directory. The live transport's docstring was also updated to match the
new default. These edits made no API calls.

## 2026-11-03 — repeat goldfish audit and live goose validation

The user's next saved live goldfish `run-07` made six valid agent requests and
again stopped inconclusive. GradCAM removed 20.0% of the image and lowered
goldfish probability by only 0.001799. LIME SLIC removed 4.05% and increased
target probability by 0.001446; its weighted training-fit R² was 0.4777.
GradCAM++ removed 20.0% and lowered probability by 0.000111. The fixed-area
omission measurements were internally consistent; the stop was an evidence
judgment, not an API or parser failure. Merely increasing the attempt limit
would not repair this goldfish result or justify marking an explanation valid.
The just-saved notebook still had the earlier hardcoded `image_id="goldfish"`,
consistent with an old VS Code editor copy being saved over an external edit.
The pre-change notebook was backed up as
`backups/agentic_lime_decisions.pre-run-result-fix-2026-11-03.ipynb`.

The notebook now derives the image folder name from `IMAGE_PATHS` for its
first session and any later sessions. The generic root is
`outputs/live-image-runs/`. Every numbered image run saves a snapshot of its
own batch table, comparison table, and batch summary, while the root-level
files show the latest Run All. `result.json`, `explanation.md`, the result
table, and batch summary now state the largest **observed** signed confidence
drop and its method even if the agent accepted no explanation. This field is
explicitly descriptive and does not convert an inconclusive judgment into a
scientific success. The user can therefore review all measured candidates
without losing the comparison result. The number of candidate attempts stayed
three because the goldfish limitation arose from the measured masks, not an
exhausted opportunity to run an otherwise promising method.

After static notebook validation, a fresh-kernel **live goose** execution
used the revised code, changing only the input path in the executed copy.
It is saved at `outputs/live-validation-goose-2026-11-03.ipynb` and
`outputs/live-image-runs/goose/run-01/`. All 19 code cells executed without
Jupyter errors. Four real OpenAI requests selected LIME SLIC, reviewed and
retried it, selected GradCAM++, and accepted GradCAM++. The cached ViT's
original goose probability was 0.950369; 20% GradCAM++ omission lowered it
to 0.058592 (signed confidence drop 0.891777). The top-1 label remained goose,
so DIR was 0. The run reported one completed and one evaluated image. LIME
had 6.8% area, 0.204712 drop, and a weighted training-fit R² around 0.39.
The saved goose mask, black omitted pixels, area, and probability arithmetic
were independently checked. The four requests used 8,949 input and 3,755
output tokens; the standard-rate cost estimate is US$0.00974725, with a
conservative transport ceiling of US$0.042271. No offline agent run was used
for this validation. The main notebook's default example was then set to goose
so its next live Run All uses the image with an observed conclusive run; other
images may still correctly end inconclusive. The full live execution was done
before this final default-path change, with the same code and goose input.

## 2026-11-03 — vulture input selected for next live run

At the user's request, the notebook's one-image `IMAGE_PATHS` setting now
points to `imagenet_sample_100/n01616318_vulture.JPEG`. The file exists and is
375 × 500 pixels, within the notebook's one-million-pixel limit. `LIVE_MODE`
remains `True`; the notebook was not executed and no API request was made for
this input change. With the current image-name derivation, its run files will
be saved under `outputs/live-image-runs/vulture/run-XX/`. The previous goose
default is backed up at
`backups/agentic_lime_decisions.pre-vulture-2026-11-03.ipynb`.

## 2026-11-03 — live-only notebook cleanup and duplicate output removal

At the user's request, the notebook was reduced from 41 to 32 cells. Removed
the standalone scripted demonstration, its request-preview cells, and the
saved four-image offline study from the live Run All path. Combined live agent
selection and review into one run cell. The notebook retains ViT, LIME,
GradCAM, GradCAM++, agent request/response validation, bounded live transport,
candidate visuals, and per-image result saving. Synthetic classifier and
scripted transport/action helpers and the unused multi-image `run_batch`
helper were moved from the notebook to the
`agentic_xai.py` compatibility loader for historical scripts and tests; the
live notebook still does not import that file. The prior notebook and loader
are backed up as `backups/agentic_lime_decisions.pre-live-only-2026-11-03.ipynb`
and `backups/agentic_xai.pre-live-only-2026-11-03.py`.

The live result cell now writes `batch_results.csv`, `batch_summary.json`,
and `workflow_comparison.csv` only into the numbered image run folder. The
three duplicate files at `outputs/live-image-runs/` were removed after
confirming every existing goose and vulture numbered run held its own copies.
The goose and vulture folders and all candidate artifacts were retained.
Notebook JSON validation and compilation of every code cell passed. The 17
existing compatibility unit tests passed; no API requests were made for this
cleanup. The latest live vulture run remains in its existing run folder and
was not overwritten.
