# Agentic XAI: live research workflow

This branch runs real local images through pretrained ViT, uses the live agent
to select and review explanations, and computes LIME, GradCAM or GradCAM++ locally.
The notebook is a single sequential Jupytext document with interactive inspection
and multi-image processing.

## Five project files

- `agentic_lime_decisions.py`: notebook code and Markdown.
- `environment.yml`: Conda environment and dependencies.
- `jupytext.toml`: local notebook pairing configuration.
- `.gitignore`: private inputs, notebooks, outputs and keys remain local.
- `README.md`: setup, execution and interpretation.

## Setup

```bash
git clone --branch seyi/zain/develop_merge https://github.com/Seyi-Daniel/agentic-colorlime-xai.git
cd agentic-colorlime-xai
conda env create -f environment.yml
conda activate agentic-lime-notebook
jupytext --sync agentic_lime_decisions.py
```

For an existing environment, use `conda env update -n agentic-lime-notebook -f environment.yml`.
Open the generated `agentic_lime_decisions.ipynb` in VS Code or JupyterLab and
select the environment's kernel. Set `IMAGE_PATHS` in the Run settings cell to
your existing image files. Put `OPENAI_API_KEY` in a local `.env` or process
environment. The first run can download pretrained ViT weights from Hugging Face.
Images and model weights are not bundled. Empty or missing image paths stop setup.

## Run

Restart the kernel and run from the top. The first image is inspected step by
step; remaining images use separate bounded sessions in the batch runner.
The live agent chooses among LIME (five segmentation methods), GradCAM and
GradCAM++. Python computes the explanation and omission measurements. The agent
then accepts a measured candidate, requests another candidate, or stops inconclusive.
The target is local top-1 unless REQUESTED_CLASS_ID supplies a valid override.

Each image allows up to three candidates and six agent decision attempts by
default. Requests use gpt-5-mini, at most 4096 output tokens, bounded image/text
inputs and no SDK retries. Live calls are billed. There is no dollar spending cap
in this implementation; configure account/project spending controls separately.
The output records provider-reported token usage rather than estimated prices.
Rerunning session creation starts a new run and a new per-image request budget.
The batch budget scales with the number of input images.

LIME deliberately perturbs/masks the real image and fits a local surrogate;
these computations are essential to the explanation algorithm. No replacement
classifier, generated input image, or scripted agent response is provided.

## Results and interpretation

Results are written under `outputs/live-research/`, with numbered per-image runs
and uniquely named batches. Each session saves result.json, timeline.json,
request_manifest.json, candidate images/arrays, and an explanation report.
The first-image walkthrough saves its CSV/summary; remaining-image batches save
their own aggregate CSV/summary. The final notebook table displays all sessions.

Signed confidence drop, CIR, actual removed area and top-1 decision change come
from actual local predictions. Best-observed candidates are descriptive, and
remain distinct from agent-accepted candidates. LIME R² is training fit, not
held-out validation. CAMs do not have a surrogate-fit R². LIME removes whole
segments while CAMs select high-valued pixels; compare actual removed area.
Agent acceptance and confidence drop alone do not establish explanation validity.

API counts are attempted provider calls, including failed requests, not a claim
that every call returned successfully. Provider usage is absent when not returned.
The table repeats the session-level API attempt count on each candidate row;
do not sum that column across candidate rows. Batch totals count each session once.
Batch errors and coverage remain explicit; DIR uses selected completed images.
No historical call estimates or prior experiment results are presented as new evidence.

## Collaborating

Edit the local notebook, then synchronize before committing:

```bash
jupytext --sync agentic_lime_decisions.ipynb
git diff -- agentic_lime_decisions.py
git add agentic_lime_decisions.py
git commit -m "Describe the change"
git push
```

Commit the Python source, not local notebook outputs. Before pulling, save/sync
and commit your changes, close the notebook editor, pull, sync the Python source,
then reopen and restart the kernel. VS Code users can install Python, Jupyter
and Jupytext Sync for notebook editing and automatic synchronization.

## Validation scope

The live-only revision is checked for syntax, Jupytext round-trip preservation,
request validation and missing-input behavior. An end-to-end real-image run
requires your images, model download/cache and API access; it is not claimed as
completed by this revision. Explanation algorithms are retained from Zain's engine.

## License


MIT License

Copyright (c) 2026 Daniel Odun-Ayo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
