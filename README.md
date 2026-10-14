# Agentic LIME notebook

`main` contains exactly three files:

- **[agentic_lime_decisions.ipynb](agentic_lime_decisions.ipynb)** — the complete, commented implementation and step-by-step runs.
- **README.md** — setup and usage instructions.
- **[environment.yml](environment.yml)** — the Conda environment for both demo and live runs.

## Set up and open

With Conda installed, run these commands from this folder:

```bash
conda env create -f environment.yml
conda activate agentic-lime-notebook
python -m ipykernel install --user --name agentic-lime-notebook --display-name "Agentic LIME"
jupyter lab agentic_lime_decisions.ipynb
```

If the environment already exists, update it instead of recreating it:

```bash
conda env update -f environment.yml
conda activate agentic-lime-notebook
```

Select the **Agentic LIME** kernel. Run the first code cell under **1. Environment
and one-time Git setup**, then continue from top to bottom. No other repository
files or editable package installation are needed. The initial installation
needs an internet connection.

## Clean notebook commits: run setup once per clone

The first code cell installs local Git settings using `nbstripout` and `nbdime`.
It creates a small cleaning script and attributes inside `.git`, so GitHub still
contains only the three files above. It is safe to run again. Re-run it if you
move the clone, recreate the environment, or select a different Python interpreter.

When you stage the notebook with Git:

- Outputs and execution counts are removed from the **staged copy**.
- Notebook and cell metadata are replaced with fixed minimal Python kernel
  metadata; editor settings, timestamps, widget state and tags are not committed.
- Cell IDs are normalized to their positions, avoiding random-ID changes.
- Code, Markdown, cell order and Markdown attachments remain tracked.
- Your working notebook keeps its outputs and local metadata.

The filter is required: if its environment or cleaning script is unavailable,
Git reports an error instead of silently staging an unclean notebook. Select the
correct kernel and re-run setup to repair it. Local exclusions cover generated
outputs, input images, notebook checkpoints and `.env` files.

`nbdime` provides cell-aware **local** diffs and merges. GitHub uses its own
notebook viewer; nbdime does not replace that viewer. A raw JSON diff can still
show cell-structure changes when you add, remove or move cells. This workflow
keeps meaningful notebook changes, including Markdown, rather than Python only.

**Every collaborator must run the setup cell in their own clone.** These settings
are local and do not travel with a clone. Editing/uploading a notebook directly
on GitHub bypasses the filter. This three-file layout has no server-side check
that can enforce cleaning for collaborators who skip setup.

Jupytext is not needed for this workflow: the `.ipynb` remains the single source
of truth, without a second `.py` file to synchronize. See the official
[nbstripout documentation](https://github.com/kynan/nbstripout) and
[nbdime Git integration guide](https://nbdime.readthedocs.io/en/stable/vcs.html).

## Two-minute check

Start from a clean checkout of the updated `main`, with no unrelated staged
changes. Run the setup cell, then run any existing demo cell and save the notebook
without editing its source. From the repository folder:

```bash
git check-attr filter diff merge -- agentic_lime_decisions.ipynb
git add agentic_lime_decisions.ipynb
git diff --cached --no-ext-diff --no-textconv -- agentic_lime_decisions.ipynb
```

The attributes should report `nbstripout`, `jupyternotebook`, `jupyternotebook`.
The diff should be empty even though you can still see outputs in Jupyter.

Next, add a Python comment inside an existing code cell, save, and repeat the last
two commands. The comment should appear in the diff; outputs and execution counts
should not. Remove the test comment, save and stage again to return to an empty
diff. No commit is needed for this check.

For an exact check of the staged data (not the working notebook):

```bash
git show :agentic_lime_decisions.ipynb | python -c 'import json,sys; n=json.load(sys.stdin); assert all(c.get("outputs", []) == [] and c.get("execution_count") is None and c["metadata"] == {} for c in n["cells"]); assert n["metadata"] == {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}; print("PASS: staged notebook is clean")'
```

`nbstripout --verify` on the working notebook can report outputs because keeping
local outputs is intentional. The staged-data check above is the relevant check.

## Everyday collaboration

After updating `main`, create a branch for your notebook changes:

```bash
git switch main
git pull --ff-only
git switch -c my-lime-change
```

Edit and run cells as usual. Save the notebook, then:

```bash
git add agentic_lime_decisions.ipynb
git diff --cached -- agentic_lime_decisions.ipynb
git commit -m "Describe the LIME change"
git push -u origin my-lime-change
```

Open a pull request into `main` on GitHub. Stage README/environment changes
explicitly when needed. If you already have uncommitted work, save it on your
working branch before switching branches.

The pre-notebook `main` is preserved on
[`archive/main-before-notebook-2026-10-13`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/tree/archive/main-before-notebook-2026-10-13).
The earlier notebook branch remains available too.

## Follow the flow

Sections 1–16 define the components. Sections 17–29 run them:

1. Load images and the classifier; inspect the original prediction.
2. Choose and lock the target class.
3. Choose segmentation and its settings.
4. Choose LIME settings and generate an explanation candidate.
5. Measure its impact and review the evidence.
6. Try another configuration or finish with a justified selection.
7. Save the results and process remaining images; inspect the batch summary.

Each `run_agent_step(session)` call advances one action. Re-running that cell
advances again. `run_agent_to_completion(session)` finishes the current session
within its limits. Re-running the session-creation cell starts a fresh run.

## Demo and live modes

The default `MODE = "demo"` uses two generated images, a synthetic classifier,
and a scripted controller with real LIME calculations. It needs no API key or
model download after installation. Its outputs demonstrate the program flow.

For the live language-model agent and pretrained ViT, set `OPENAI_API_KEY` in
your environment before launching Jupyter (or in a local `.env`). In section 2:

```python
MODE = "live"
IMAGE_PATHS = ["images/first.jpg", "images/second.jpg"]
AGENT_MODEL = "gpt-5-mini"
```

Supply your own image files; paths resolve from the notebook's working directory.
Restart the kernel and run from the top. Live mode downloads the classifier if
needed and makes paid OpenAI API calls. `SEND_VISUALS=False` sends settings,
predictions and measurements; `True` also sends image evidence.

## Agent decisions and evaluation

The agent controls all A–L decision categories: target class, segmentation method
and settings, perturbation count, hidden-region replacement, distance measure,
locality width, weighting function, maximum fitted features, feature selection,
surrogate model and regularization strength. Choices are validated and bounded.
This is LIME only, with Ridge or Lasso as the surrogate inside LIME.

One target is locked per image. The agent can retry segmentation methods with
different settings and compare candidates. The classifier, random seed, omission
color and evaluation-area policy stay fixed. Budgets prevent unlimited runs;
failed and inconclusive outcomes remain explicit.

Candidate results include local fidelity (training weighted R²), feature counts,
CIR, removed area and decision change. Training fidelity is not held-out
validation, and CIR alone does not establish explanation correctness. Batch DIR
uses the selected explanation from each completed image and reports its
denominator and coverage.

## Saved results

Running the notebook creates `outputs/lime-notebook/` locally, containing reports,
timelines, candidate images, arrays and batch summaries. Live runs also save
request/response audit records. These are generated results, separate from the
three source files on the branch. Keep API keys out of commits and clear private
live outputs before sharing the notebook.

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
