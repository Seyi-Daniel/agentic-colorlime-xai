# Agentic LIME: notebook editing, readable GitHub changes

Work in a Jupyter notebook in VS Code. Commit its Jupytext Python representation
so GitHub shows code and Markdown changes rather than notebook JSON.

**The implementation remains one sequential document.** `# %%` markers preserve
code cells; `# %% [markdown]` markers preserve headings and explanations. Everyone
can reconstruct the notebook from the same tracked source.

## Files on this branch

- **[agentic_lime_decisions.py](agentic_lime_decisions.py)** — the complete notebook source tracked by Git.
- **README.md** — setup and collaboration instructions.
- **[environment.yml](environment.yml)** — the Conda environment and Python packages.
- **[jupytext.toml](jupytext.toml)** — pairing and metadata policy shared by collaborators.
- **.gitignore** — excludes local notebooks, outputs, images, and private `.env` files.
- **.vscode/extensions.json** — recommends the Python, Jupyter, and Jupytext Sync extensions.

`agentic_lime_decisions.ipynb` is generated locally and is not tracked on this
branch. The previous notebook-only workflow remains on
[`codex/notebook-collaboration`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/tree/codex/notebook-collaboration).
Your existing test-print cell is preserved in the Python source.

## 1. Clone and install with Conda

Use a new folder to try this branch:

```bash
git clone --branch codex/jupytext-notebook-workflow https://github.com/Seyi-Daniel/agentic-colorlime-xai.git agentic-lime-jupytext
cd agentic-lime-jupytext
conda env create -f environment.yml
conda activate agentic-lime-notebook
```

If that Conda environment already exists, replace the create command with:

```bash
conda env update -n agentic-lime-notebook -f environment.yml
```

Conda automatically processes the environment file's `pip:` section. There is no
separate requirements file or virtual-environment setup. Git and Conda must be
installed first, and package installation needs an internet connection.

## 2. Reconstruct and open your notebook in VS Code

From this repository folder, with the Conda environment active:

```bash
jupytext --sync agentic_lime_decisions.py
```

This creates `agentic_lime_decisions.ipynb` with the same code, Markdown, and cell
order. It does not execute any cells. Outputs appear when you run the notebook.
You can regenerate the notebook from a fresh clone whenever needed.

Open this folder in VS Code. Install the recommended extensions:

1. **Python** by Microsoft (`ms-python.python`).
2. **Jupyter** by Microsoft (`ms-toolsai.jupyter`).
3. **Jupytext Sync** by Victor Negirneac (`caenrigen.jupytext-sync`).

Use **Python: Select Interpreter** to select the `agentic-lime-notebook` Conda
environment; this lets Jupytext Sync find the installed Jupytext package. Open
`agentic_lime_decisions.ipynb` and select that environment as the notebook kernel
as well. These are separate selections in VS Code.

Keep both the `.ipynb` and `.py` files open in tabs. Edit the `.ipynb` normally.
Jupytext Sync updates the paired source when you save. Wait for synchronization
before staging. **Installing the Python package alone does not enable automatic
syncing in VS Code.** If the extension cannot find Jupytext, run **Jupytext: Locate
Python and Jupytext** or **Jupytext: Show Logs** from the Command Palette and check
that it is using the Conda environment.

You do not need `jupyter lab`, a registered named kernel, or the old Git-cleaning
setup cell for this VS Code workflow.

## 3. Edit, review, commit, and push

Run cells, inspect results, edit code or Markdown, and save as usual. Before each
commit, use the terminal in the repository folder to ensure synchronization:

```bash
conda activate agentic-lime-notebook
jupytext --sync agentic_lime_decisions.ipynb
git diff -- agentic_lime_decisions.py
git add agentic_lime_decisions.py
git diff --cached
git commit -m "Describe the notebook change"
git push
```

The explicit sync is also the fallback if you do not install the VS Code sync
extension. Commit the `.py` source, not the `.ipynb`. Stage README/environment/
configuration edits explicitly when needed. The initial clone already tracks the
remote test branch, so `git push` updates it. For a new feature branch, create it
with `git switch -c my-lime-change` and publish it with
`git push -u origin my-lime-change`; then open a pull request.

**Expected:** running cells, changing kernels, and saving outputs causes no Python
source diff. Changing a Python line or Markdown text changes that line in GitHub.
Adding a cell adds a `# %%` boundary plus its source, without notebook JSON.

## 4. Receive a collaborator's changes

Save, synchronize, and commit your own source changes first. Close the notebook
editor before pulling, so an old in-memory notebook cannot overwrite newer source.
Then run:

```bash
git pull --ff-only
jupytext --sync agentic_lime_decisions.py
```

Reopen the notebook in VS Code. The collaborator's code and cell structure are now
present. The sync can retain local outputs for unchanged cells, but those outputs
are not evidence that the new source has run. Restart the kernel and rerun the
relevant cells (or run from the top) to refresh your results.

If you and a collaborator edited the same source lines, resolve the conflict in
`.py` like an ordinary Python file, then sync and reopen the notebook. If Jupytext
reports conflicting edits to both local representations, preserve both versions
and reconcile their source before syncing; do not force an overwrite blindly.

## 5. Small collaboration test

From a fresh clone after initial synchronization:

1. Run the first code cell and save. Sync the notebook with the command above.
   `git diff -- agentic_lime_decisions.py` should be empty.
2. Add `# Collaboration test` inside an existing code cell, save, and sync.
   The Python diff should show only that comment. Commit and push it.
3. In a second clone, create the local notebook with `jupytext --sync
   agentic_lime_decisions.py`. The comment should appear in the reconstructed cell.
4. Edit a different cell in the second notebook, sync, commit, and push. In the
   first clone, close the notebook and follow the pull/sync steps above.

No notebook execution counts, editor metadata, or outputs belong in those commits.
The ignored `.ipynb` can still be marked modified by the editor because it holds
local results; Git Source Control should show the shared `.py` source edits.

## Other ways to use the same source

The `# %%` cells can also run individually in VS Code's Python Interactive window.
In JupyterLab, Jupytext supports opening `.py` files as notebooks and saving both
paired representations. Start JupyterLab from this folder with the Conda environment
active if you prefer that interface; restart its server after first installing Jupytext.

You can run this project's default offline demo directly as Python:

```bash
python agentic_lime_decisions.py
```

This runs every code cell from top to bottom, including the batch walkthrough.
It creates local results in `outputs/lime-notebook/`. Figures depend on your
Matplotlib backend; headless environments may not show windows. Notebook-style
rich display is not identical to terminal output. Future IPython magic commands
or shell escapes would need review before standalone execution.

Only notebook inputs and cell boundaries travel through the Python source.
Generated charts, execution counts, outputs, and editor metadata remain local.
Embedded cell attachments are not the collaboration format for this branch;
use separately versioned image assets if you later need persistent illustrations.
There are no Git filters, hooks, or automatic commits required by this workflow.
The old clone's notebook filter can remain unused: ignored notebooks are not staged.

Official references: [Jupytext pairing](https://jupytext.org/using/paired-notebooks/),
[VS Code integration](https://jupytext.org/integrations/vs-code/), and the
[Jupytext Sync extension](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync).

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
source files on the branch. Keep API keys out of commits and clear private
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
