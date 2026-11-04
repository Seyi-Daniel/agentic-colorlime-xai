# Agentic LIME notebook

This project has three source files:

- **[agentic_lime_decisions.ipynb](agentic_lime_decisions.ipynb)** — all implementation code and the sequential walkthrough.
- **README.md** — installation, usage, and collaboration instructions.
- **[environment.yml](environment.yml)** — the Conda environment, including the notebook and collaboration tools.

## 1. Install with Conda

Install Git and Conda first. Open a terminal in this repository's folder and run:

```bash
conda env create -f environment.yml
conda activate agentic-lime-notebook
python -m ipykernel install --user --name agentic-lime-notebook --display-name "Agentic LIME"
jupyter lab agentic_lime_decisions.ipynb
```

Conda creates the environment and installs all listed packages. The `pip:` section
inside `environment.yml` is processed automatically inside that Conda environment;
there is no separate requirements file or manual package-by-package installation.
The initial installation needs internet access.

If the environment already exists, replace the first command with:

```bash
conda env update -n agentic-lime-notebook -f environment.yml
```

`jupyter lab ...` opens the notebook for editing. You can also open it in another
notebook editor. Select the **Agentic LIME** kernel in either case.

## 2. Run the notebook's setup cell once

Run the first code cell under **1. Environment and one-time Git setup**. It
configures Git for this clone using the Python environment selected as the kernel.
It is safe to run again. Then continue through the notebook from top to bottom.

This setup cell uses two tools already listed in the Conda environment:

- **nbstripout** removes execution clutter from the copy that Git stages for a commit.
- **nbdime** makes local notebook diffs and merges aware of cells and their source.

All generated Git settings and the cleaning helper live inside the clone's hidden
`.git` folder. They do not add source files or change global Git settings.
Re-run the cell if you move the clone or recreate/change its Python environment.
The configured environment must remain installed for Git to use these tools.

When you run cells and save, your local outputs and charts remain visible. When
you run `git add`, Git prepares a cleaned copy for committing:

- Code, Markdown, cell order, and embedded Markdown attachments are preserved.
- Outputs and execution counts are removed.
- Editor metadata, timestamps, widget state, and cell tags are removed; fixed minimal Python kernel metadata remains.
- Random cell IDs are replaced with positional IDs. Inserting or moving cells can still change these IDs.

GitHub receives that cleaned notebook, so execution clutter does not obscure code
changes. Committed notebooks do not include generated outputs or charts. GitHub
uses its own notebook viewer; nbdime improves comparisons on your computer.
For a local visual conflict resolver, use `git mergetool --tool=nbdime`, inspect the
resolved cells, and stage the result.

**Each collaborator must run the setup cell in their own clone.** This three-file
layout has no pre-commit hook or GitHub check. Skipping setup, uploading notebooks,
or editing directly in GitHub can bypass cleanup. If Git reports that its required
filter is unavailable, activate the correct environment and re-run the setup cell.

Local Git exclusions cover generated outputs, input images, checkpoints, and `.env`
files. These exclusions do not remove files that are already tracked. The notebook
remains the single implementation file; no paired Python source file is required.

## 3. Check that cleanup works

Start from a clean checkout after setup, with no unrelated staged changes.
Run the setup cell or an existing demo cell, then save without editing its source.
From the repository folder, run:

```bash
git add agentic_lime_decisions.ipynb
git diff --cached
```

**Expected: no differences**, while your outputs remain visible in the editor.

Next, add a comment such as `# Collaboration test` inside an existing code cell,
save, and repeat those commands. **Expected: the added comment appears without
execution clutter.** Remove the comment, save, and stage again to return to an
empty diff. No commit is needed for this test.

This test assumes the setup changes themselves are already in your starting
commit. If you are reviewing a new setup version, its intentional source changes
will also appear until that version is committed.

## 4. Collaborate normally

Start with a clean working tree, update `main`, and create a branch:

```bash
git switch main
git pull --ff-only
git switch -c my-lime-change
```

Edit, run, and save your notebook as usual. Then review and publish your change:

```bash
git add agentic_lime_decisions.ipynb
git diff --cached
git commit -m "Describe the LIME change"
git push -u origin my-lime-change
```

Open a pull request into `main` on GitHub. Stage README or environment changes
explicitly when needed. Keep changes small and coordinate edits to the same cells.
Save uncommitted work on your working branch before switching branches.

Official tool references: [nbstripout](https://github.com/kynan/nbstripout) and
[nbdime Git integration](https://nbdime.readthedocs.io/en/stable/vcs.html).

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
