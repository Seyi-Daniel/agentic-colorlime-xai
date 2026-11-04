# Agentic LIME notebook

This branch contains exactly three files:

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

Select the **Agentic LIME** kernel and run cells from top to bottom. No other
repository files or editable package installation are needed. The initial
installation needs an internet connection.

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
