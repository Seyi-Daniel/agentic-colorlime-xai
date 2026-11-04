# Setup for the preserved Zain snapshot

The supplied notebook, README 1.md and DEVELOPMENT_LOG.md are unchanged.
The environment file is restored from zain_previous; no image, model weights,
external scripts, presentation or historical output folders were supplied.
References to those artifacts in the original README describe Zain's machine.

```bash
conda env create -f environment.yml
conda activate agentic-lime-notebook
```

Open `agentic_lime_decisions 1.ipynb` and select this environment's kernel.
Set IMAGE_PATHS to one existing local image and set OPENAI_API_KEY in .env.
Before the first run, populate the model cache (requires internet):

```bash
python -c 'from transformers import AutoImageProcessor, AutoModelForImageClassification; m="google/vit-base-patch16-224"; AutoImageProcessor.from_pretrained(m); AutoModelForImageClassification.from_pretrained(m)'
```

Run All makes real, billed API requests. The supplied cost guard is an estimate
from the original implementation, not a guaranteed current price or spending cap.
Results go to outputs/live-image-runs/<image>/run-XX/.
