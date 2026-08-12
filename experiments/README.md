# Experiments

## `colorlime_benchmark.py`

This is the standalone six-method ViT/ImageNet benchmark. It preserves the
original experimental semantics, including
standard LIME's mean off-state, Color-LIME Black, four additional replacement
variants, 1,000 perturbations, deterministic seeds, checkpointing, and compact
result export.

Run `python colorlime_benchmark.py --self-test` before a full experiment.

## `colorlime_subdivision_followup.py`

This later experiment subdivides color groups using several spatial and
texture-aware strategies. It evaluated 100 correctly classified images and is
reported separately from the 84-image benchmark.

Both scripts remain standalone so each experimental result stays coupled to the
implementation that generated it.
