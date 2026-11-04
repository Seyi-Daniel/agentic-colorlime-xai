# Research timeline

Development began in July 2026 with notebook experiments comparing spatial
superpixels and color-group features. The implementation then evolved into an
agent-controlled evaluation loop.

| Period | Milestone |
|---|---|
| July 15–16, 2026 | Initial CNN/ViT experiments comparing default LIME with color-group features. |
| July 17, 2026 | First runnable agent wrapper with multiple segmenters and a deterministic fallback. |
| July 19, 2026 | Multi-`k` Color-LIME analysis and performance optimization. |
| July 21, 2026 | Sequential method execution and CIR-based evaluation. |
| July 24, 2026 | Source inspection, evidence review, stopping gates, and application-level run audits. |
| August 6, 2026 | Six-method benchmark on 84 correctly classified images. |
| August 10, 2026 | Hierarchical color-subdivision follow-up on 100 correctly classified images. |

The package in `src/agentic_colorlime/` implements the July 24 agent design.
`experiments/colorlime_benchmark.py` and `results/benchmark/` correspond to the
August 6 benchmark. `experiments/colorlime_subdivision_followup.py` contains the
August 10 exploratory extension.
