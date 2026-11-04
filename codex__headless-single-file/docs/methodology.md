# Benchmark methodology

## Research comparison

The reported benchmark compares default image LIME against Color-LIME Black on a
Vision Transformer. The benchmark screened the first 100 ImageNet-1k validation
examples and evaluated the 84 examples for which
`google/vit-base-patch16-224` matched the ground-truth class.

Experiment-defining settings are preserved in
[`configs/benchmark.yaml`](../configs/benchmark.yaml) and the emitted
configuration in [`results/benchmark/run_config.json`](../results/benchmark/run_config.json).
The recorded `output_dir` is repository-relative; all experiment-defining values
match the original run.

## Metrics

For a model's original predicted class `y`, original image `x`, and image
`x_omit` after blacking out an explanation's positive critical region:

```text
CIR = max(P(y | x) - P(y | x_omit), 0)
DIR = 1[argmax P(. | x_omit) != y]
```

- **Confidence Impact Ratio (CIR)** measures the nonnegative absolute drop in
  confidence for the original predicted class.
- **Decision Impact Ratio (DIR)** is the fraction of evaluated images for which
  the top-1 prediction changes after omission.

For every method, positive LIME features are ranked by weight and accumulated
until at least 20% of the image is covered. Because features differ in size, the
actual removed fraction can exceed 20%.

## Verified results

| Measure | Default LIME | Color-LIME Black |
|---|---:|---:|
| Evaluated images | 84 | 84 |
| Median CIR | 0.678376 | 0.894176 |
| Mean CIR | 0.567887 | 0.751884 |
| DIR | 0.607143 | 0.821429 |
| Decision changes | 51 | 69 |

Paired comparison:

- Color-LIME CIR was higher on 57 images and lower on 27.
- Both methods changed the decision on 45 images.
- Color-LIME alone changed it on 24 images.
- Default LIME alone changed it on 6 images.
- Neither changed it on 9 images.

All values are derived from the committed per-image CSV. Run
`python scripts/analyze_benchmark.py --verify` to recalculate them.

## Controls

- Same 84 images and target class for every method.
- Same ViT model and preprocessing.
- 1,000 LIME perturbations per method.
- Deterministic seed 42.
- Color-LIME `k=256`, weighted by original color frequency.
- CIR/DIR use black deletion and the same 20% target area for every method.

## Limitations and confounds

1. **Subset selection.** Results describe 84 correct predictions among the first
   100 validation examples, not all ImageNet classes or arbitrary images.
2. **Off-state difference.** Default LIME used its standard mean replacement
   during surrogate fitting; Color-LIME Black used black. Although CIR/DIR used
   black deletion for both, the generated explanations do not isolate
   segmentation alone.
3. **Omission artifacts.** Black deletion may create out-of-distribution inputs.
   A confidence drop supports sensitivity to the selected region but is not
   proof of causal correctness.
4. **Feature geometry.** Color groups may be spatially disconnected and may
   select background pixels sharing object colors.
5. **No human-ground-truth masks.** CIR and DIR measure model response, not
   agreement with human annotations.
6. **No broad model study.** The headline result is specific to one ViT and one
   experimental slice.

## Next validity checks

- Match off-state policies in a segmentation-only ablation.
- Repeat across multiple seeds and larger stratified samples.
- Evaluate additional architectures.
- Compare black deletion with blur, inpainting, and distribution-aware removal.
- Add localization metrics where trustworthy masks exist.
- Report stability, runtime, API cost, and uncertainty intervals.
