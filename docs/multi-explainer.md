# Multiple images and explanation methods

## Execution flow

For each image:

1. Predict its original class with the shared classifier.
2. Start a new agent conversation with that prediction and image profile.
3. Call `select_explainer` with `lime`, `lime_lasso`, or `shap` and a rationale.
4. Optionally inspect source, then execute one of the five segmentations offered
   for that explainer. All five existing segmentations are supported by all three
   implementations.
5. Rank positive segment attributions and select whole segments until the area
   target is met (or positive segments run out).
6. Calculate CIR by omitting the selected region, then review evidence and
   counterevidence.
7. Either select another explainer/segmentation pair or finish with a rationale.

There are 15 distinct pairs. A pair is attempted at most once per image; a failed
SHAP/SLIC attempt does not prevent a LIME/SLIC attempt. After every successful
explanation, CIR and evidence review remain mandatory. Selecting an explainer
commits the next execution to that family; it cannot be switched repeatedly
without execution. The final choice can refer to any successfully evaluated
candidate, subject to the existing review and CIR trade-off requirements.

## Implemented algorithms

### LIME and sparse LIME

Both use the existing LIME image sampler, distance weighting, masking baseline,
and fixed segmentation. Ordinary LIME uses its default Ridge regression. Sparse
LIME passes a scikit-learn `Lasso` regressor through LIME's `model_regressor` API.
With sample weights normalized by scikit-learn, its objective is

```
weighted squared error / (2 * number of samples) + alpha * sum(abs(coefficients))
```

The positive coefficient ranking supplies the critical mask. Lasso can produce
fewer nonzero coefficients, and a large alpha can produce an empty mask. The
record includes the alpha, full signed coefficients, and weighted surrogate
R-squared. This is a LIME configuration variant; it is not presented as a new
published algorithm or as LEMON. LEMON's published alternative-sampling approach
would require a separately validated image integration.

### Kernel SHAP

For segments numbered 1 through M, define a binary vector z indicating which
segments are visible. The model function given to `shap.KernelExplainer` creates
the corresponding masked image and returns only the original target class's
probability. The background is one all-zero vector; the explained input is an
all-one vector. The identity link keeps values in probability units:

```
p(original target | original image) = p(original target | all-hidden image) + sum(phi)
```

Hidden regions use `hide_color` (default black), or their own mean RGB color if
that setting is null. The reported SHAP values therefore describe contributions
relative to this masking reference, not a population of natural images. Some
other SHAP implementations use pixel hierarchies or gradients; this implementation
specifically uses segment-based Kernel SHAP and does not require model gradients.

`shap_num_samples` controls the requested coalition-sampling budget. SHAP may
perform fewer evaluations when the coalitions can be enumerated exactly. We
explicitly set `l1_reg="num_features(10)"` (capped at M), matching the documented
sparse feature-selection convention rather than depending on a library default.
Attributions are estimates and may be sensitive to sampling and feature selection.
The additivity residual only checks the relation above; it is not a faithfulness
or surrogate-fit score.

The callback materializes only `inference_batch_size` RGB images at a time.
`shap_max_segments` bounds the feature dimension before SHAP's own coalition
arrays are allocated. It does not silently change segmentation. Exceeding the
limit records a failed candidate that the agent can respond to. Kernel SHAP's
legacy NumPy random-state use is seeded under a lock and restored afterwards;
LIME uses its own seeded random state.

## Results and compatibility

Existing LIME IDs (`slic`, `colorlime`, etc.) remain unchanged. New pair IDs are
`shap_slic`, `lime_lasso_slic`, and so on. Each candidate has `explainer`,
`segmentation_method`, `explanation_seconds`, and `explanation_diagnostics`.
`local_surrogate_score` is null for SHAP; the legacy `lime_local_surrogate_score`
and `lime_seconds` keys only appear on LIME variants. All signed feature weights
are saved, although the displayed critical-region mask includes positive weights
only. No classifier retraining or agent-controlled numeric tuning is introduced.

Multiple-image inputs are processed sequentially. The classifier is loaded once;
each image receives a new target, runtime, conversation, and output folder.
Numbered folders allow duplicate filenames. `batch_summary.json` is updated
atomically after each status change and contains each image's result or error.
An image error does not stop the remaining images. A failed agent run preserves
partial candidate and tool records. Interrupted batches are not automatically
resumed. A classifier-initialization failure marks every item failed.

Single-image CLI calls preserve their original output structure. Multiple-image
CLI calls print the batch summary and exit with status 1 if any item failed.
The app offers multiple uploads, newline-separated paths, progress, a results
table, and per-image explanation selection.

## Validation and scope

Offline tests execute the real LIME, Lasso-LIME, and SHAP engines against a small
classifier with analytically known color contributions. They cover positive
weights, sparse/empty explanations, SHAP additivity, masking baselines, a single
segment, inference batch bounds, and the segment limit. A scripted API response
fixture drives the real agent/runtime through all three explainers, CIR, reviews,
and final selection. Batch tests cover duplicate names and continuing after errors.
These tests do not establish explanation quality on ImageNet or measure how a
live language model chooses between the new algorithms. The committed historical
benchmark and its definitions have not been changed or expanded to the new methods.

## References

- [LIME image API, including custom regressors](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image)
- [LIME image implementation](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py)
- [scikit-learn Lasso objective and sample weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [KernelExplainer: masking reference, identity link, sample budget, and feature selection](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)
- [LEMON authors' implementation](https://github.com/iamDecode/lemon)
