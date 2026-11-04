import numpy as np

from agentic_colorlime.config import ExperimentConfig
from agentic_colorlime.segmentations import SEGMENTATION_FUNCTIONS


def test_every_segmentation_returns_image_sized_integer_labels():
    y, x = np.mgrid[0:48, 0:48]
    image = np.stack([x * 5, y * 5, ((x + y) * 2)], axis=-1).clip(0, 255).astype(np.uint8)
    config = ExperimentConfig(
        slic_n_segments=16,
        watershed_markers=16,
        colorlime_k=6,
        felzenszwalb_min_size=8,
    )
    for function in SEGMENTATION_FUNCTIONS.values():
        result = function(image, config)
        assert result.labels.shape == image.shape[:2]
        assert np.issubdtype(result.labels.dtype, np.integer)
        assert len(np.unique(result.labels)) >= 1
