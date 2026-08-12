import numpy as np

from agentic_colorlime.cir import calculate_cir


class FakePredictor:
    def predict_proba(self, images):
        image = images if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)[0]
        confidence = float(np.asarray(image).mean() / 255.0)
        return np.array([[confidence, 1.0 - confidence]], dtype=float)

    def label(self, class_id):
        return str(class_id)


def test_cir_is_nonnegative_probability_drop():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    result = calculate_cir(
        image=image,
        critical_mask=mask,
        target_class_id=0,
        original_target_probability=1.0,
        predictor=FakePredictor(),
        omission_rgb=(0, 0, 0),
    )
    assert result.cir == 0.5
    assert result.relative_cir == 0.5
