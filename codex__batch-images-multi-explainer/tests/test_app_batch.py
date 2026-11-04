"""Exercise the actual Streamlit batch UI without model downloads or paid calls."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def test_app_multi_image_results_and_shap_display(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from agentic_colorlime import batch as batch_module
    from agentic_colorlime.batch import BatchResult

    model_module = ModuleType('agentic_colorlime.model_runner')
    model_module.HuggingFaceImageClassifier = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, 'agentic_colorlime.model_runner', model_module)
    image_path = tmp_path / 'input.png'
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    calls = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        assert len(kwargs['images']) == 2
        candidate = dict(explainer='shap', segmentation_method='slic', method='shap_slic',
                         artifacts={'explanation_overlay': str(image_path)}, cir=.2,
                         relative_cir=.25, critical_area_fraction=.2, local_surrogate_score=None,
                         explanation_diagnostics={'additivity_residual': 0.0})
        comparison = dict(selected_cir_rank=1, evaluated_candidate_count=1,
                          highest_observed_cir=.2, absolute_gap_from_highest=0,
                          selected_is_highest_observed_cir=True)
        decision = SimpleNamespace(final_method='shap_slic', explanation_calls_used=1,
                                   raw={}, evidence_comparison=comparison, tools_executed=['shap_slic'],
                                   tools_failed={}, tools_not_executed=[], trace=[], audit_directory=str(tmp_path))
        result = SimpleNamespace(target_class_label='test', target_probability=.8, selected_candidate=candidate,
                                 decision=decision, candidates=[candidate], image_profile={}, run_dir=str(tmp_path))
        items = [dict(index=i, name=f'image-{i}', status='completed', selected_candidate=candidate) for i in range(2)]
        kwargs['on_progress'](2, 2, items[-1])
        return BatchResult(str(tmp_path), items=items, results={0: result, 1: result})

    monkeypatch.setattr(batch_module, 'run_batch', fake_batch)
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    app.radio[0].set_value('Local path').run()
    app.text_area[0].set_value(f'{image_path}\n{image_path}')
    next(w for w in app.text_input if w.label == 'OpenAI API key').set_value('offline-test')
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 1
    assert any(metric.label == 'SHAP additivity residual' for metric in app.metric)
    selector = next(w for w in app.selectbox if w.label == 'View image explanation')
    selector.set_value(1).run()
    assert not app.exception
    assert len(calls) == 1  # Viewing another result must not rerun inference.
