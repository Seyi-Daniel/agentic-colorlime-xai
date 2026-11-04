"""The commented snapshot must be complete, independently runnable, and behaviorally consistent."""
import ast
import builtins
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_colorlime.config import ExperimentConfig as ReferenceConfig
from agentic_colorlime.segmentations import SEGMENTATION_FUNCTIONS as REFERENCE_SEGMENTATIONS
from test_explainers_and_batch import AdditivePredictor, ScriptedResponses


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'agentic_xai_commented.py'


@pytest.fixture
def single(tmp_path, monkeypatch):
    # Copy only this file: no src/ package or configs/ directory accompanies it.
    copy = tmp_path / SOURCE.name
    shutil.copyfile(SOURCE, copy)
    original_import = builtins.__import__

    def disallow_project_import(name, *args, **kwargs):
        if name.startswith('agentic_colorlime'):
            raise AssertionError(f'Single-file application tried to import {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', disallow_project_import)
    spec = importlib.util.spec_from_file_location('standalone_under_test', copy)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_snapshot_contains_every_application_definition_without_package_imports():
    tree = ast.parse(SOURCE.read_text())
    present = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    required = set()
    for path in (ROOT / 'src/agentic_colorlime').glob('*.py'):
        required.update(node.name for node in ast.parse(path.read_text()).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    required.remove('main')
    required.update({'cli_main', 'streamlit_main'})
    assert required <= present
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or '').startswith('agentic_colorlime')
        if isinstance(node, ast.Import):
            assert not any(item.name.startswith('agentic_colorlime') for item in node.names)


def test_copied_file_cli_help_works_without_checkout(tmp_path):
    copy = tmp_path / SOURCE.name
    shutil.copyfile(SOURCE, copy)
    result = subprocess.run([sys.executable, str(copy), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--shap-samples' in result.stdout and '--image' in result.stdout


def test_embedded_defaults_match_agent_demo(single):
    standalone_config, embedded = single.load_profile(None)
    original_config, original_profile = single.load_profile(ROOT / 'configs/agent-demo.yaml')
    assert standalone_config.to_dict() == original_config.to_dict()
    assert embedded['application'] == original_profile['application']


@pytest.mark.parametrize('method', ['slic', 'quickshift', 'felzenszwalb', 'watershed', 'colorlime'])
def test_segmentations_match_modular_version(single, method):
    # Reference definitions were imported before the standalone import guard.
    image = np.random.default_rng(7).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    settings = dict(slic_n_segments=8, watershed_markers=8, colorlime_k=4)
    actual = single.SEGMENTATION_FUNCTIONS[method](image, single.ExperimentConfig(**settings))
    expected = REFERENCE_SEGMENTATIONS[method](image, ReferenceConfig(**settings))
    np.testing.assert_array_equal(actual.labels, expected.labels)
    assert actual.params == expected.params


def test_copied_batch_runs_all_explainers_and_isolates_images(single, tmp_path, monkeypatch):
    predictor = AdditivePredictor()
    constructions = []
    states = []
    real_agent = single.OpenAIAdaptiveAgent

    class OfflineAgent(real_agent):
        def __init__(self, **kwargs):
            self.model = 'offline-test'
            self.audit_root = kwargs['audit_root']

        def run(self, runtime, **kwargs):
            assert runtime.explanation_calls_used == 0
            states.append(runtime)
            self.client = SimpleNamespace(responses=ScriptedResponses(runtime))
            return super().run(runtime, **kwargs)

    def create_classifier(*args, **kwargs):
        constructions.append(kwargs)
        return predictor

    monkeypatch.setattr(single, 'OpenAIAdaptiveAgent', OfflineAgent)
    monkeypatch.setattr(single, 'HuggingFaceImageClassifier', create_classifier)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 1] = 255
    result = single.run_batch(
        images=[single.ImageInput('same.png', image), single.ImageInput('bad.png', b'bad'),
                single.ImageInput('same.png', image)],
        model_id_or_path='offline', openai_api_key='offline-test', output_root=tmp_path,
        config=single.ExperimentConfig(colorlime_k=2, lime_num_samples=100, shap_num_samples=32),
        send_visuals_to_agent=False,
    )
    assert len(constructions) == 1
    assert result.summary()['succeeded'] == 2 and result.summary()['failed'] == 1
    assert len(states) == 2 and states[0] is not states[1]
    assert states[0].predictor is states[1].predictor
    for item in result.results.values():
        assert {c['explainer'] for c in item.candidates} == {'lime', 'lime_lasso', 'shap'}
        assert len(item.decision.raw['evidence_reviews']) == 3
        assert item.selected_candidate['cir'] == pytest.approx(.3)
    saved = json.loads((Path(result.batch_dir) / 'batch_summary.json').read_text())
    assert [item['status'] for item in saved['items']] == ['completed', 'failed', 'completed']
    # Source inspection resolves the copied functions, including their branch comments.
    source = states[0].inspect_method_source('shap_slic')
    assert 'def run_shap(' in source['explainer_source_code']
    assert 'PERTURBATION LOOP' in source['explainer_source_code']


def test_copied_file_dispatches_to_streamlit(single):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(single.__file__, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'Agentic Color-LIME XAI'
    app.radio[0].set_value('Local path').run()
    assert not app.exception
    assert 'one per line' in app.text_area[0].label


def test_single_file_ui_runs_batch_and_reuses_saved_results(single, tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from PIL import Image

    real_agent = single.OpenAIAdaptiveAgent
    agent_calls = []

    class OfflineAgent(real_agent):
        def __init__(self, **kwargs):
            self.model = 'offline-test'
            self.audit_root = kwargs['audit_root']

        def run(self, runtime, **kwargs):
            agent_calls.append(runtime)
            self.client = SimpleNamespace(responses=ScriptedResponses(runtime))
            return super().run(runtime, **kwargs)

    monkeypatch.setattr(single, 'OpenAIAdaptiveAgent', OfflineAgent)
    monkeypatch.setattr(single, 'HuggingFaceImageClassifier', lambda *a, **kw: AdditivePredictor())
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    monkeypatch.chdir(tmp_path)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 1] = 255
    image_path = tmp_path / 'example.png'
    Image.fromarray(image).save(image_path)
    # Run the actual UI function from the copied module, with offline dependencies.
    app = AppTest.from_string('from standalone_under_test import streamlit_main\nstreamlit_main()',
                              default_timeout=60).run()
    assert not app.exception
    app.radio[0].set_value('Local path').run()
    app.text_area[0].set_value(f'{image_path}\n{image_path}')
    app.checkbox[0].set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert len(agent_calls) == 2
    selector = next(w for w in app.selectbox if w.label == 'View image explanation')
    selector.set_value(1).run()
    assert not app.exception
    assert len(agent_calls) == 2
