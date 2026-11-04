"""Exercise the standalone command-line file with actual explainers and offline agent responses."""
import ast
import builtins
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import numpy as np
import pytest
from PIL import Image

from test_explainers_and_batch import AdditivePredictor, ScriptedResponses

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'agentic_xai_headless.py'


@pytest.fixture
def headless(tmp_path, monkeypatch):
    destination = tmp_path / SOURCE.name
    shutil.copyfile(SOURCE, destination)
    original_import = builtins.__import__

    def no_application_or_gui(name, *args, **kwargs):
        if name.startswith(('agentic_colorlime', 'streamlit', 'tkinter', 'PyQt', 'PySide')):
            raise AssertionError(f'Headless file tried to import {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_application_or_gui)
    spec = importlib.util.spec_from_file_location('headless_under_test', destination)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def configure_offline(module, monkeypatch):
    predictor = AdditivePredictor()
    states, loads = [], []
    original_agent = module.OpenAIAdaptiveAgent

    class OfflineAgent(original_agent):
        def __init__(self, **kwargs):
            self.model = 'offline-test'
            self.audit_root = kwargs['audit_root']

        def run(self, runtime, **kwargs):
            states.append(runtime)
            self.client = SimpleNamespace(responses=ScriptedResponses(runtime))
            return super().run(runtime, **kwargs)

    def load(*args, **kwargs):
        loads.append(kwargs)
        # Third-party stdout must not corrupt the command's JSON output.
        print('classifier initialized')
        return predictor

    monkeypatch.setattr(module, 'OpenAIAdaptiveAgent', OfflineAgent)
    monkeypatch.setattr(module, 'HuggingFaceImageClassifier', load)
    return states, loads


@pytest.fixture
def image_path(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 1] = 255
    path = tmp_path / 'test image.png'
    Image.fromarray(image).save(path)
    return path


def test_no_gui_code_or_project_imports():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    assert 'streamlit' not in source.lower()
    assert 'session_state' not in source
    assert not any(isinstance(n, ast.FunctionDef) and n.name == 'streamlit_main' for n in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.level
            assert not (node.module or '').startswith(('agentic_colorlime', 'streamlit', 'pandas', 'tkinter'))
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith(('agentic_colorlime', 'streamlit', 'pandas', 'tkinter')) for n in node.names)


def test_copied_file_has_command_help(tmp_path):
    copy = tmp_path / SOURCE.name
    shutil.copyfile(SOURCE, copy)
    result = subprocess.run([sys.executable, str(copy), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--image' in result.stdout and '--output-root' in result.stdout


@pytest.mark.parametrize('multiple', [False, True])
def test_cli_saves_reports_all_explainers_and_clean_stdout(headless, tmp_path, monkeypatch, capsys, image_path, multiple):
    states, loads = configure_offline(headless, monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    monkeypatch.chdir(tmp_path)
    paths = [str(image_path)] * (2 if multiple else 1)
    monkeypatch.setattr(sys, 'argv', [SOURCE.name, '--image', *paths,
                                    '--output-root', str(tmp_path / 'results with spaces'),
                                    '--lime-samples', '100', '--shap-samples', '32',
                                    '--no-send-visuals-to-agent'])
    headless.cli_main()
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert summary['succeeded'] == len(paths) and summary['failed'] == 0
    assert 'classifier initialized' in output.err
    assert f'[{len(paths)}/{len(paths)}]' in output.err
    assert len(loads) == 1 and len(states) == len(paths)
    batch_dir = Path(summary['batch_dir'])
    report = (batch_dir / 'batch_report.md').read_text()
    assert 'Read explanation' in report
    for item in summary['items']:
        directory = Path(item['run_dir'])
        result = json.loads((directory / 'result.json').read_text())
        assert {c['explainer'] for c in result['candidates']} == {'lime', 'lime_lasso', 'shap'}
        assert result['selected_candidate']['cir'] == pytest.approx(.3)
        text = (directory / 'explanation.md').read_text()
        assert 'Why the agent stopped' in text and 'Counterevidence:' in text
        assert 'CIR (absolute probability drop)' in text
        for filename in ('input_image.png', 'run_summary.json', 'candidate_results.json',
                         'agent_tool_trace.json', 'decision_timeline.json'):
            assert (directory / filename).exists()
        for candidate in result['candidates']:
            for path in candidate['artifacts'].values():
                assert Path(path).is_file()
        assert 'offline-test' not in (directory / 'result.json').read_text()


def test_partial_failure_continues_and_cli_exits_one(headless, tmp_path, monkeypatch, capsys, image_path):
    states, loads = configure_offline(headless, monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    monkeypatch.setattr(sys, 'argv', [SOURCE.name, '--image', str(tmp_path / 'missing.png'), str(image_path),
                                    '--output-root', str(tmp_path / 'outputs'), '--lime-samples', '100',
                                    '--no-send-visuals-to-agent'])
    with pytest.raises(SystemExit) as error:
        headless.cli_main()
    assert error.value.code == 1
    summary = json.loads(capsys.readouterr().out)
    assert [item['status'] for item in summary['items']] == ['failed', 'completed']
    assert len(states) == 1 and len(loads) == 1
    assert 'Error:' in (Path(summary['batch_dir']) / 'batch_report.md').read_text()


def test_classifier_initialization_failure_is_saved(headless, tmp_path, monkeypatch, image_path):
    def fail(*args, **kwargs):
        raise RuntimeError('test model load failure')

    monkeypatch.setattr(headless, 'HuggingFaceImageClassifier', fail)
    result = headless.run_batch(images=[headless.ImageInput('test', image_path)], model_id_or_path='test',
                                openai_api_key='offline-test', output_root=tmp_path / 'outputs')
    assert result.summary()['failed'] == 1
    assert 'test model load failure' in (Path(result.batch_dir) / 'batch_report.md').read_text()


def test_programmatic_results_and_bytes_input(headless, tmp_path, monkeypatch, image_path):
    configure_offline(headless, monkeypatch)
    decoded = headless.ImageInput('encoded', image_path.read_bytes()).load()
    np.testing.assert_array_equal(decoded, headless.load_image_from_bytes(io.BytesIO(image_path.read_bytes())))
    result = headless.run_experiment(image=decoded, model_id_or_path='offline', openai_api_key='offline-test',
                                     output_root=tmp_path / 'outputs', send_visuals_to_agent=False,
                                     config=headless.ExperimentConfig(lime_num_samples=100, colorlime_k=2))
    assert isinstance(result, headless.ExperimentResult)
    assert result.decision.explanation_calls_used == 3
    assert result.selected_candidate['cir'] == pytest.approx(.3)
    report_dir = Path(result.run_dir)
    link = headless.report_link('report', report_dir / 'explanation.md', report_dir.parent)
    relative = unquote(link.split('](')[1][:-1])
    assert not Path(relative).is_absolute()
    assert (report_dir.parent / relative).is_file()


def test_agent_failure_preserves_partial_records(headless, tmp_path, monkeypatch, image_path):
    class FailingAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, runtime, **kwargs):
            runtime.select_explainer('lime', 'Start with a local surrogate.')
            raise RuntimeError('test interrupted controller')

    monkeypatch.setattr(headless, 'OpenAIAdaptiveAgent', FailingAgent)
    batch = headless.run_batch(images=[headless.ImageInput('test', image_path)], model_id_or_path='test',
                               openai_api_key='offline-test', predictor=AdditivePredictor(), output_root=tmp_path)
    assert batch.summary()['failed'] == 1
    trace = list(Path(batch.batch_dir).glob('image-0001/run-*/agent_tool_trace.json'))
    assert len(trace) == 1
    assert json.loads(trace[0].read_text())[0]['event'] == 'explainer_selection'
