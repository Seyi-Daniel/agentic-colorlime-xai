import json
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_colorlime.batch import ImageInput, run_batch
from agentic_colorlime.config import ExperimentConfig
from agentic_colorlime.image_profile import profile_image
from agentic_colorlime.lime_engine import run_lime
from agentic_colorlime.openai_agent import OpenAIAdaptiveAgent
from agentic_colorlime.runtime import ToolRuntime
from agentic_colorlime.shap_engine import run_shap


class AdditivePredictor:
    def __init__(self):
        self.batch_sizes = []

    def predict_proba(self, images):
        images = np.asarray(images) / 255.0
        if images.ndim == 3:
            images = images[None, ...]
        self.batch_sizes.append(len(images))
        p = 0.1 + 0.6 * images[..., 0].mean(axis=(1, 2)) + 0.2 * images[..., 1].mean(axis=(1, 2))
        return np.column_stack([p, 1 - p])

    def label(self, class_id):
        return str(class_id)

    def predict_one(self, image):
        probabilities = self.predict_proba(image)[0]
        target = int(np.argmax(probabilities))
        return target, self.label(target), float(probabilities[target]), probabilities


@pytest.fixture
def example():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 1] = 255
    segments = np.zeros((8, 8), dtype=np.int32)
    segments[:, 4:] = 1
    return image, segments


@pytest.mark.parametrize('variant', ['lime', 'lime_lasso'])
def test_real_lime_variants_recover_positive_region(example, variant):
    image, segments = example
    result = run_lime(image=image, predictor=AdditivePredictor(), target_class_id=0,
                      segments=segments, config=ExperimentConfig(lime_num_samples=200), variant=variant)
    assert result.explainer == variant
    assert result.positive_features[0][0] == 0
    assert result.surrogate_score > 0.9
    assert result.actual_area_fraction == 0.5
    assert result.diagnostics['surrogate'] == ('lasso' if variant == 'lime_lasso' else 'ridge')


def test_lasso_alpha_changes_sparsity(example):
    image, segments = example
    result = run_lime(image=image, predictor=AdditivePredictor(), target_class_id=0,
                      segments=segments, config=ExperimentConfig(lime_num_samples=100, lime_lasso_alpha=1),
                      variant='lime_lasso')
    assert result.positive_features == []
    assert not result.critical_mask.any()


def test_kernel_shap_known_values_bounded_batches_and_random_state(example):
    image, segments = example
    predictor = AdditivePredictor()
    state = np.random.get_state()
    result = run_shap(image=image, predictor=predictor, target_class_id=0,
                      segments=segments * 10 + 10,
                      config=ExperimentConfig(shap_num_samples=32, inference_batch_size=1))
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert dict(result.diagnostics['feature_weights']) == pytest.approx({10: 0.3, 20: 0.1})
    assert abs(result.diagnostics['additivity_residual']) < 1e-10
    assert result.surrogate_score is None
    assert max(predictor.batch_sizes) == 1
    assert result.selected_features[0][0] == 10


@pytest.mark.parametrize('mean_baseline', [False, True])
def test_kernel_shap_single_segment_and_mean_baseline(example, mean_baseline):
    image, segments = example
    if not mean_baseline:
        segments = np.zeros_like(segments)
    result = run_shap(image=image, predictor=AdditivePredictor(), target_class_id=0,
                      segments=segments, config=ExperimentConfig(hide_color=None if mean_baseline else 0))
    assert sum(w for _, w in result.diagnostics['feature_weights']) == pytest.approx(0 if mean_baseline else 0.4)
    if mean_baseline:
        assert not result.critical_mask.any()


def test_shap_segment_limit_fails_before_inference(example):
    image, segments = example
    predictor = AdditivePredictor()
    with pytest.raises(ValueError, match='configured limit'):
        run_shap(image=image, predictor=predictor, target_class_id=0, segments=segments,
                 config=ExperimentConfig(shap_max_segments=1))
    assert predictor.batch_sizes == []


def make_runtime(tmp_path, example):
    image, _ = example
    predictor = AdditivePredictor()
    return ToolRuntime(image=image, image_profile=profile_image(image), predictor=predictor,
                       target_class_id=0, target_class_label='red', original_target_probability=0.5,
                       config=ExperimentConfig(colorlime_k=2, lime_num_samples=100, shap_num_samples=32),
                       output_dir=tmp_path)


def test_hierarchy_and_failure_recovery(tmp_path, example):
    runtime = make_runtime(tmp_path, example)
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    assert [t['name'] for t in tools] == ['select_explainer']
    assert not mapping
    runtime.select_explainer('shap', 'Test additive contributions.')
    tools, mapping = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    assert len(mapping) == 5
    assert all(runtime.catalog.get(m).explainer == 'shap' for m in mapping.values())
    with pytest.raises(RuntimeError, match='selected explainer'):
        runtime.select_explainer('lime', 'Cannot switch without execution.')
    with pytest.raises(RuntimeError, match='Select the candidate'):
        runtime.run_method('slic', selection_rationale='x', uncertainty_addressed='x', expected_signal='x', confidence=.5)
    runtime.record_method_failure('shap_slic', 'Test failure')
    assert runtime.selected_explainer is None
    assert 'slic' in runtime.unattempted_methods()
    assert 'shap_slic' not in runtime.unattempted_methods()


class ScriptedResponses:
    """Drive the real agent and runtime through all three real explanation engines."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.families = iter(['lime', 'shap', 'lime_lasso'])
        self.count = 0

    def create(self, **payload):
        names = {t['name'] for t in payload['tools']}
        runtime = self.runtime
        if 'calculate_cir' in names:
            name, arguments = 'calculate_cir', {'candidate_id': runtime.pending_cir_candidate_id}
        elif 'review_candidate_evidence' in names:
            ready = runtime.explanation_calls_used == 3
            name = 'review_candidate_evidence'
            arguments = dict(candidate_id=runtime.pending_review_candidate_id,
                             evidence_summary='Known additive image tested.', cir_assessment='supports',
                             counterevidence_considered='Masking may remove large regions.',
                             unresolved_uncertainty='' if ready else 'Compare another explainer.',
                             next_action='ready_to_finish' if ready else 'try_another_method',
                             action_reason='Evidence reviewed.')
        elif 'finish_selection' in names:
            name = 'finish_selection'
            best = runtime.best_observed_cir_candidate()
            arguments = dict(selected_candidate_id=best.candidate_id, decision_confidence=.8,
                             rationale='Known positive region was recovered.', cir_assessment='supports',
                             stopping_reason='All three explainers compared.', why_not_highest_cir='',
                             alternative_xai_suggestion='none', alternative_xai_reason='')
        elif 'select_explainer' in names:
            name = 'select_explainer'
            arguments = dict(explainer=next(self.families), rationale='Compare this attribution algorithm.')
        else:
            family = runtime.selected_explainer
            name = 'run_colorlime' if family == 'lime' else f'run_{family}_colorlime'
            arguments = dict(selection_rationale='Group the two colors.', uncertainty_addressed='Which color supports the target?',
                             expected_signal='Red receives positive weight.', confidence=.7)
        assert name in names
        self.count += 1
        return SimpleNamespace(output=[SimpleNamespace(type='function_call', name=name,
                   arguments=json.dumps(arguments), call_id=f'call-{self.count}')])


def test_real_agent_loop_across_three_explainers(tmp_path, example):
    runtime = make_runtime(tmp_path, example)
    agent = object.__new__(OpenAIAdaptiveAgent)
    agent.client = SimpleNamespace(responses=ScriptedResponses(runtime))
    agent.model = 'offline-test'
    agent.audit_root = tmp_path / 'audit'
    result = agent.run(runtime, send_visuals_to_agent=False)
    assert result.explanation_calls_used == 3
    assert {c.lime.explainer for c in runtime.evaluated_candidates()} == {'lime', 'shap', 'lime_lasso'}
    summaries = json.loads((tmp_path / 'candidate_results.json').read_text())
    shap_candidate = next(c for c in summaries if c['explainer'] == 'shap')
    assert 'lime_local_surrogate_score' not in shap_candidate
    assert shap_candidate['local_surrogate_score'] is None
    assert len({c['artifacts']['explanation_overlay'] for c in summaries}) == 3
    selections = [e for e in runtime.trace if e['event'] == 'explainer_selection']
    assert len(selections) == 3


def test_batch_keeps_duplicate_names_separate_and_continues_after_failure(tmp_path, monkeypatch, example):
    from agentic_colorlime import batch as module
    calls = []
    shared_predictor = object()

    def fake_run(**kwargs):
        assert kwargs['predictor'] is shared_predictor
        calls.append(kwargs)
        return SimpleNamespace(run_dir=str(kwargs['output_root']), target_class_label='test',
                               target_probability=.8, selected_candidate={'explainer': 'lime'},
                               decision=SimpleNamespace(raw={'rationale': 'test'}))

    monkeypatch.setattr(module, 'run_experiment', fake_run)
    image, _ = example
    batch = run_batch(images=[ImageInput('same.png', image), ImageInput('bad.png', b'invalid'),
                              ImageInput('same.png', image)], model_id_or_path='test',
                      openai_api_key='offline-test', predictor=shared_predictor, output_root=tmp_path)
    assert batch.summary()['succeeded'] == 2 and batch.summary()['failed'] == 1
    assert list(batch.results) == [0, 2]
    assert calls[0]['output_root'] != calls[1]['output_root']
    saved = json.loads((__import__('pathlib').Path(batch.batch_dir) / 'batch_summary.json').read_text())
    assert [item['status'] for item in saved['items']] == ['completed', 'failed', 'completed']
    assert 'openai_api_key' not in json.dumps(saved)


def test_batch_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match='at least one'):
        run_batch(images=[], model_id_or_path='test', openai_api_key='', output_root=tmp_path)


def test_batch_uses_fresh_agent_state_for_each_image(tmp_path, monkeypatch, example):
    from agentic_colorlime import pipeline
    states = []

    class OfflineAgent:
        def __init__(self, **kwargs):
            self.audit_root = kwargs['audit_root']

        def run(self, runtime, **kwargs):
            assert runtime.explanation_calls_used == 0
            assert runtime.pending_cir_candidate_id is None
            states.append(runtime)
            agent = object.__new__(OpenAIAdaptiveAgent)
            agent.client = SimpleNamespace(responses=ScriptedResponses(runtime))
            agent.model = 'offline-test'
            agent.audit_root = self.audit_root
            return agent.run(runtime, **kwargs)

    monkeypatch.setattr(pipeline, 'OpenAIAdaptiveAgent', OfflineAgent)
    image, _ = example
    batch = run_batch(images=[ImageInput('a', image), ImageInput('b', image)],
                      model_id_or_path='test', openai_api_key='offline-test',
                      predictor=AdditivePredictor(), output_root=tmp_path,
                      send_visuals_to_agent=False,
                      config=ExperimentConfig(colorlime_k=2, lime_num_samples=100, shap_num_samples=32))
    assert batch.summary()['failed'] == 0
    assert len(states) == 2 and states[0] is not states[1]
    assert states[0].predictor is states[1].predictor
    assert states[0].output_dir != states[1].output_dir
    assert set(states[0].candidates_by_id).isdisjoint(states[1].candidates_by_id)


def test_cli_multiple_paths_and_new_settings(monkeypatch):
    from agentic_colorlime.cli import parse_args
    monkeypatch.setattr('sys.argv', ['agentic-colorlime', '--image', 'a.png', 'b.png',
                                   '--image', 'c.png', '--shap-samples', '64',
                                   '--lime-lasso-alpha', '0.002'])
    args = parse_args()
    assert args.image == ['a.png', 'b.png', 'c.png']
    assert args.shap_samples == 64 and args.lime_lasso_alpha == .002


def test_last_failed_attempt_reopens_review_of_existing_evidence(tmp_path, example):
    runtime = make_runtime(tmp_path, example)
    runtime.select_explainer('lime', 'Start with LIME.')
    generated = runtime.run_method('colorlime', selection_rationale='Group colors.',
                                   uncertainty_addressed='Which color matters?', expected_signal='Red matters.', confidence=.5)
    runtime.calculate_candidate_cir(generated['candidate_id'])
    runtime.review_candidate_evidence(generated['candidate_id'], evidence_summary='Evidence exists.',
                                     cir_assessment='inconclusive', counterevidence_considered='Only one observation.',
                                     unresolved_uncertainty='Compare another method.', next_action='try_another_method',
                                     action_reason='More evidence would help.')
    for method in runtime.unattempted_methods():
        runtime.record_method_failure(method, 'Unavailable in this test.')
    tools, _ = OpenAIAdaptiveAgent._dynamic_tools(runtime)
    assert [t['name'] for t in tools] == ['review_candidate_evidence']
    assert runtime.pending_review_candidate_id == generated['candidate_id']


def test_illegal_cir_call_cannot_discard_pending_review(tmp_path, example):
    runtime = make_runtime(tmp_path, example)
    scripted = ScriptedResponses(runtime)
    sent_illegal = False

    def responses(**kwargs):
        nonlocal sent_illegal
        if runtime.pending_review_candidate_id and not sent_illegal:
            sent_illegal = True
            return SimpleNamespace(output=[SimpleNamespace(type='function_call', name='calculate_cir',
                       arguments=json.dumps({'candidate_id': 'unknown'}), call_id='illegal')])
        return scripted.create(**kwargs)

    agent = object.__new__(OpenAIAdaptiveAgent)
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=responses))
    agent.model = 'offline-test'
    agent.audit_root = tmp_path / 'audit'
    agent.run(runtime, send_visuals_to_agent=False)
    assert sent_illegal
    assert not runtime.failed_cir_methods
    assert len([e for e in runtime.trace if e['event'] == 'evidence_review']) == 3
