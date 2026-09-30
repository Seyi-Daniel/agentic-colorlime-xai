"""Run notebook definitions directly with real LIME and offline controller fixtures."""
import ast
import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/agentic_lime_decisions.ipynb'


@pytest.fixture(scope='module')
def nb():
    notebook = json.loads(NOTEBOOK.read_text())
    module = types.ModuleType('notebook_under_test')
    sys.modules[module.__name__] = module
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown' and source.startswith('# Walkthrough'):
            break
        if cell['cell_type'] == 'code':
            exec(compile(source, cell['id'], 'exec'), module.__dict__)
    return module


def picture():
    x = np.full((24, 24, 3), [10, 20, 60], dtype=np.uint8)
    x[2:20, 2:16] = [220, 20, 20]
    x[8:16, 16:22] = [25, 180, 30]
    return x


def session(nb, tmp_path, **kwargs):
    return nb.LimeSession(picture(), nb.DemoClassifier(), nb.DemoController(),
                          tmp_path / 'image', **kwargs)


def prepare(nb, s, method='slic', params=None, options=None):
    target = s.original_top1 if s.requested_class is None else s.requested_class
    nb.execute_action(s, 'choose_target', {'class_id': target, 'rationale': 'Test fixed target.'})
    nb.execute_action(s, 'choose_segmentation', {'method': method, 'rationale': 'Test segmenter.'})
    params = params or {'n_segments': 9, 'compactness': 10., 'sigma': 0.}
    nb.execute_action(s, 'configure_segmentation', {'parameters': params, 'rationale': 'Test settings.'})
    options = options or {**nb.LIME_REFERENCE_DEFAULTS, 'num_samples': 120, 'num_features': 4, 'hide_color': 'black'}
    nb.execute_action(s, 'configure_lime', {'parameters': options, 'rationale': 'Test LIME options.'})
    return options


def test_notebook_contains_all_code_without_legacy_imports():
    book = json.loads(NOTEBOOK.read_text())
    assert book['nbformat'] == 4
    assert len(book['cells']) >= 40
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert not (n.module or '').startswith(('agentic_colorlime', 'shap', 'streamlit'))
            if isinstance(n, ast.Import):
                assert not any(a.name.startswith(('agentic_colorlime', 'shap', 'streamlit')) for a in n.names)


def test_complete_flow_and_saved_evidence(nb, tmp_path):
    s = session(nb, tmp_path)
    nb.run_agent_to_completion(s, show_steps=False)
    assert s.status == 'completed'
    assert len(s.candidates) == 2
    assert s.rounds == 15
    actions = [e['action'] for e in s.timeline]
    assert actions[:7] == ['choose_target', 'choose_segmentation', 'configure_segmentation',
                          'configure_lime', 'run_lime', 'measure_impact', 'review_evidence']
    for c in s.candidates:
        assert c['impact']['cir'] >= 0
        assert c['target'] == s.target
        assert c['local_fidelity_r2'] is not None
        for name in ['arrays.npz', 'segmentation.png', 'explanation.png', 'critical_mask.png', 'omitted.png']:
            assert (Path(c['folder']) / name).is_file()
    assert (s.directory / 'explanation.md').exists()
    saved = json.loads((s.directory / 'result.json').read_text())
    assert saved['status'] == 'completed'
    assert saved['mode'] == 'offline_scripted_demo'
    # Original execution event must not acquire later review/CIR mutations.
    event = next(e for e in s.timeline if e['action'] == 'run_lime')
    assert event['result']['status'] == 'explained'
    assert 'impact' not in event['result']


@pytest.mark.parametrize('method,params', [
    ('slic', {'n_segments': 9, 'compactness': 2., 'sigma': 0.}),
    ('quickshift', {'kernel_size': 3, 'max_dist': 15., 'ratio': .5}),
    ('felzenszwalb', {'scale': 100., 'sigma': .5, 'min_size': 10}),
    ('watershed', {'markers': 9, 'compactness': .001}),
    ('watershed', {'markers': None, 'compactness': 0.}),
    ('colorlime', {'k': 3, 'n_init': 2, 'max_iter': 100, 'tol': 1e-4}),
])
def test_all_segmentations_run_real_lime(nb, tmp_path, method, params):
    s = session(nb, tmp_path)
    prepare(nb, s, method, params)
    c = nb.execute_action(s, 'run_lime', {})
    assert c['number_of_segments'] <= s.policy.max_segments
    assert c['method'] == method
    assert c['segmentation'] == params
    assert c['feature_weights']


@pytest.mark.parametrize('selection', ['auto', 'forward_selection', 'highest_weights', 'lasso_path', 'none'])
def test_each_feature_selection(nb, tmp_path, selection):
    s = session(nb, tmp_path)
    options = {**nb.LIME_REFERENCE_DEFAULTS, 'num_samples': 150, 'num_features': 3,
               'hide_color': 'black', 'feature_selection': selection}
    prepare(nb, s, options=options)
    c = nb.execute_action(s, 'run_lime', {})
    if selection == 'none':
        assert c['fitted_feature_count'] == c['number_of_segments']
    else:
        assert c['fitted_feature_count'] <= 3


@pytest.mark.parametrize('distance,kernel,baseline,surrogate,alpha', [
    ('cosine', 'exponential', 'black', 'ridge', .1),
    ('hamming', 'laplacian', 'white', 'lasso', .001),
    ('euclidean', 'inverse_quadratic', 'segment_mean', 'ridge', 2.),
    ('cosine', 'laplacian', 'custom_rgb', 'lasso', .005),
])
def test_settings_reach_lime_engine(nb, tmp_path, monkeypatch, distance, kernel, baseline, surrogate, alpha):
    s = session(nb, tmp_path)
    options = {**nb.LIME_REFERENCE_DEFAULTS, 'num_samples': 140, 'num_features': 3,
               'distance_metric': distance, 'kernel': kernel, 'kernel_width': 1.2,
               'hide_color': baseline, 'hide_rgb': [11, 22, 33] if baseline == 'custom_rgb' else None,
               'surrogate': surrogate, 'alpha': alpha}
    observed = {}
    real = nb.LimeImageExplainer
    class Spy(real):
        def __init__(self, **kw):
            observed['constructor'] = kw
            super().__init__(**kw)
        def explain_instance(self, *args, **kw):
            observed['call'] = kw
            return super().explain_instance(*args, **kw)
    monkeypatch.setattr(nb, 'LimeImageExplainer', Spy)
    prepare(nb, s, options=options)
    c = nb.execute_action(s, 'run_lime', {})
    assert c['options'] == options
    kw = observed['call']
    assert kw['num_samples'] == 140 and kw['num_features'] == 3
    assert kw['distance_metric'] == distance and kw['top_labels'] is None
    assert kw['labels'] == (s.target,)
    assert kw['model_regressor'].alpha == alpha
    assert isinstance(kw['model_regressor'], nb.Ridge if surrogate == 'ridge' else nb.Lasso)
    d = np.array([0., .5, 1.])
    expected = {'exponential': np.sqrt(np.exp(-(d/1.2)**2)),
                'laplacian': np.exp(-d/1.2), 'inverse_quadratic': 1/(1+(d/1.2)**2)}[kernel]
    np.testing.assert_allclose(observed['constructor']['kernel'](d, 1.2), expected)
    assert kw['hide_color'] == {'black': 0, 'white': 255, 'segment_mean': None,
                               'custom_rgb': [11,22,33]}[baseline]


def test_target_can_be_non_top_class_and_cannot_change(nb, tmp_path):
    s = session(nb, tmp_path, requested_class=2)
    prepare(nb, s)
    assert s.target == 2
    with pytest.raises(ValueError):
        nb.execute_action(s, 'choose_target', {'class_id': 0, 'rationale': 'Switch target.'})
    c = nb.execute_action(s, 'run_lime', {})
    impact = nb.execute_action(s, 'measure_impact', {})
    assert impact['original_target_probability'] == pytest.approx(s.probabilities[2])
    assert impact['decision_changed'] == (impact['omitted_top1'] != s.original_top1)
    assert c['target'] == 2


def test_state_gates_review_and_finish(nb, tmp_path):
    s = session(nb, tmp_path)
    prepare(nb, s)
    with pytest.raises(ValueError):
        nb.execute_action(s, 'finish_selection', {})
    nb.execute_action(s, 'run_lime', {})
    assert [t['name'] for t in nb.allowed_actions(s)] == ['measure_impact']
    nb.execute_action(s, 'measure_impact', {})
    assert [t['name'] for t in nb.allowed_actions(s)] == ['review_evidence']
    with pytest.raises(ValueError, match='uncertainty'):
        nb.execute_action(s, 'review_evidence', {'support':'Observed.', 'counterevidence':'Limits.',
                    'uncertainty':'Still uncertain.', 'next_action':'finish', 'reason':'Done.'})
    assert s.stage == 'review'


def test_invalid_actions_and_budget_are_not_success(nb, tmp_path):
    class Wrong:
        mode = 'test'
        def choose(self, s, tools): return 'finish_selection', {}
        def observe(self, e): pass
    s = session(nb, tmp_path)
    s.controller = Wrong()
    nb.run_agent_to_completion(s, False)
    assert s.status == 'protocol_error' and s.decision is None
    other = nb.LimeSession(picture(), nb.DemoClassifier(), nb.DemoController(), tmp_path/'limited',
                           nb.RunPolicy(max_rounds=2))
    nb.run_agent_to_completion(other, False)
    assert other.status == 'budget_exhausted' and other.decision is None


def test_operational_failure_recovers_and_preserves_records(nb, tmp_path, monkeypatch):
    s = session(nb, tmp_path)
    prepare(nb, s)
    real = nb.explain_candidate
    def fail(*args): raise RuntimeError('Synthetic candidate failure')
    monkeypatch.setattr(nb, 'explain_candidate', fail)
    nb.run_agent_step(s)
    assert s.stage == 'decide' and s.candidates[0]['status'] == 'failed'
    monkeypatch.setattr(nb, 'explain_candidate', real)
    nb.run_agent_to_completion(s, False)
    assert s.status == 'completed' and len(s.candidates) == 2


def test_batch_partial_failure_and_dir_denominator(nb, tmp_path):
    result = nb.run_batch([tmp_path/'missing.png', picture(), picture()], nb.DemoClassifier(),
                          nb.DemoController, tmp_path, show_steps=False)
    summary = result['summary']
    assert summary['attempted_images'] == 3 and summary['completed_images'] == 2
    assert summary['dir_denominator'] == 2 and summary['coverage'] == pytest.approx(2/3)
    assert len(summary['errors']) == 1
    assert summary['DIR'] == sum(r['decision_changed'] for r in summary['selected_results'])/2
    assert result['sessions'][0].controller is not result['sessions'][1].controller


def test_invalid_parameters_rejected_before_mutation(nb, tmp_path):
    s = session(nb, tmp_path)
    prepare(nb, s)
    s.stage = 'lime'
    original = copy.deepcopy(s.options)
    for patch in [{'num_samples': -1}, {'alpha': float('nan')}, {'kernel':'exec'},
                  {'hide_color':'custom_rgb','hide_rgb':None}, {'feature_selection':'forward_selection','num_features':100}]:
        with pytest.raises(Exception):
            nb.execute_action(s, 'configure_lime', {'parameters': {**original, **patch}, 'rationale':'Test invalid.'})
        assert s.options == original
    assert s.stage == 'lime'


def test_live_controller_protocol_without_network(nb, tmp_path):
    requests = []
    class Responses:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            name = kwargs['tools'][0]['name']
            args = ({'class_id': 0, 'rationale': 'Test target.'} if name == 'choose_target'
                    else {'method':'slic','rationale':'Test grouping.'})
            return types.SimpleNamespace(model_dump=lambda **_: {'output': [
                {'type':'reasoning','id':'reason','encrypted_content':'opaque-test'},
                {'type':'function_call','name':name,'arguments':json.dumps(args),
                 'call_id':f'call-{len(requests)}','id':f'fc-{len(requests)}'}]})
    client = types.SimpleNamespace(responses=Responses())
    s = session(nb, tmp_path)
    s.controller = nb.OpenAIController(client=client)
    nb.run_agent_step(s)
    nb.run_agent_step(s)
    assert s.stage == 'segmentation'
    assert requests[0]['tool_choice'] == 'required'
    assert requests[0]['parallel_tool_calls'] is False
    assert requests[0]['store'] is False
    assert all(t['strict'] for t in requests[0]['tools'])
    history = requests[1]['input']
    assert any(i.get('type') == 'function_call_output' and i['call_id']=='call-1' for i in history)
    assert any(i.get('type') == 'reasoning' for i in history)
    assert len(list((s.directory/'audit').glob('*-response.json'))) == 2


def test_cir_uses_fixed_target_and_clips_increases(nb, tmp_path):
    s = session(nb, tmp_path)
    s.target = 1
    s.probabilities = np.array([.6, .3, .1])
    s.original_top1 = 0
    folder = tmp_path/'candidate'; folder.mkdir()
    c = {'id':'candidate', 'folder':str(folder)}
    s.arrays['candidate'] = {'mask':np.ones(picture().shape[:2],dtype=bool)}
    class Predictor:
        def predict_proba(self, images): return np.tile([.2,.7,.1],(len(images),1))
    s.predictor = Predictor()
    impact = nb.measure_impact(s,c)
    assert impact['cir'] == 0  # Target 1 increased, although original top class lost.
    assert impact['decision_changed'] is True
    assert impact['omitted_target_probability'] == .7


def test_finish_requires_explicit_tradeoff_and_is_idempotent(nb, tmp_path):
    s = session(nb,tmp_path)
    nb.run_agent_to_completion(s,False)
    s.done=False; s.stage='decide'; s.finish_authorized=True; s.decision=None
    first,second=s.candidates
    first['impact']['cir']=.1; second['impact']['cir']=.2
    args={'candidate_id':first['id'],'rationale':'Better compactness and fit.',
          'stopping_reason':'Comparison complete.','why_not_highest_cir':'','limitations':'Training-only fit.'}
    with pytest.raises(ValueError,match='trade-off'):
        nb.execute_action(s,'finish_selection',args)
    assert s.decision is None
    args['why_not_highest_cir']='Smaller actual area and better fit justify a lower CIR.'
    nb.execute_action(s,'finish_selection',args)
    rounds=s.rounds
    assert nb.run_agent_step(s)['status']=='completed'
    assert s.rounds==rounds


def test_duplicate_attempt_and_oversized_segmentation_are_recorded(nb,tmp_path):
    s=session(nb,tmp_path)
    prepare(nb,s)
    key=json.dumps({'method':s.method,'segmentation':s.segmentation,'options':s.options},sort_keys=True)
    s.seen.add(nb.hashlib.sha256(key.encode()).hexdigest())
    event=nb.run_agent_step(s)
    assert event['status']=='error' and 'already attempted' in event['error']
    assert s.stage=='decide' and len(s.candidates)==1
    with pytest.raises(ValueError,match='limit'):
        nb.segment_image(picture(),'colorlime',{'k':3,'n_init':1,'max_iter':10,'tol':1e-4},
                         nb.RunPolicy(max_segments=2))


def test_no_completed_images_has_null_dir(nb,tmp_path):
    result=nb.run_batch([tmp_path/'absent.png'],nb.DemoClassifier(),nb.DemoController,
                        tmp_path,show_steps=False)
    assert result['summary']['DIR'] is None
    assert result['summary']['dir_denominator']==0
    assert result['summary']['coverage']==0


def test_inconclusive_review_does_not_select(nb,tmp_path):
    s=session(nb,tmp_path)
    prepare(nb,s)
    nb.execute_action(s,'run_lime',{})
    nb.execute_action(s,'measure_impact',{})
    nb.execute_action(s,'review_evidence',{'support':'Some effect.','counterevidence':'Poor fidelity.',
                      'uncertainty':'Unresolved.','next_action':'inconclusive','reason':'Insufficient evidence.'})
    assert s.status=='inconclusive' and s.decision is None


def test_live_controller_malformed_json_is_returned_as_tool_error(nb,tmp_path):
    class Responses:
        def create(self,**kwargs):
            return types.SimpleNamespace(model_dump=lambda **_: {'output':[
                {'type':'function_call','call_id':'broken','name':'choose_target','arguments':'{bad json'}]})
    s=session(nb,tmp_path)
    s.controller=nb.OpenAIController(client=types.SimpleNamespace(responses=Responses()))
    event=nb.run_agent_step(s)
    assert event['status']=='error' and s.stage=='target'
    assert s.controller.history[-1]['type']=='function_call_output'
    assert s.controller.history[-1]['call_id']=='broken'
