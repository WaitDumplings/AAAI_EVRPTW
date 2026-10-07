"""Budget extension waits for committed source results and preserves ownership."""
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import extend_reward_norm_comparison as extension


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_source_budget_extension_rejects_schedule_changes_and_shorter_horizon(tmp_path, monkeypatch):
    cfg = dict(training=dict(epochs=80, lr_schedule='constant', lr_warmup_epochs=0,
        entropy_initial_coef=.002, entropy_final_coef=.002), evaluation=dict(eval_interval=50))
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    manifest = dict(protocol=dict(epochs=80, world_size_per_arm=1), arms={'baseline':dict(epochs=80,config=str(path))})
    monkeypatch.setattr(extension,'verify_manifest', lambda _:None)
    extension.validate_plan(manifest,300)
    extension.validate_plan(manifest,301)
    for epochs in (40,80,0,300.5,True):
        with pytest.raises(ValueError):extension.validate_plan(manifest,epochs)
    cfg['training']['entropy_final_coef']=.001
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='entropy'):extension.validate_plan(manifest,300)
    cfg['training']['entropy_final_coef']=.002
    cfg['training']['lr_schedule']='warmup_cosine'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='LR'):extension.validate_plan(manifest,300)


def source_run(root, name, seed, state):
    path=root/'results/optimization'/name
    write(path/'manifest.json',dict(protocol=dict(seed=seed)))
    write(path/'status.json',dict(state=state))
    return path


def test_seed_selection_requires_exactly_one_active_local_run(tmp_path,monkeypatch):
    monkeypatch.setattr(extension,'CODE_ROOT',tmp_path)
    source_run(tmp_path,'REWARD_NORM_OLD',3010,'interrupted')
    expected=source_run(tmp_path,'REWARD_NORM_CURRENT',3010,'running')
    source_run(tmp_path,'REWARD_NORM_OTHER',3009,'running')
    assert extension.select_source(None,3010)==expected
    source_run(tmp_path,'REWARD_NORM_AMBIGUOUS',3010,'running')
    with pytest.raises(ValueError,match='Expected one'):extension.select_source(None,3010)


def setup_waiter(tmp_path, state='running', arm_state='running'):
    source=tmp_path/'source'
    write(source/'manifest.json',dict(protocol=dict(epochs=80)))
    write(source/'status.json',dict(state=state,arms={'baseline':dict(state=arm_state,completed_training_epochs=17)}))
    output=tmp_path/'extension'
    plan=dict(source_experiment=str(source),source_manifest_sha256=extension.digest(source/'manifest.json'),
        source_target_epochs=80,target_epochs=300,poll_seconds=1)
    write(output/'extension_plan.json',plan)
    return source,output


def test_waiter_imports_only_after_source_completion_and_launches_resume(tmp_path,monkeypatch):
    source,output=setup_waiter(tmp_path)
    write(source/'comparison.json',dict(protocol=dict(epochs=80),matched_validation_epochs={}))
    events=[]
    monkeypatch.setattr(extension.signal,'signal',lambda *a:None)
    def tick(_):
        assert not events
        pending=json.loads((output/'status.json').read_text())
        assert pending['target_epochs']==300
        assert json.loads((output/'comparison.json').read_text())['protocol']['epochs']==300
        assert pending['arms']['baseline']['completed_training_epochs']==17
        assert pending['arms']['baseline']['target_epochs']==300
        write(source/'status.json',dict(state='completed',arms={'baseline':dict(state='completed',completed_training_epochs=80)}))
    monkeypatch.setattr(extension.time,'sleep',tick)
    monkeypatch.setattr(extension,'prepare_continuation',lambda *a:events.append('import full state'))
    monkeypatch.setattr(extension,'supervise',lambda *a,**kw:events.append('start resume'))
    extension.supervise_extension(output)
    assert events==['import full state','start resume']
    assert json.loads((source/'status.json').read_text())['state']=='completed'


@pytest.mark.parametrize('state,arm_state',[('failed','failed'),('interrupted','interrupted'),('running','failed')])
def test_failed_source_cannot_launch_partial_continuation(tmp_path,monkeypatch,state,arm_state):
    source,output=setup_waiter(tmp_path,state,arm_state)
    before=(source/'status.json').read_bytes()
    monkeypatch.setattr(extension.signal,'signal',lambda *a:None)
    monkeypatch.setattr(extension,'prepare_continuation',lambda *a:pytest.fail('Must not import incomplete source'))
    with pytest.raises(ValueError,match='did not complete'):
        extension.supervise_extension(output)
    assert (source/'status.json').read_bytes()==before
    assert json.loads((output/'status.json').read_text())['state']=='failed'


def test_source_manifest_drift_rejected_before_resume(tmp_path,monkeypatch):
    source,output=setup_waiter(tmp_path)
    write(source/'manifest.json',dict(changed=True))
    monkeypatch.setattr(extension.signal,'signal',lambda *a:None)
    with pytest.raises(ValueError,match='manifest changed'):
        extension.supervise_extension(output)


@pytest.mark.parametrize('arm',['baseline','reward','normalization','combined'])
@pytest.mark.parametrize('interval', [20, 50])
def test_prepare_continuation_imports_full_state_and_preserves_training_protocol(tmp_path,arm,interval):
    import copy
    import torch
    from test_reward_norm_extension import fixture_checkpoint, fixture_history
    payload,cfg=fixture_checkpoint(tmp_path,arm)
    cfg['training']['epochs']=40
    cfg['evaluation']['eval_interval']=interval
    payload['config']=copy.deepcopy(cfg)
    payload['epoch']=40
    payload['training_resume_state'].update(completed_epoch=40,next_training_epoch=41)
    if 'reward_normalization_state' in payload:
        for name in ('actor','critic'):
            payload['reward_normalization_state'][name]['update_count']=torch.tensor(40)
    source_spec,_=fixture_history(tmp_path)
    expected_source = [0,20,40] if interval == 20 else [0,40]
    if interval == 50:
        logs = Path(source_spec['log_dir'])
        (logs/'eval_log.csv').write_text('epoch,eval_status,eval_num_instances\n0,ok,1000\n40,ok,1000\n')
        (Path(source_spec['checkpoint_dir'])/'best_checkpoint.json').write_text('{"epoch":40}')
    config_path=tmp_path/'old_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    source_spec.update(config=str(config_path),config_sha256=extension.digest(config_path),epochs=40)
    checkpoint=Path(source_spec['checkpoint_dir'])/'checkpoint_final.pt'
    torch.save(payload,checkpoint)
    old=dict(code_root=str(tmp_path/'frozen'),source=dict(files={}),
        init_checkpoint=str(checkpoint),init_checkpoint_sha256=extension.digest(checkpoint),
        inputs={},arms={arm:source_spec},protocol=dict(epochs=40),gpus=[0])
    experiment=tmp_path/'E300';experiment.mkdir()
    plan=dict(code_root=old['code_root'],source=old['source'],source_experiment=str(tmp_path/'old'),
        source_target_epochs=40,target_epochs=300,seed=3009,run_id='E300',output_root=str(tmp_path/'repo'))
    result=extension.prepare_continuation(experiment,plan,old)
    spec=result['arms'][arm]
    assert spec['restored_through_epoch']==40 and spec['epochs']==300
    assert spec['required_validation_epochs']==sorted(set(expected_source + list(range(interval,301,interval))))
    assert 'preflight' not in spec
    resumed=yaml.safe_load(Path(spec['config']).read_text())
    assert resumed['training']['epochs']==300
    assert resumed['evaluation']['eval_interval']==interval
    assert resumed['experiment_protocol']['inherited_validation_epochs']==expected_source
    for key in ('gamma','learning_rate','ppo_update_epochs','n_traj','reward_norm_mode'):
        assert resumed['training'][key]==cfg['training'][key]
    assert resumed['offline']['init_checkpoint_path']==cfg['offline']['init_checkpoint_path']
    assert resumed['offline']['resume_checkpoint_path']==spec['resume_checkpoint']
    assert extension.digest(Path(spec['resume_checkpoint']))==extension.digest(checkpoint)
    assert (Path(spec['checkpoint_dir'])/'checkpoint_best.pt').read_bytes()==b'preserve exact bytes'
    assert len((Path(spec['log_dir'])/'train_log.csv').read_text().splitlines())==41
    assert old['protocol']['epochs']==40
    extension.verify_manifest(result)
    Path(spec['resume_checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='resume checkpoint changed'):
        extension.verify_manifest(result)


def test_source_scoped_lock_prevents_duplicate_extension_preparation(tmp_path,monkeypatch):
    import fcntl
    from types import SimpleNamespace
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(extension,'select_source',lambda *a:source)
    monkeypatch.setattr(extension,'_prepare_extension',lambda *a:pytest.fail('Duplicate preparation entered'))
    with (source/'extension_prepare.lock').open('a+') as held:
        fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            extension.prepare_extension(SimpleNamespace(experiment=source,seed=None))
