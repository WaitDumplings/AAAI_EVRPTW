"""Guard the controlled experiment, resource ownership, and portable launch inputs."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_reward_norm_comparison as launch
UNITS = dict(reward_distance_scale_km=43.638668060302734, observation_distance_scale_km=43.638668060302734)
from offline2online.training_schedule import schedule_for_epoch


def source_config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/experiments/reward_norm_vrptw100.yaml').read_text())


def test_factorial_changes_only_gamma_and_normalization(tmp_path):
    base = source_config()
    original = copy.deepcopy(base)
    common = []
    for arm, (gamma, norm) in launch.ARMS.items():
        cfg = launch.build_arm(base, arm=arm, output=tmp_path / arm, run_name=arm,
            init_checkpoint=tmp_path / 'shared.pt', data_root=tmp_path / 'data', seed=3010, units=UNITS)
        assert cfg['training']['gamma'] == gamma
        assert cfg['training']['reward_norm_mode'] == norm
        assert cfg['training']['num_envs_per_gpu'] * cfg['training']['n_traj'] == 3200
        assert cfg['training']['ppo_update_epochs'] == 5
        assert cfg['evaluation']['eval_seed'] == 17003010
        assert cfg['training']['post_init_seed'] == 3010
        assert cfg['offline']['init_checkpoint_path'] == str(tmp_path / 'shared.pt')
        assert cfg['model'] == original['model']
        assert cfg['env'] == dict(original['env'], **UNITS, reward_distance_scale_mode='single_customer_repair_median')
        assert cfg['offline']['use_priority_sampler'] is False
        assert cfg['training']['epochs'] == 80
        for epoch in (1, 20, 40, 80):
            assert schedule_for_epoch(cfg, epoch) == {'learning_rate': 1e-5, 'ent_coef': .002}
        for section, key in [('training', 'gamma'), ('training', 'reward_norm_mode'),
                             ('training', 'monitor_output_dir'), ('evaluation', 'eval_output_dir')]:
            cfg[section].pop(key)
        cfg.pop('run_name')
        cfg.pop('experiment_protocol')
        common.append(cfg)
    assert all(cfg == common[0] for cfg in common)
    assert base == original


def test_resume_and_partial_evaluation_do_not_leak_from_source(tmp_path):
    base = source_config()
    base['training']['resume_checkpoint_path'] = 'old.pt'
    base['offline']['resume_start_epoch'] = 300
    base['evaluation']['eval_limit'] = 32
    base['evaluation']['eval_num_batches'] = 1
    cfg = launch.build_arm(base, arm='combined', output=tmp_path, run_name='probe',
        init_checkpoint=tmp_path / 'shared.pt', data_root=tmp_path / 'data', seed=3009, units=UNITS)
    assert not any(k.startswith('resume_') for section in ('training', 'offline') for k in cfg[section])
    assert 'eval_limit' not in cfg['evaluation'] and 'eval_num_batches' not in cfg['evaluation']
    assert cfg['evaluation']['eval_interval'] == 20
    assert cfg['evaluation']['eval_before_training'] is True
    assert cfg['experiment_protocol']['test_enabled'] is False
    assert all('/data/Maojie' not in str(v) for section in ('data', 'offline', 'evaluation') for v in cfg[section].values())


def test_completion_requires_exact_budget_final_eval_and_checkpoint(tmp_path):
    spec = dict(epochs=80, log_dir=str(tmp_path), checkpoint_dir=str(tmp_path))
    detail = dict(latest_validation=dict(epoch='80', eval_status='ok', eval_num_instances='1000'))
    (tmp_path / 'checkpoint_final.pt').touch()
    def history(epochs):
        (tmp_path / 'train_log.csv').write_text('epoch\n' + ''.join(f'{e}\n' for e in epochs))
    history(range(1, 81))
    assert launch.successful_training(spec, detail)
    for values in (range(1, 80), [*range(1, 80), 79], [*range(1, 80), 81]):
        history(values)
        assert not launch.successful_training(spec, detail)
    history(range(1, 81))
    detail['latest_validation']['epoch'] = '60'
    assert not launch.successful_training(spec, detail)
    detail['latest_validation'].update(epoch='80', eval_num_instances='32')
    assert not launch.successful_training(spec, detail)


def test_gpu_idle_requires_no_compute_and_low_memory_and_utilization():
    card = dict(has_compute_process=False, used_mib=100, utilization=0)
    assert launch.idle_gpu(card)
    assert not launch.idle_gpu(dict(card, has_compute_process=True))
    assert not launch.idle_gpu(dict(card, used_mib=1000))
    assert not launch.idle_gpu(dict(card, utilization=40))


def test_previous_automatic_test_keeps_gpu_reserved(tmp_path):
    path = tmp_path / 'status.json'
    def state(value):
        path.write_text(json.dumps(dict(state='running', arms={'old':dict(gpu=0, state=value, stage='test_best', completed_training_epochs=500)})))
    state('running')
    assert launch.prerequisite_busy([str(path)], 0)
    assert launch.prerequisite_busy([str(path)], 1) is None
    state('completed')
    assert launch.prerequisite_busy([str(path)], 0) is None
    path.write_text('{')
    assert launch.prerequisite_busy([str(path)], 0)


def test_gpu_lock_is_exclusive_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.tempfile, 'gettempdir', lambda: str(tmp_path))
    first = launch.acquire_gpu_lock('GPU-test')
    assert first is not None
    assert launch.acquire_gpu_lock('GPU-test') is None
    first.close()
    second = launch.acquire_gpu_lock('GPU-test')
    assert second is not None
    second.close()


def test_frozen_source_survives_later_main_checkout_change(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; root = repo / 'CaliRoute'
    root.mkdir(parents=True)
    (root / 'results').mkdir()
    tracked = root / 'model.py'; tracked.write_text('value = 1\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'add', 'CaliRoute/model.py'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial'], cwd=repo, check=True)
    monkeypatch.setattr(launch, 'CODE_ROOT', root)
    snapshot = tmp_path / 'frozen/CaliRoute'
    metadata = launch.source_snapshot(snapshot)
    tracked.write_text('value = 2\n')
    assert (snapshot / 'model.py').read_text() == 'value = 1\n'
    assert launch.digest(snapshot / 'model.py') == metadata['files']['model.py']
    assert (snapshot / 'results').resolve() == root / 'results'


@pytest.mark.parametrize('value', ['0,0', '-1', '0,1,2,3,4', 'a'])
def test_bad_gpu_mapping_rejected(value):
    with pytest.raises(ValueError):
        launch.parse_gpus(value)


def test_checkpoint_units_are_explicit_and_missing_values_fail():
    payload = dict(config=dict(env=dict(reward_distance_scale_km=43.5)))
    assert launch.checkpoint_units(payload) == dict(reward_distance_scale_km=43.5, observation_distance_scale_km=43.5)
    payload['config']['env']['observation_distance_scale_km'] = 44.0
    assert launch.checkpoint_units(payload)['observation_distance_scale_km'] == 44.0
    for value in (None, -1, 0, float('nan')):
        payload['config']['env']['reward_distance_scale_km'] = value
        with pytest.raises(ValueError, match='record positive'):
            launch.checkpoint_units(payload)


def test_foreign_or_stale_running_status_is_not_an_automatic_prerequisite(tmp_path, monkeypatch):
    path = tmp_path / 'status.json'
    path.write_text(json.dumps(dict(state='running', supervisor_pid=987654)))
    original = Path.read_bytes
    def cmdline(self):
        if str(self) == '/proc/987654/cmdline':
            return b'python\0unrelated.py\0--supervise\0' + str(tmp_path).encode() + b'\0'
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', cmdline)
    assert not launch.local_live_prerequisite(path)
    def valid_cmdline(self):
        if str(self) == '/proc/987654/cmdline':
            return b'python\0/run_vrptw_update_sweep.py\0--supervise\0' + str(tmp_path).encode() + b'\0'
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', valid_cmdline)
    assert launch.local_live_prerequisite(path)
    path.write_text(json.dumps(dict(state='completed', supervisor_pid=987654)))
    assert not launch.local_live_prerequisite(path)


def test_preflight_is_small_separate_and_does_not_replace_formal_initialization(tmp_path):
    cfg = launch.build_arm(source_config(), arm='combined', output=tmp_path, run_name='formal',
        init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path/'data', seed=3009, units=UNITS)
    original = copy.deepcopy(cfg)
    smoke = launch.build_preflight(cfg, tmp_path/'preflight')
    assert smoke['training']['epochs'] == 2
    assert smoke['training']['num_envs_per_gpu'] == 4
    assert smoke['training']['n_traj'] == 4
    assert smoke['training']['ppo_step_chunk_size'] == 4
    assert smoke['evaluation']['eval_limit'] == 4
    assert smoke['run_name'] == 'formal_PREFLIGHT'
    assert smoke['offline']['init_checkpoint_path'] == cfg['offline']['init_checkpoint_path']
    assert smoke['training']['reward_norm_mode'] == 'physical_shared_popart'
    assert cfg == original


def test_partial_eval_csv_does_not_crash_reporting(tmp_path):
    arms={}
    for arm in ('baseline', 'combined'):
        folder=tmp_path/arm;folder.mkdir()
        (folder/'eval_log.csv').write_text('epoch,eval_status,eval_avg_objective_distance_km\n0,ok,\n')
        arms[arm]=dict(log_dir=str(folder))
    report=launch.comparison_report(dict(arms=arms,protocol={}),dict(state='running',arms={}))
    assert report['initial_evaluation_consistent'] is None


def test_supervisor_keeps_preflight_epochs_separate_and_starts_formal_from_manifest(tmp_path, monkeypatch):
    def spec(name, epochs, count):
        folder=tmp_path/name;folder.mkdir()
        return dict(epochs=epochs, validation_instances=count, output_dir=str(folder),
            log_dir=str(folder),checkpoint_dir=str(folder),command=[name])
    formal=spec('formal',80,1000);formal['preflight']=spec('preflight',2,4)
    manifest=dict(code_root=str(tmp_path),arms={'baseline':formal},gpus=[0],idle_checks=2,
        poll_seconds=1,wait_for_experiments=[],protocol={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'status.json').write_text(json.dumps({'state':'prepared'}))
    monkeypatch.setattr(launch,'verify_manifest',lambda _: None)
    card=dict(index=0,uuid='GPU-test',name='2080 Ti',used_mib=0,total_mib=11264,utilization=0,has_compute_process=False)
    monkeypatch.setattr(launch,'gpu_snapshot',lambda:{0:card})
    monkeypatch.setattr(launch.signal,'signal',lambda *a:None)
    monkeypatch.setattr(launch.time,'sleep',lambda _:None)
    lock=(tmp_path/'gpu.lock').open('a+')
    monkeypatch.setattr(launch,'acquire_gpu_lock',lambda _:lock)
    commands=[]
    class FinishedProcess:
        def __init__(self,command,**kwargs):
            commands.append(command)
            item=formal['preflight'] if command==['preflight'] else formal
            folder=Path(item['log_dir'])
            (folder/'train_log.csv').write_text('epoch\n'+''.join(f'{e}\n' for e in range(1,item['epochs']+1)))
            (folder/'checkpoint_final.pt').touch()
            self.pid=100+len(commands)
        def poll(self):return 0
    monkeypatch.setattr(launch.subprocess,'Popen',FinishedProcess)
    def progress(spec,detail,**kwargs):
        detail.update(target_epochs=spec['epochs'],completed_training_epochs=spec['epochs'],
            latest_validation=dict(epoch=str(spec['epochs']),eval_status='ok',eval_num_instances=str(spec['validation_instances'])))
    monkeypatch.setattr(launch,'refresh_progress',progress)
    launch.supervise(tmp_path)
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']=='completed'
    assert commands==[['preflight'],['formal']]
    detail=status['arms']['baseline']
    assert detail['completed_training_epochs']==80
    assert detail['target_epochs']==80
    assert detail['preflight_progress']['completed_training_epochs']==2
    assert detail['preflight_progress']['state']=='completed'
    hardware=[json.loads(line) for line in (tmp_path/'hardware.jsonl').read_text().splitlines()]
    assert any(row['active_stages']=={'baseline':'training'} for row in hardware)
    assert all('used_mib' in row['gpus'][0] and 'utilization' in row['gpus'][0] for row in hardware)
    assert lock.closed


def test_prepare_fails_fast_for_readable_nonexistent_gpu(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(launch,'gpu_snapshot',lambda:{0:dict(name='2080 Ti')})
    with pytest.raises(ValueError,match='do not exist'):
        launch.prepare(SimpleNamespace(arms='baseline',gpus='7'))


def test_hardware_unavailable_is_permitted_during_cpu_only_preparation(monkeypatch):
    def unavailable():raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(launch,'gpu_snapshot',unavailable)
    assert launch.probe_requested_gpus([0,1,2,3]) is None


def test_mixed_hardware_block_is_rejected_when_readable(monkeypatch):
    monkeypatch.setattr(launch,'gpu_snapshot',lambda:{0:dict(name='2080 Ti'),1:dict(name='A6000')})
    with pytest.raises(ValueError,match='one model'):
        launch.probe_requested_gpus([0,1])
