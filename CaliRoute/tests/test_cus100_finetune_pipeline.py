import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_cus100_finetune import select_parameters


def rows(distance=100., feasible=1000):
    return [{'instance_id': f'val_{i:04d}', 'objective_distance_km': distance,
             'feasible': i < feasible, 'route_validation': {'checked': True, 'valid': True}}
            for i in range(1000)]


def test_feasibility_precedes_distance():
    selected = select_parameters(rows(100, 999), rows(110, 1000))
    assert selected['selected'] == 'candidate'
    assert selected['jointly_feasible_count'] == 999


def test_distance_selects_control_when_tuning_hurts_quality():
    assert select_parameters(rows(100), rows(100.1))['selected'] == 'control'
    assert select_parameters(rows(100), rows(99.9))['selected'] == 'candidate'


def test_numerical_ties_prefer_fewer_updates():
    assert select_parameters(rows(100), rows(100.00005))['selected'] == 'candidate'


def test_distance_uses_same_feasible_instances():
    a, b = rows(100, 999), rows(101, 999)
    a[0]['feasible'] = False
    a[999]['feasible'] = True
    a[999]['objective_distance_km'] = 10000.
    selected = select_parameters(a, b)
    assert selected['selected'] == 'control'
    assert selected['jointly_feasible_count'] == 998
    assert selected['joint_mean_distance_km'] == {'control': 100., 'candidate': 101.}


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'different_ids', 'nan', 'unchecked', 'invalid', 'all_failed'])
def test_incomplete_or_invalid_evaluations_cannot_choose_a_long_run(bad):
    a, b = rows(), rows()
    if bad == 'duplicate':
        b[-1] = copy.deepcopy(b[0])
    elif bad == 'missing':
        b.pop()
    elif bad == 'different_ids':
        b[0]['instance_id'] = 'different'
    elif bad == 'nan':
        b[0]['objective_distance_km'] = float('nan')
    elif bad == 'unchecked':
        b[0]['route_validation']['checked'] = False
    elif bad == 'invalid':
        b[0]['route_validation']['valid'] = False
    elif bad == 'all_failed':
        for row in a + b:
            row['feasible'] = False
    with pytest.raises(ValueError):
        select_parameters(a, b)


def test_partial_monitor_write_preserves_last_progress_but_final_read_is_strict(monkeypatch):
    import run_cus100_finetune as pipeline
    import json

    status = {'latest_training_epoch': 20}
    def incomplete_write(spec):
        raise json.JSONDecodeError('partial best metadata', '', 0)
    monkeypatch.setattr(pipeline, 'phase_progress', incomplete_write)
    pipeline.refresh_progress({}, status)
    assert status['latest_training_epoch'] == 20
    assert 'JSONDecodeError' in status['progress_read_warning']
    with pytest.raises(json.JSONDecodeError):
        pipeline.refresh_progress({}, status, finished=True)
    monkeypatch.setattr(pipeline, 'phase_progress', lambda spec: {'latest_training_epoch': 21})
    pipeline.refresh_progress({}, status)
    assert status['latest_training_epoch'] == 21
    assert 'progress_read_warning' not in status


@pytest.mark.parametrize('fail_stage', [None, 'candidate'])
def test_worker_orders_phases_and_never_reaches_test_after_failed_training(tmp_path, monkeypatch, fail_stage):
    import json
    import run_cus100_finetune as pipeline

    problem_root = tmp_path / 'cvrp'
    problem_root.mkdir()
    init = problem_root / 'ppo_init' / 'checkpoints' / 'checkpoint_best.pt'
    specs = {}
    for phase, budget in [('ppo_init', 100), ('control', 40), ('candidate', 40), ('long_candidate', 1000)]:
        output = problem_root / phase
        output.mkdir()
        checkpoints = output / 'checkpoints'
        checkpoints.mkdir()
        (checkpoints / 'checkpoint_final.pt').write_bytes(b'final')
        (checkpoints / 'checkpoint_best.pt').write_bytes(b'best')
        config = output / 'config.yaml'
        config.write_text('{}')
        specs[phase] = {'output_dir': str(output), 'config': str(config), 'config_sha256': pipeline.digest(config),
                        'checkpoint_dir': str(checkpoints), 'epochs': budget,
                        'command': ['fake_training', phase], 'environment': {}}
        if phase in ('control', 'candidate'):
            evaluation = output / 'evaluations'
            evaluation.mkdir()
            distances = 101. if phase == 'control' else 100.
            (evaluation / 'epoch_0040.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows(distances)))
    manifest = {'git_commit': 'frozen', 'code_root': str(tmp_path), 'seed': 3009,
                'test_root': '/held_out/test_release', 'gurobi_root': '/held_out/references',
                'tasks': {'cvrp': {'phases': specs, 'init_checkpoint': str(init), 'gpus': [0, 2]}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'frozen\n')
    monkeypatch.setattr(pipeline.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(pipeline, 'phase_progress', lambda spec: {
        'completed_training_epochs': spec['epochs'],
        'latest_validation': {'eval_status': 'ok', 'eval_num_instances': '1000'}})
    calls = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 123
            self.returncode = 1 if command == ['fake_training', fail_stage] else 0
            calls.append(command)
            if '--output-dir' in command:
                destination = Path(command[command.index('--output-dir') + 1])
                (destination / 'summary.json').write_text('{"status":"completed"}')
        def poll(self):
            return self.returncode
    monkeypatch.setattr(pipeline.subprocess, 'Popen', FakeProcess)
    code = pipeline.worker(tmp_path, 'cvrp')
    state = json.loads((problem_root / 'status.json').read_text())
    if fail_stage:
        assert code == 1
        assert state['state'] == 'failed'
        assert calls == [['fake_training', p] for p in ('ppo_init', 'control', 'candidate')]
        assert not (problem_root / 'test_best').exists()
    else:
        assert code == 0
        assert state['state'] == 'completed'
        assert calls[:4] == [['fake_training', p] for p in ('ppo_init', 'control', 'candidate', 'long_candidate')]
        assert len(calls) == 5
        test_command = calls[-1]
        checkpoint = test_command[test_command.index('--checkpoint') + 1]
        assert checkpoint == str(problem_root / 'long_candidate' / 'checkpoints' / 'checkpoint_best.pt')
        assert state['selection']['selected'] == 'candidate'
