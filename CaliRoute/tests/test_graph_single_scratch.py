"""One GPU per task uses the declared scratch and search budgets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_evrptw_dual_scratch as launch
import run_reward_norm_comparison as shared
from test_scratch_comparison import base_config
from test_scratch_comparison import prepared as scratch_prepared


@pytest.mark.parametrize('task', ['evrptw', 'vrptw'])
@pytest.mark.parametrize('encoder_variant', ['current', 'graph'])
def test_single_card_preserves_two_card_rollout_optimizer_and_search_budgets(tmp_path, task, encoder_variant):
    common = dict(variant='optimized', output=tmp_path / 'optimized', run_name='SINGLE_TEST',
                  data_root=tmp_path / 'AAAI_Dataset', task=task, seed=3011,
                  encoder_variant=encoder_variant)
    single = launch.build_config(base_config(), world_size=1, batch_per_gpu=64, **common)
    dual = launch.build_config(base_config(), world_size=2, batch_per_gpu=32, **common)
    train, protocol = single['training'], single['experiment_protocol']
    assert protocol['world_size'] == 1
    assert train['num_envs_per_gpu'] == 64 and train['n_traj'] == 50
    assert train['ppo_update_epochs'] == 5 and train['num_minibatches'] == 4
    assert train['gradient_accumulation_steps'] == 1 and train['target_kl'] is None
    assert train['learning_rate'] == 1e-4
    assert single['model'] == dual['model']
    for field, expected in [('global_instances_per_rollout', 64),
                            ('global_trajectories_per_rollout', 3200),
                            ('global_instances_per_optimizer_step', 16)]:
        assert protocol[field] == dual['experiment_protocol'][field] == expected
    assert single['offline']['exploration_instances'] == 8
    assert dual['offline']['exploration_instances'] == 4
    assert protocol['search_budget']['max_instances_per_rank'] == 8
    assert protocol['search_budget']['trajectories_per_instance'] == 8
    assert protocol['search_budget']['max_global_trajectories'] == 64
    assert dual['experiment_protocol']['search_budget']['max_global_trajectories'] == 64
    assert single['evaluation'] == dual['evaluation']
    assert single['evaluation']['eval_n_traj'] == 50
    assert single['evaluation']['eval_interval'] == 50
    assert 'eval_limit' not in single['evaluation']
    launch.scratch.assert_scratch(single)
    preflight = launch.preflight_config(single, tmp_path / 'preflight')
    assert preflight['experiment_protocol']['world_size'] == 1
    assert preflight['training']['num_envs_per_gpu'] == 64
    assert preflight['offline']['exploration_instances'] == 8
    assert preflight['offline']['exploration_interval'] == 1
    assert preflight['training']['epochs'] == 2
    launch.scratch.assert_scratch(preflight)


@pytest.mark.parametrize(('task', 'gpu'), [('evrptw', '0'), ('vrptw', '1')])
def test_single_card_shell_selects_its_task_gpu_and_allows_tuning(tmp_path, task, gpu):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake))
    for name in ('SEED', 'GPUS', 'GPU', 'RUN_ID', 'DATA_ROOT'):
        env.pop(name, None)
    script = ROOT / 'scripts/run_graph_rdi100_single.sh'

    def parse(*extra):
        command = json.loads(subprocess.check_output(['bash', str(script), task, *extra], env=env, text=True))
        assert command[:2] == ['-B', str(ROOT / 'scripts/run_evrptw_dual_scratch.py')]
        return launch.make_parser().parse_args(command[2:])

    defaults = parse()
    assert defaults.task == task and defaults.gpus == gpu
    assert defaults.single_gpu
    assert defaults.variant == 'optimized' and defaults.encoder_variant == 'graph'
    assert defaults.launch and not defaults.prepare_only
    assert defaults.seed == 3011 and defaults.epochs == 1500
    assert defaults.batch_per_gpu == 32 and defaults.chunk_size == 120
    assert defaults.expert_chunk_size == 128
    tuned = parse('--prepare-only', '--gpus', '3', '--batch-per-gpu', '96', '--chunk-size', '16',
                  '--expert-chunk-size', '128', '--epochs', '300', '--encoder-variant', 'current',
                  '--after-run', str(tmp_path / 'previous'))
    assert tuned.prepare_only and not tuned.launch
    assert tuned.gpus == '3' and tuned.batch_per_gpu == 96
    assert tuned.chunk_size == 16 and tuned.expert_chunk_size == 128
    assert tuned.epochs == 300 and tuned.encoder_variant == 'current'
    assert tuned.after_run == [tmp_path / 'previous']


def test_single_gpu_guard_rejects_a_pair_before_probing_hardware_or_writing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'CODE_ROOT', tmp_path)
    monkeypatch.setattr(shared, 'probe_requested_gpus',
                        lambda *_: pytest.fail('Invalid single-card request probed hardware'))
    args = launch.make_parser().parse_args(['--variant', 'optimized', '--encoder-variant', 'graph',
        '--single-gpu', '--gpus', '0,1', '--prepare-only'])
    with pytest.raises(ValueError, match='--single-gpu requires exactly one GPU ID'):
        launch.prepare(args)
    assert not (tmp_path / 'results').exists()


def test_two_independent_single_task_preparations_have_separate_gpus_sources_and_commands(scratch_prepared, monkeypatch):
    state = scratch_prepared
    monkeypatch.setattr(launch, 'CODE_ROOT', state.root)
    requested_gpus = []

    def cpu_probe(gpus):
        requested_gpus.append(list(gpus))
        return None

    monkeypatch.setattr(shared, 'probe_requested_gpus', cpu_probe)
    data = state.root.parent / 'AAAI_Dataset'
    runs = []
    for task, gpu in [('evrptw', 0), ('vrptw', 1)]:
        for split, count in [('train', 5000), ('val', 1000)]:
            folder = data / 'dataset' / task / split / 'Cus100'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'instances.pkl').write_bytes(b'not unpickled during preparation')
            (folder / 'metadata.json').write_text(json.dumps(dict(num_instances=count, num_customers=100,
                num_charging_stations=20 if task == 'evrptw' else 0)))
            (folder / ('expert_solutions.csv' if split == 'train' else 'gurobi_summary.csv')).write_text('instance_id\nfixture_0\n')
        args = launch.make_parser().parse_args(['--task', task, '--variant', 'optimized', '--encoder-variant', 'graph',
            '--seed', '3011', '--prepare-only', '--single-gpu', '--gpus', str(gpu), '--batch-per-gpu', '32', '--data-root', str(data),
            '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml')])
        run = launch.prepare(args)
        runs.append(run)
        assert f'{task.upper()}100_SINGLE_SCRATCH_OPTIMIZED_GRAPH_S3011_' in run.name
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['init_checkpoint'] is None and manifest['initialization_mode'] == 'scratch'
        assert manifest['hardware_at_prepare'] is None and not state.launches
        assert manifest['gpus'] == [gpu]
        assert manifest['additional_sources'] == {}
        protocol = manifest['protocol']
        assert protocol['world_size'] == protocol['world_size_per_arm'] == 1
        assert protocol['global_batch'] == 32 and protocol['n_traj'] == 50
        assert protocol['global_trajectories_per_rollout'] == 1600
        assert protocol['global_instances_per_optimizer_step'] == 8
        assert protocol['search_budget']['max_instances_per_rank'] == 8
        assert protocol['search_budget']['max_global_trajectories'] == 64
        assert protocol['gpu_preflight']['world_size'] == 1
        assert protocol['gpu_preflight']['batch_per_rank'] == 32
        assert protocol['gpu_preflight']['eval_batch_size'] == 16
        assert protocol['gpu_preflight']['eval_n_traj'] == 50
        spec = manifest['arms']['optimized']
        assert spec['required_validation_epochs'] == list(range(0, 1501, 50))
        for stage in (spec, spec['preflight']):
            command = stage['command']
            assert '--nproc-per-node=1' in command and '--nproc-per-node=2' not in command
            assert 'offline2online.train' in command and '--max-restarts=0' in command
            config = yaml.safe_load(Path(stage['config']).read_text())
            assert config['model']['use_joint_graph_encoder']
            assert config['data']['problem_type'] == task
            assert config['experiment_protocol']['world_size'] == 1
            assert config['training']['num_envs_per_gpu'] == 32
            assert config['training']['monitor_output_dir'] == str(Path(stage['output_dir']) / 'monitoring')
            assert config['offline']['exploration_instances'] == 8
            launch.scratch.assert_scratch(config)
        shared.verify_manifest(manifest)
    assert requested_gpus == [[0], [1]]
    assert runs[0] != runs[1]
    manifests = [json.loads((run / 'manifest.json').read_text()) for run in runs]
    assert manifests[0]['code_root'] != manifests[1]['code_root']
    assert manifests[0]['arms']['optimized']['checkpoint_dir'] != manifests[1]['arms']['optimized']['checkpoint_dir']
    assert not state.launches
