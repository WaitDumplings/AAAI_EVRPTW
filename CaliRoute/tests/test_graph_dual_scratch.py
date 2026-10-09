"""Graph-encoder experiments preserve the explore protocol and scratch isolation."""
from __future__ import annotations

import copy
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


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_graph_changes_encoder_only_and_preserves_training_protocol(tmp_path, task):
    base = base_config()
    before = copy.deepcopy(base)
    common = dict(variant='optimized', output=tmp_path / 'optimized', run_name='GRAPH_TEST',
                  data_root=tmp_path / 'AAAI_Dataset', task=task, seed=3011)
    current = launch.build_config(base, **common)
    graph = launch.build_config(base, encoder_variant='graph', **common)
    assert base == before
    changed_model_keys = {key for key in set(current['model']) | set(graph['model'])
                          if current['model'].get(key) != graph['model'].get(key)}
    assert changed_model_keys == {'use_joint_graph_encoder', 'joint_graph_edge_dim',
                                 'joint_graph_dropout', 'use_edge_relation_encoder'}
    assert graph['model']['use_joint_graph_encoder'] is True
    assert graph['model']['joint_graph_edge_dim'] == 32
    assert graph['model']['joint_graph_dropout'] == 0.0
    assert graph['model']['use_resource_decoder'] and graph['model']['use_agda_v2']
    for name in ('use_edge_relation_encoder', 'use_edge_value_messages', 'use_edge_state_updates'):
        assert graph['model'][name] is False
    for section in ('data', 'env', 'training', 'evaluation', 'offline', 'advantage', 'pbrs', 'critic'):
        assert current[section] == graph[section], section
    train, offline, protocol = graph['training'], graph['offline'], graph['experiment_protocol']
    assert train['ppo_update_epochs'] == 5 and train['target_kl'] is None
    assert train['gamma'] == 1.0 and train['reward_norm_mode'] == 'physical_shared_popart'
    assert train['num_envs_per_gpu'] == 32 and train['n_traj'] == 50
    assert train['ppo_step_chunk_size'] == 8 and train['epochs'] == 1500
    assert offline['branch_exploration_enabled'] and offline['exploration_interval'] == 5
    assert protocol['encoder_variant'] == 'graph' and protocol['implementation'] == 'explore'
    assert protocol['architecture']['training_bundle'] == 'explore'
    launch.scratch.assert_scratch(graph)
    preflight = launch.preflight_config(graph, tmp_path / 'preflight')
    assert preflight['model'] == graph['model']
    assert preflight['offline']['exploration_interval'] == 1
    launch.scratch.assert_scratch(preflight)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_graph_encoder_cannot_change_the_original_baseline(tmp_path, task):
    with pytest.raises(ValueError, match='requires --variant optimized'):
        launch.build_config(base_config(), variant='original', encoder_variant='graph', task=task,
                            output=tmp_path, run_name='INVALID', data_root=tmp_path)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_graph_shell_defaults_and_overrides_are_parsed_as_requested(tmp_path, task):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake))
    for name in ('SEED', 'GPUS', 'RUN_ID', 'DATA_ROOT'):
        env.pop(name, None)
    script = ROOT / 'scripts/run_graph_rdi100_dual.sh'
    def parse(*extra):
        command = json.loads(subprocess.check_output(['bash', str(script), task, *extra], env=env, text=True))
        assert command[:2] == ['-B', str(ROOT / 'scripts/run_evrptw_dual_scratch.py')]
        return launch.make_parser().parse_args(command[2:])
    defaults = parse()
    assert defaults.task == task and defaults.variant == 'optimized'
    assert defaults.encoder_variant == 'graph' and defaults.launch
    assert defaults.seed == 3011 and defaults.epochs == 1500 and defaults.gpus == '0,1'
    assert defaults.batch_per_gpu == 32 and defaults.chunk_size == 8 and defaults.expert_chunk_size == 64
    override = parse('--prepare-only', '--gpus', '1,2', '--epochs', '300', '--batch-per-gpu', '64', '--chunk-size', '32')
    assert override.prepare_only and not override.launch
    assert override.gpus == '1,2' and override.epochs == 300
    assert override.batch_per_gpu == 64 and override.chunk_size == 32


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_graph_cpu_prepare_freezes_each_task_without_weights_or_processes(scratch_prepared, monkeypatch, task):
    state = scratch_prepared
    monkeypatch.setattr(launch, 'CODE_ROOT', state.root)
    data = state.root.parent / 'AAAI_Dataset'
    for split, count in [('train', 5000), ('val', 1000)]:
        folder = data / 'dataset' / task / split / 'Cus100'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'instances.pkl').write_bytes(b'not unpickled during preparation')
        (folder / 'metadata.json').write_text(json.dumps(dict(num_instances=count, num_customers=100,
            num_charging_stations=20 if task == 'evrptw' else 0)))
        (folder / ('expert_solutions.csv' if split == 'train' else 'gurobi_summary.csv')).write_text('instance_id\nfixture_0\n')
    args = launch.make_parser().parse_args(['--task', task, '--variant', 'optimized', '--encoder-variant', 'graph',
        '--seed', '3011', '--prepare-only', '--data-root', str(data),
        '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml')])
    run = launch.prepare(args)
    assert f'{task.upper()}100_DUAL_SCRATCH_OPTIMIZED_GRAPH_S3011_' in run.name
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['init_checkpoint'] is None and manifest['initialization_mode'] == 'scratch'
    assert manifest['hardware_at_prepare'] is None and not state.launches
    assert manifest['additional_sources'] == {}
    assert manifest['protocol']['encoder_variant'] == 'graph'
    assert manifest['protocol']['architecture']['use_joint_graph_encoder'] is True
    assert manifest['protocol']['global_batch'] == 64 and manifest['protocol']['world_size_per_arm'] == 2
    spec = manifest['arms']['optimized']
    for stage in (spec, spec['preflight']):
        assert '--nproc-per-node=2' in stage['command']
        assert 'offline2online.train' in stage['command']
        config = yaml.safe_load(Path(stage['config']).read_text())
        assert config['model']['use_joint_graph_encoder']
        assert config['data']['problem_type'] == task
        assert config['data']['num_charging_stations'] == (20 if task == 'evrptw' else 0)
        launch.scratch.assert_scratch(config)
    shared.verify_manifest(manifest)
