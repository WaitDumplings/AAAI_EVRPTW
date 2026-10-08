"""The old baseline must use historical defaults, not current feature switches."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import original_scratch_config as original


def config(tmp_path, **overrides):
    arguments = dict(output=tmp_path / 'arm', run_name='ORIGINAL_SCRATCH',
                     data_root=tmp_path / 'AAAI_Dataset', seed=3009, epochs=300,
                     chunk_size=8, eval_interval=50)
    arguments.update(overrides)
    return original.build_original_config(**arguments)


def historical_source(path):
    result = subprocess.run(['git', 'show', f'{original.ORIGINAL_COMMIT}:{path}'],
                            cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        pytest.skip('Historical Git object is unavailable in this checkout')
    return result.stdout


def static_assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise AssertionError(f'No assignment found: {name}')


def dict_value(node, key):
    assert isinstance(node, ast.Dict)
    return next(value for candidate, value in zip(node.keys, node.values) if ast.literal_eval(candidate) == key)


def test_preserves_original_runtime_sections_and_scratch_contract(tmp_path):
    cfg = config(tmp_path)
    for key in ('model', 'env', 'critic', 'advantage', 'pbrs'):
        assert cfg[key] == original.ORIGINAL_PUBLIC_SLPPO[key]
    assert cfg['training']['gamma'] == .99
    assert cfg['training']['learning_rate'] == 1e-4
    assert cfg['training']['ppo_update_epochs'] == 4
    assert cfg['training']['ent_coef'] == .01
    assert cfg['offline']['sl_coef'] == .5
    assert cfg['offline']['use_priority_sampler'] is False
    assert cfg['offline']['priority_mix_rho'] == .5
    assert cfg['offline']['priority_alpha'] == .7
    assert cfg['data']['train_sample_mode'] == 'shuffle_cycle'
    assert cfg['offline']['expert_dataset_path'] == cfg['data']['train_dataset_path']
    assert cfg['offline']['expert_solution_path'].endswith('/dataset/vrptw/train/Cus100/expert_solutions.csv')
    assert cfg['evaluation']['eval_path'].endswith('/dataset/vrptw/val/Cus100')
    assert 'eval_limit' not in cfg['evaluation']
    assert 'eval_before_training' not in cfg['evaluation']
    for section in ('training', 'offline'):
        assert not any('checkpoint_path' in key for key in cfg[section])
    # New trainer knobs cannot silently turn a current pipeline into "legacy".
    assert 'reward_norm_mode' not in cfg['training']
    assert 'target_kl' not in cfg['training']
    assert 'normalization' not in cfg
    assert not (tmp_path / 'arm').exists()


def test_all_budget_sampling_changes_are_explained_and_configs_are_independent(tmp_path):
    frozen = deepcopy(original.ORIGINAL_PUBLIC_SLPPO)
    cfg = config(tmp_path, learning_rate=2e-4, epochs=150, chunk_size=12, eval_interval=25)
    provenance = cfg['experiment_protocol']
    changes = {item['parameter']: item for item in provenance['protocol_overrides']}
    real_changes = set()
    for section, values in original.ORIGINAL_PUBLIC_SLPPO.items():
        if not isinstance(values, dict):
            continue
        for key, before in values.items():
            if cfg[section][key] != before:
                dotted = f'{section}.{key}'
                real_changes.add(dotted)
                assert changes[dotted]['original'] == before
                assert changes[dotted]['used'] == cfg[section][key]
                assert changes[dotted]['reason']
    assert real_changes == set(changes)
    assert provenance['global_trajectories_per_rollout'] == 64 * 50
    assert provenance['attempted_optimizer_steps_per_epoch'] == 4 * 4
    assert provenance['expected_train_instances'] == 5000
    assert provenance['validation_instances'] == 1000
    assert not provenance['dataset_counts_verified_by_builder']
    cfg['model']['embedding_dim'] = 999
    assert original.ORIGINAL_PUBLIC_SLPPO == frozen
    assert config(tmp_path)['model']['embedding_dim'] == 256


def test_constants_match_original_git_public_preset_without_importing_old_runtime():
    # Parse source text only: no import of historic/current trainer or model.
    methods = ast.parse(historical_source('CaliRoute/caliroute/methods.py'))
    advantages = ast.literal_eval(static_assignment(methods, 'SLPPO_ADVANTAGE_DEFAULTS'))
    assert original.ORIGINAL_PUBLIC_SLPPO['advantage'] == advantages
    preset = dict_value(static_assignment(methods, 'METHOD_PRESETS'), 'slppo')
    kwargs = {node.arg: node.value for node in preset.keywords}
    offline = ast.literal_eval(kwargs['offline'])
    offline['method'] = ast.literal_eval(kwargs['trainer_method'])
    assert original.ORIGINAL_PUBLIC_SLPPO['offline'] == offline
    assert original.ORIGINAL_PUBLIC_SLPPO['training']['ppo_update_epochs'] == ast.literal_eval(kwargs['ppo_update_epochs'])

    public = ast.parse(historical_source('CaliRoute/caliroute/config.py'))
    builder = next(node for node in public.body if isinstance(node, ast.FunctionDef) and node.name == 'build_training_config')
    base = static_assignment(builder, 'cfg')
    for section in ('env', 'critic', 'pbrs'):
        assert original.ORIGINAL_PUBLIC_SLPPO[section] == ast.literal_eval(dict_value(base, section))
    # Check every literal emitted by the original public config builder.
    for section in ('model', 'training', 'evaluation', 'data'):
        values = dict_value(base, section)
        for key, value in zip(values.keys, values.values):
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            assert original.ORIGINAL_PUBLIC_SLPPO[section][ast.literal_eval(key)] == literal
    assert ast.literal_eval(static_assignment(public, 'DEFAULT_ROLLOUT_STEPS'))['vrptw'] == original.ORIGINAL_PUBLIC_SLPPO['training']['rollout_steps']


def test_old_environment_accepts_all_generated_env_keys(tmp_path):
    source = ast.parse(historical_source('CaliRoute/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py'))
    constructor = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    env_args = {node.arg for node in constructor.args.args}
    factory_keys = {'use_fast_env', 'use_jit_mask', 'info_level'}
    assert set(config(tmp_path)['env']) <= env_args | factory_keys


def test_provenance_blob_hashes_identify_the_exact_source():
    for filename, recorded in original.ORIGINAL_SOURCE_BLOBS.items():
        payload = historical_source(filename).encode('utf-8')
        actual = hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()
        assert recorded == actual


@pytest.mark.parametrize('changes', [
    {'epochs': 0}, {'epochs': True}, {'epochs': 1.5},
    {'chunk_size': 0}, {'chunk_size': 202}, {'eval_interval': 0},
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'learning_rate': float('inf')},
    {'seed': -1}, {'seed': 2**32}, {'seed': True},
    {'run_name': ''}, {'run_name': '..'}, {'run_name': 'outside/arm'},
])
def test_invalid_protocol_does_not_write_or_prepare_anything(tmp_path, changes):
    with pytest.raises(ValueError):
        config(tmp_path, **changes)
    assert list(tmp_path.iterdir()) == []


def test_cli_only_writes_external_config_and_hash_bound_provenance(tmp_path):
    output = tmp_path / 'config' / 'legacy.yaml'
    subprocess.run([sys.executable, str(ROOT / 'scripts/original_scratch_config.py'),
                    '--output', str(output), '--data-root', str(tmp_path / 'AAAI_Dataset'),
                    '--run-name', 'ORIGINAL_CLI', '--seed', '3010', '--chunk-size', '12'],
                   check=True, cwd=tmp_path, capture_output=True, text=True)
    cfg = yaml.safe_load(output.read_text())
    provenance = json.loads(output.with_suffix('.provenance.json').read_text())
    assert cfg['training']['ppo_step_chunk_size'] == 12
    assert cfg['experiment_protocol']['seed'] == 3010
    assert provenance['original_source']['commit'] == original.ORIGINAL_COMMIT
    assert provenance['config_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance['config_path'] == str(output)
    assert {path.name for path in output.parent.iterdir()} == {'legacy.yaml', 'legacy.provenance.json'}
