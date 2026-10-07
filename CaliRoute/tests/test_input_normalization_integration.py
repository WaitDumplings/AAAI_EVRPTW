from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from offline2online import input_normalization as norm
from offline2online import trainer
from offline2online.observation_storage import snapshot_observation
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations
from test_model_design_optimizations import agent
from test_ppo_shared_forward import _candidate


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cfg(*, context=True, mode='depot_fixed', unit=43.638668):
    return {'env': {'observation_coordinate_mode': mode,
                    'observation_input_context': context,
                    'observation_distance_scale_km': unit},
            'model': {'use_physical_input_context': context,
                      'physical_input_context_hidden_dim': 32}}


def model(config):
    result = agent(use_physical_input_context=config['model']['use_physical_input_context'])
    norm.configure(result, config)
    return result


def save(path, config):
    original = model(config)
    optimizer = torch.optim.AdamW(original.parameters(), lr=.007)
    trainer.save_checkpoint(path, original, optimizer, config, epoch=3, seed=3009)
    return original


@pytest.mark.parametrize('env_flag,model_flag', [(True, False), (False, True)])
def test_config_requires_matching_environment_and_model_flags(env_flag, model_flag):
    config = cfg()
    config['env']['observation_input_context'] = env_flag
    config['model']['use_physical_input_context'] = model_flag
    with pytest.raises(ValueError, match='must agree'):
        norm.signature(config)


@pytest.mark.parametrize('unit', [None, 0., -1., float('nan'), float('inf')])
@pytest.mark.parametrize('context,mode', [(True, 'legacy_minmax'), (False, 'depot_fixed')])
def test_physical_features_require_explicit_positive_finite_fixed_distance_unit(unit, context, mode):
    with pytest.raises(ValueError, match='observation_distance_scale_km'):
        norm.signature(cfg(context=context, mode=mode, unit=unit))


def test_contract_is_customer_count_independent_and_distinct_from_reward_normalization():
    config = cfg()
    profile = norm.signature(config)
    assert len(profile['node_context_features']) == 12
    assert len(profile['graph_context_features']) == 10
    changed = {**config, 'data': {'num_customers': 1000}, 'training': {'gamma': 1.}}
    changed['env'] = {**changed['env'], 'reward_distance_scale_km': 200.}
    assert norm.signature(changed) == profile
    with pytest.raises(ValueError, match='adapter'):
        norm.configure(agent(), config)


def test_full_resume_restores_matching_profile_and_optimizer(tmp_path):
    config = cfg()
    path = tmp_path / 'checkpoint.pt'
    original = save(path, config)
    restored = model(config)
    optimizer = torch.optim.AdamW(restored.parameters(), lr=.02)
    info = trainer._load_training_checkpoint(restored, optimizer, path, 'cpu')
    assert info['optimizer_loaded']
    assert optimizer.param_groups[0]['lr'] == .007
    assert info['input_normalization']['load'] == 'full_resume'
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize('change', ['unit', 'mode', 'context'])
def test_changed_profile_rejected_before_full_resume_mutates_model_or_optimizer(tmp_path, change):
    source = cfg()
    path = tmp_path / 'checkpoint.pt'
    save(path, source)
    target = copy.deepcopy(source)
    if change == 'unit':
        target['env']['observation_distance_scale_km'] *= 2
    elif change == 'mode':
        target['env']['observation_coordinate_mode'] = 'legacy_minmax'
    else:
        target['env']['observation_input_context'] = False
        target['model']['use_physical_input_context'] = False
    restored = model(target)
    previous = copy.deepcopy(restored.state_dict())
    optimizer = torch.optim.AdamW(restored.parameters(), lr=.02)
    with pytest.raises(ValueError, match='changed on resume'):
        trainer._load_training_checkpoint(restored, optimizer, path, 'cpu', strict=False)
    assert optimizer.param_groups[0]['lr'] == .02
    for key, value in previous.items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_legacy_weights_only_input_migration_is_allowed_and_recorded(tmp_path):
    source = cfg(context=False, mode='legacy_minmax')
    source_model = model(source)
    path = tmp_path / 'old.pt'
    # Simulate a checkpoint created before input metadata existed.
    torch.save({'model_state_dict': source_model.state_dict(), 'config': source, 'epoch': 300}, path)
    target = cfg()
    upgraded = model(target)
    info = trainer._load_agent_checkpoint(upgraded, path, 'cpu', strict=False)
    migration = info['input_normalization']
    assert migration['load'] == 'weights_only' and migration['migrated']
    assert not migration['source_profile_recorded']
    assert migration['source']['observation_coordinate_mode'] == 'legacy_minmax'
    assert migration['target'] == norm.signature(target)
    assert info['missing_keys'] and all(key.startswith('backbone.physical_input_adapter.') for key in info['missing_keys'])
    out = tmp_path / 'new.pt'
    trainer.save_checkpoint(out, upgraded, torch.optim.Adam(upgraded.parameters()), target, 1, 3009)
    saved = torch.load(out, weights_only=False)
    assert saved['input_normalization_signature'] == norm.signature(target)
    assert saved['input_normalization_initialization'] == migration
    resumed = model(target)
    trainer._load_training_checkpoint(resumed, torch.optim.Adam(resumed.parameters()), out, 'cpu')
    assert resumed._input_normalization_initialization == migration
    incompatible_resume = model(target)
    with pytest.raises(ValueError, match='changed on resume'):
        trainer._load_training_checkpoint(incompatible_resume, torch.optim.Adam(incompatible_resume.parameters()), path, 'cpu', strict=False)


def test_legacy_full_resume_without_profile_still_works(tmp_path):
    config = cfg(context=False, mode='legacy_minmax')
    old_model = model(config)
    path = tmp_path / 'legacy.pt'
    torch.save({'model_state_dict': old_model.state_dict(), 'config': config}, path)
    restored = model(config)
    trainer._load_training_checkpoint(restored, torch.optim.Adam(restored.parameters()), path, 'cpu')
    for key, value in old_model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize('tamper', ['missing', 'distance', 'schema', 'features'])
def test_physical_resume_rejects_missing_or_tampered_profile(tmp_path, tamper):
    config = cfg()
    path = tmp_path / 'physical.pt'
    save(path, config)
    saved = torch.load(path, weights_only=False)
    if tamper == 'missing':
        del saved['input_normalization_signature']
    elif tamper == 'distance':
        saved['config']['env']['observation_distance_scale_km'] *= 2
    elif tamper == 'schema':
        saved['input_normalization_signature']['context_schema'] = 'unknown_schema'
    else:
        saved['input_normalization_signature']['node_context_features'][0] = 'different_feature'
    torch.save(saved, path)
    restored = model(config)
    with pytest.raises(ValueError, match='signature'):
        trainer._load_training_checkpoint(restored, torch.optim.Adam(restored.parameters()), path, 'cpu')


def test_weights_only_whitelist_does_not_allow_missing_shared_model_parameters():
    base = agent()
    upgraded = agent(use_physical_input_context=True)
    state = dict(base.state_dict())
    del state['backbone.dist_bias_scale']
    with pytest.raises(RuntimeError, match='dist_bias_scale'):
        trainer._load_initial_model_state(upgraded, state, strict=False)


def test_context_storage_is_shared_within_episode_but_dynamic_state_is_owned():
    obs = {'node_input_context': np.ones((5, 12), dtype=np.float32),
           'graph_input_context': np.ones(10, dtype=np.float32),
           'current_time': np.zeros(3, dtype=np.float32)}
    cache = {}
    first = snapshot_observation(obs, cache)
    second = snapshot_observation(obs, cache)
    batch_cache = {}
    first_batch = stack_observations([first], batch_cache)
    second_batch = stack_observations([second], batch_cache)
    for key in ('node_input_context', 'graph_input_context'):
        assert first[key] is second[key]
        assert first_batch[key] is second_batch[key]
        assert first[key] is not obs[key]
    assert first['current_time'] is not second['current_time']


def test_teacher_forced_route_cache_preserves_physical_context_probabilities_and_gradients():
    candidates = [_candidate([1, 4, 2, 0, 3, 0]), _candidate([1, 0, 2, 3, 0])]
    for index, candidate in enumerate(candidates):
        node = np.full((5, 12), .2 + index, dtype=np.float32)
        graph = np.full(10, .5 + index, dtype=np.float32)
        for obs in candidate.observations:
            obs['node_input_context'] = node
            obs['graph_input_context'] = graph
    baseline = agent(use_physical_input_context=True, cache_static_observations=True)
    with torch.no_grad():
        baseline.backbone.physical_input_adapter.node_mlp[-1].weight.normal_(std=.02)
        baseline.backbone.physical_input_adapter.graph_mlp[-1].weight.normal_(std=.02)
    cached = copy.deepcopy(baseline)
    old = trainer._expert_route_mean_logprobs(baseline, candidates, 'cpu', 20)
    new = trainer._expert_route_mean_logprobs(cached, candidates, 'cpu', 20, cache_static=True)
    torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)
    old.sum().backward()
    new.sum().backward()
    for name, parameter in baseline.named_parameters():
        if parameter.grad is not None:
            actual = dict(cached.named_parameters())[name].grad
            torch.testing.assert_close(actual, parameter.grad, rtol=1e-4, atol=3e-6, msg=name)


def test_invalid_physical_configuration_fails_before_device_initialization():
    config = cfg(unit=None)
    config['training'] = {}
    with patch.object(trainer.DistributedContext, 'initialize', side_effect=AssertionError('too late')):
        with pytest.raises(ValueError, match='observation_distance_scale_km'):
            trainer.train_from_config(config, seed=1, device='cpu')


def test_frozen_test_evaluator_keeps_context_constructor_arguments():
    from test_cus100_best_eval import evaluator
    from offline2online.models import Agent
    config = cfg()
    config['model']['physical_input_context_hidden_dim'] = 64
    kwargs = evaluator.constructor_kwargs(config, Agent, 'cpu')
    assert kwargs['use_physical_input_context'] is True
    assert kwargs['physical_input_context_hidden_dim'] == 64


def test_checkpoint_cannot_claim_a_different_adapter_hidden_dimension():
    config = cfg()
    actual = model(config)
    changed = copy.deepcopy(config)
    changed['model']['physical_input_context_hidden_dim'] = 64
    with pytest.raises(ValueError, match='hidden dimension'):
        norm.checkpoint_metadata(actual, changed)
