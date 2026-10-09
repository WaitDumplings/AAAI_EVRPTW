"""P0/P1 on real routing observations, cached PPO replay and checkpoint contracts."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import collect_rollout, stack_observations
from offline2online.instance_adapter import adapt_instance_payload
from offline2online.models import Agent
from offline2online import input_normalization, model_integration, trainer
from test_joint_graph_training import _payload
from test_resource_isolation import corrupt_inactive


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def env(task):
    instance = adapt_instance_payload(_payload(f'p0p1_{task}', task), problem_type=task,
                                       strict_road_metric=True)
    return make_terran_env(instance=instance, n_traj=3, use_jit_mask=False,
        observation_coordinate_mode='depot_fixed', observation_distance_scale_km=10.,
        observation_input_context=True, prefer_explicit_edge_matrices=True, info_level='full')


def observation(task):
    environment = env(task)
    raw, _ = environment.reset(seed=13)
    obs = {key: torch.as_tensor(value).clone() for key, value in stack_observations([raw]).items()}
    return obs, environment


def options(**overrides):
    return dict(embedding_dim=32, n_encode_layers=2, use_dynamic_decision_encoder=True,
        use_physical_input_context=True, optimize_dynamic_projections=True,
        cache_static_observations=True, use_resource_isolation=True,
        use_directed_road_profile=True, use_directed_score_mixer=True,
        directed_profile_hidden_dim=32, directed_score_hidden=8, **overrides)


def agent(**changes):
    config = options()
    config.update(changes)
    return Agent(**config)


def nonzero_adapters(model):
    # Exercise learned pathways, not just residual-head zero initialization.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(token in name for token in ('directed_road_profile.', 'directed_score_mixer.',
                                                'physical_input_adapter.')):
                parameter.add_(torch.randn_like(parameter) * .025)


def loss(output):
    return -output[1].mean() + output[3].square().mean() - .01 * output[2].mean()


@pytest.mark.parametrize('task', ['cvrp', 'vrptw', 'evrptw'])
def test_real_environment_scratch_policy_cached_replay_and_gradients(task):
    torch.manual_seed(861)
    fresh = agent()
    nonzero_adapters(fresh)
    cached = deepcopy(fresh)
    obs, environment = observation(task)
    assert bool(obs['graph_input_context'][0, 6]) == (task == 'evrptw')
    assert bool(obs['graph_input_context'][0, 8]) == (task != 'cvrp')
    cache = cached.backbone.encode(obs)
    nxt, *_ = environment.step(np.ones(3, dtype=np.int64))
    nxt = {key: torch.as_tensor(value).clone() for key, value in stack_observations([nxt]).items()}
    action = torch.full((1, 3), 2, dtype=torch.long)
    expected = fresh.get_action_and_value(nxt, action=action)
    with patch.object(cached.backbone.encoder, 'forward', side_effect=AssertionError('reencoded')):
        actual = cached.get_action_and_value_cached(nxt, action=action, cached_embeddings=cache)
    for left, right in zip(expected, actual[:4]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    loss(expected).backward()
    loss(actual).backward()
    for (name, first), (_, second) in zip(fresh.named_parameters(), cached.named_parameters()):
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0, msg=name)
            assert torch.isfinite(first.grad).all(), name
    for suffix in ('backbone.directed_road_profile.road.0.weight',
                   'backbone.encoder.layers.0.attn.MHA.directed_score_mixer.net.0.weight'):
        assert dict(fresh.named_parameters())[suffix].grad.abs().sum() > 0, suffix


@pytest.mark.parametrize('task', ['cvrp', 'vrptw'])
def test_p0p1_model_policy_value_and_gradients_ignore_nan_inactive_dummy_fields(task):
    torch.manual_seed(722)
    model = agent()
    nonzero_adapters(model)
    obs, _ = observation(task)
    before = deepcopy(obs)
    action = torch.ones(1, 3, dtype=torch.long)
    expected = model.get_action_and_value(obs, action=action)
    for nonfinite in (False, True):
        altered = corrupt_inactive(obs, task, nonfinite)
        output = model.get_action_and_value(altered, action=action)
        for left, right in zip(expected, output):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        cache = model.backbone.encode(altered)
        replay = model.get_action_and_value_cached(altered, action=action, cached_embeddings=cache)
        for left, right in zip(expected, replay[:4]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        loss(output).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    for key in obs:
        torch.testing.assert_close(obs[key], before[key], atol=0, rtol=0)


def test_batched_mixed_resource_tasks_with_cpu_amp_and_zero_inactive_gradient():
    model = agent()
    nonzero_adapters(model)
    cv, _ = observation('cvrp')
    vr, _ = observation('vrptw')
    mixed = {key: torch.cat((cv[key], vr[key])) for key in cv}
    mixed['edge_energy'].fill_(float('nan'))
    mixed['edge_energy'].requires_grad_()
    mixed['edge_time'][0].fill_(float('nan'))
    mixed['edge_time'].requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model.get_action_and_value(mixed, action=torch.ones(2, 3, dtype=torch.long))
        objective = loss(output)
    objective.backward()
    assert torch.isfinite(objective)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert mixed['edge_energy'].grad is None or not mixed['edge_energy'].grad.any()
    assert mixed['edge_time'].grad is not None
    assert not mixed['edge_time'].grad[0].any()
    assert torch.isfinite(mixed['edge_time'].grad).all()


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_complete_real_rollout_uses_one_static_encoding_and_matches_reencoding(task):
    model = agent()
    nonzero_adapters(model)
    with patch.object(model.backbone.encoder, 'forward', wraps=model.backbone.encoder.forward) as calls:
        torch.manual_seed(851)
        cached = collect_rollout(model, [env(task), env(task)], 16, 'sample', 'cpu', seed=7)
        assert calls.call_count == 1
    model.backbone.supports_static_rollout_cache = False
    torch.manual_seed(851)
    fresh = collect_rollout(model, [env(task), env(task)], 16, 'sample', 'cpu', seed=7)
    for name in ('actions', 'old_logprobs', 'rewards', 'dones', 'values', 'valid', 'entropies'):
        torch.testing.assert_close(getattr(cached, name), getattr(fresh, name), atol=0, rtol=0, msg=name)
    assert torch.isfinite(cached.old_logprobs).all() and torch.isfinite(cached.values).all()
    assert cached.valid.any()
    # PPO replay starts with ratio exactly 1 before the optimizer moves weights.
    action = cached.actions[0]
    _, replay_logprob, _, _ = model.get_action_and_value(cached.observations[0], action=action)
    torch.testing.assert_close(replay_logprob, cached.old_logprobs[0], atol=0, rtol=0)


def test_full_agent_shared_parameters_and_rng_preserved_for_original_and_p1():
    torch.manual_seed(958)
    original = agent(use_resource_isolation=False, use_directed_road_profile=False,
                     use_directed_score_mixer=False, use_physical_input_context=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(958)
    upgraded = agent()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, parameter in original.state_dict().items():
        assert torch.equal(upgraded.state_dict()[name], parameter), name
    assert not any('directed_' in name or 'resource_isolation' in name for name in original.state_dict())
    # With all resources active, zero-head P1 leaves the original policy intact.
    obs, _ = observation('evrptw')
    action = torch.ones(1, 3, dtype=torch.long)
    for old, new in zip(original.get_action_and_value(obs, action=action),
                        upgraded.get_action_and_value(obs, action=action)):
        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-6)


def config(**overrides):
    model = options()
    model.update(overrides)
    return dict(model=model, env=dict(observation_coordinate_mode='depot_fixed',
        observation_input_context=True, observation_distance_scale_km=10.), offline=dict(method='ppo'))


def configured(cfg):
    model = Agent(**cfg['model'])
    input_normalization.configure(model, cfg)
    model_integration.configure(model, cfg)
    return model


def test_ppo_warmup_checkpoint_roundtrip_into_same_architecture_sl_stage(tmp_path):
    cfg = config()
    source = configured(cfg)
    nonzero_adapters(source)
    path = tmp_path / 'warmup_epoch_0100.pt'
    trainer.save_checkpoint(path, source, torch.optim.Adam(source.parameters(), lr=1e-4), cfg, 100, 33)
    stored = torch.load(path, weights_only=False)
    profile = stored['model_integration_signature']
    assert all(profile[key] for key in ('use_resource_isolation', 'use_directed_road_profile', 'use_directed_score_mixer'))
    assert profile['directed_profile_hidden_dim'] == 32 and profile['directed_score_hidden'] == 8
    main_cfg = deepcopy(cfg)
    main_cfg['offline']['method'] = 'sl_ppo'
    target = configured(main_cfg)
    result = trainer._load_agent_checkpoint(target, path, 'cpu', strict=True)
    assert result['model_integration']['load'] == 'weights_only'
    assert not result['missing_keys'] and not result['unexpected_keys']
    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name]), name
    # Full resume is supported too; optimization state must restore correctly.
    optimizer = torch.optim.Adam(target.parameters(), lr=.02)
    trainer._load_training_checkpoint(target, optimizer, path, 'cpu', strict=True)
    assert optimizer.param_groups[0]['lr'] == 1e-4


@pytest.mark.parametrize('change', [{'use_directed_score_mixer': False},
                                  {'use_directed_road_profile': False},
                                  {'use_resource_isolation': False},
                                  {'directed_score_hidden': 12}])
@pytest.mark.parametrize('resume', [False, True])
def test_p0p1_checkpoint_rejects_cross_profile_loading_before_mutation(tmp_path, change, resume):
    cfg = config()
    source = configured(cfg)
    path = tmp_path / 'source.pt'
    trainer.save_checkpoint(path, source, torch.optim.Adam(source.parameters()), cfg, 100, 33)
    target_cfg = config(**change)
    target = configured(target_cfg)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='P0/P1 architecture or resource semantics changed'):
        if resume:
            trainer._load_training_checkpoint(target, torch.optim.Adam(target.parameters()), path, 'cpu', strict=False)
        else:
            trainer._load_agent_checkpoint(target, path, 'cpu', strict=False)
    for name, value in before.items():
        assert torch.equal(value, target.state_dict()[name]), name


def test_p0p1_signature_validates_flags_and_missing_metadata():
    for key in ('use_resource_isolation', 'use_directed_road_profile', 'use_directed_score_mixer'):
        with pytest.raises(ValueError, match='boolean'):
            model_integration.signature(config(**{key: 'yes'}))
    cfg = config()
    with pytest.raises(ValueError, match='missing'):
        model_integration.checkpoint_profile({'config': cfg})
    broken = dict(config=cfg, model_integration_signature=model_integration.signature(cfg))
    broken['model_integration_signature']['directed_score_hidden'] = 9
    with pytest.raises(ValueError, match='does not match'):
        model_integration.checkpoint_profile(broken)
