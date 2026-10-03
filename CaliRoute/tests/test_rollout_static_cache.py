from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from evrptw_core.schema import EVRPTWInstance
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout,
    encode_static_rollout,
    rollout_eval_batch,
    rollout_single_instance,
    stack_observations,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.attention_model_wrapper import Agent as LegacyAgent
from offline2online.models import Agent
from offline2online.trainer import _rollout_eval_batch_min_median


@pytest.fixture(autouse=True)
def _small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _env(scale=1.0):
    coordinates = np.array([[0, 0], [1, 0], [0, 2], [2, 2], [1, 1]], dtype=np.float32)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1) * scale
    instance = EVRPTWInstance.from_dict({
        "instance_id": f"synthetic_{scale}", "working_start_s": 0, "working_end_s": 3600,
        "depot": coordinates[0], "customers": coordinates[1:4], "charging_stations": coordinates[4:],
        "distance_matrix_km": distance, "demands_cm3": [1, 1, 1], "package_counts": [1, 1, 1],
        "service_time_s": [10, 20, 30], "tw_s": [[0, 3600]] * 3, "cs_time_to_depot_s": [60],
        "vehicle": {"cargo_capacity_cm3": 2, "battery_capacity_kwh": 5,
                    "consumption_kwh_per_km": 0.4, "full_charge_time_s": 60},
        "speed_profile": {"effective_speed_kmh": 60},
    })
    return make_terran_env(instance=instance, n_traj=3, use_jit_mask=False, info_level="full")


def _agent(dynamic=True):
    torch.manual_seed(21)
    return Agent(embedding_dim=32, n_encode_layers=1, use_dynamic_decision_encoder=dynamic)


def _run(agent, decode="sample"):
    torch.manual_seed(91)
    return collect_rollout(agent, [_env(), _env(1.3)], 24, decode, "cpu", seed=7)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_fixed_actions_keep_static_inputs_logits_and_values_equal(dynamic, training):
    agent = _agent(dynamic).train(training)
    env = _env()
    observation, _ = env.reset(seed=1)
    first = stack_observations([observation])
    cached = encode_static_rollout(agent, first)
    assert all(not tensor.requires_grad for tensor in cached)
    static_keys = ("depot_loc", "cus_loc", "rs_loc", "demand", "service_time", "time_window",
                   "edge_distance", "edge_time", "edge_energy", "battery_capacity", "loading_capacity")
    # Includes customer visits, charging, depot reset, and a second route.
    for destination in (1, 4, 2, 0, 3, 0):
        obs = stack_observations([observation])
        for key in static_keys:
            np.testing.assert_array_equal(obs[key], first[key])
        with torch.no_grad():
            fresh = agent.backbone(obs)
            reused = agent.backbone.decode(obs, cached)
            torch.testing.assert_close(reused[0], fresh[0], rtol=0, atol=0)
            torch.testing.assert_close(agent.critic(reused), agent.critic(fresh), rtol=0, atol=0)
        assert observation["action_mask"][:, destination].all()
        observation, _, _, _, _ = env.step(np.full(3, destination))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("decode", ["greedy", "sample"])
def test_complete_rollout_matches_original_and_encodes_once(dynamic, decode):
    agent = _agent(dynamic)
    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        cached = _run(agent, decode)
        assert encoder.call_count == 1
    agent.backbone.supports_static_rollout_cache = False
    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        original = _run(agent, decode)
        assert encoder.call_count == len(original.observations) > 1
    for key in ("actions", "old_logprobs", "rewards", "dones", "route_boundaries", "values", "valid", "entropies"):
        torch.testing.assert_close(getattr(cached, key), getattr(original, key), rtol=0, atol=0)
    for got, expected in zip(cached.final_infos, original.final_infos):
        for key in ("success", "objective_distance_km", "vehicle_count", "routes"):
            assert np.array_equal(got[key], expected[key]) if key != "routes" else got[key] == expected[key]


def test_cache_lifetime_ends_at_rollout_and_training_still_reencodes_with_gradients():
    agent = _agent()
    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        batch = _run(agent)
        _run(agent)
        assert encoder.call_count == 2
        _, logprob, _, value = agent.get_action_and_value(batch.observations[0], action=batch.actions[0])
        (logprob.mean() + value.mean()).backward()
        assert encoder.call_count == 3
    grads = [parameter.grad for parameter in agent.backbone.encoder.parameters()]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_legacy_dropout_backbone_retains_uncached_rollout():
    torch.manual_seed(8)
    agent = LegacyAgent(embedding_dim=32, n_encode_layers=1).train()
    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        batch = _run(agent)
        assert encoder.call_count == len(batch.observations) > 1


@pytest.mark.parametrize("runner", ["single", "batch", "min_median"])
def test_evaluation_paths_encode_once_and_match_uncached_results(runner):
    agent = _agent().eval()

    def run():
        torch.manual_seed(42)
        options = dict(decode_mode="sample", max_steps=24, device="cpu", seed=9)
        if runner == "single":
            return [rollout_single_instance(agent, _env(), **options)]
        function = rollout_eval_batch if runner == "batch" else _rollout_eval_batch_min_median
        return function(agent, [_env(), _env(1.3)], **options)

    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        cached = run()
        assert encoder.call_count == 1
    agent.backbone.supports_static_rollout_cache = False
    with patch.object(agent.backbone.encoder, "forward", wraps=agent.backbone.encoder.forward) as encoder:
        original = run()
        assert encoder.call_count > 1
    for got, expected in zip(cached, original):
        for key in got:
            if "runtime" not in key:
                assert got[key] == expected[key]
