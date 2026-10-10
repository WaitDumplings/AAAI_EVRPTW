"""The reduced recipe retains physical graph decisions with fewer modules.

These are CPU integration checks, using the production model width and real
small routing environments. They verify contracts and gradients, not quality,
Cus100 memory requirements, multi-GPU synchronization, or training throughput.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch

from caliroute.recipes import build_recipe_config
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env_fast import EVRPTWVectorEnvFast
from offline2online.models import Agent
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from test_input_normalization import _instance


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _recipe(tmp_path, problem="vrptw", preset="core"):
    return build_recipe_config(problem=problem, preset=preset, encoder="graph",
        data_root=tmp_path / "AAAI_Dataset", output_dir=tmp_path / "output",
        run_name="CPU_CORE_CONTRACT", seed=3011)


def _agent(cfg):
    # distance_injection is launcher metadata rather than an Agent argument.
    model = dict(cfg["model"])
    assert set(model) - set(inspect.signature(Agent).parameters) == {"distance_injection"}
    model.pop("distance_injection")
    return Agent(**model, device="cpu", use_decomposed_critic=cfg["critic"]["use_decomposed_critic"])


def _env(cfg):
    instance = _instance()
    if cfg["data"]["problem_type"] == "vrptw":
        instance.vehicle.update(consumption_kwh_per_km=0., full_charge_time_s=0.)
        instance.metadata["charging_constraint"] = False
        instance = replace(instance,
            charging_stations=np.empty((0, 2), dtype=np.float32),
            distance_matrix_km=instance.distance_matrix_km[:5, :5],
            cs_time_to_depot_s=np.empty(0, dtype=np.float32))
    # Explicit T/E deliberately disagree with a scalar speed/consumption rule.
    # The model and environment must preserve these authoritative edge costs.
    travel = instance.distance_matrix_km * 2.
    travel[1, 2] += 3.
    energy = instance.distance_matrix_km * .4
    energy[2, 1] += .7
    if cfg["data"]["problem_type"] == "vrptw":
        energy.fill(0.)
    instance = replace(instance, travel_time_matrix_s=travel, energy_matrix_kwh=energy)
    options = dict(cfg["env"])
    assert options.pop("use_fast_env") is True
    options["use_jit_mask"] = False  # CPU test avoids compiling an optional kernel.
    return EVRPTWVectorEnvFast(instance=instance, n_traj=2, **options)


def _count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _loss(output):
    return -output[1].mean() + output[3].square().mean() - .01 * output[2].mean()


def _assert_observation_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
def test_core_removes_modules_and_parameters_but_keeps_physical_graph_decoder(tmp_path, problem):
    core = _agent(_recipe(tmp_path, problem, "core"))
    reference = _agent(_recipe(tmp_path, problem, "reference"))
    small, large = core.backbone, reference.backbone
    assert small.rdi_adapter is None and large.rdi_adapter is not None
    assert small.static_fusion is None and large.static_fusion is not None
    dde, reference_dde = small.decoder.dynamic_graph_kv_encoder, large.decoder.dynamic_graph_kv_encoder
    assert dde.agda_adapter is None and reference_dde.agda_adapter is not None
    assert dde.enabled and dde.agda_physical_candidate_features and dde.agda_smooth_distance_features
    assert small.physical_input_adapter is not None
    assert small.joint_graph_encoder is not None and small.encoder is None
    assert small.edge_relation_encoder is not None and small.use_encoder_distance_bias
    edge_only, full = small.decoder.resource_decoder, large.decoder.resource_decoder
    assert not edge_only.use_resources and edge_only.use_edge_relations
    assert edge_only.observation_mode == "feasible"
    assert full.use_resources and full.observation_mode == "dual"
    assert set(dict(edge_only.named_children())) == {"edge_action_key", "edge_action_bias"}
    assert edge_only.edge_action_key.in_features == 32
    assert edge_only.precompute(torch.zeros(1, 5, 256)) is None
    removed = (_count(large.rdi_adapter) + _count(large.static_fusion)
               + _count(reference_dde.agda_adapter) + _count(full) - _count(edge_only))
    assert removed > 0
    assert _count(large) - _count(small) == removed
    assert _count(reference) - _count(core) == removed
    # Preserve the learned edge interface; use_resource_decoder=False does not
    # mean the entire ResourceDecisionAdapter class disappears.
    keys = core.state_dict()
    assert not any("static_fusion." in key or "rdi_adapter." in key or "agda_adapter." in key for key in keys)
    assert "backbone.decoder.resource_decoder.edge_action_bias.weight" in keys


def test_optional_removals_preserve_shared_initial_weights_and_rng(tmp_path):
    torch.manual_seed(611)
    reference = _agent(_recipe(tmp_path, preset="reference"))
    expected_rng = torch.rand(4)
    torch.manual_seed(611)
    core = _agent(_recipe(tmp_path, preset="core"))
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
    reference_state = reference.state_dict()
    for name, value in core.state_dict().items():
        torch.testing.assert_close(value, reference_state[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
def test_core_cached_and_fresh_replay_match_all_gradients(tmp_path, problem):
    torch.manual_seed(712)
    cfg = _recipe(tmp_path, problem)
    fresh = _agent(cfg)
    cached = deepcopy(fresh)
    obs, _ = _env(cfg).reset(seed=9)
    action = torch.ones(1, 2, dtype=torch.long)
    expected = fresh.get_action_and_value(obs, action=action)
    cache = cached.backbone.encode(obs)
    nodes = obs["edge_distance"].shape[-1]
    assert cache[5]["edge_relations"].shape == (1, nodes, nodes, 32)
    assert cache[5]["resource_readout"] is None
    with patch.object(cached.backbone.joint_graph_encoder, "forward", side_effect=AssertionError("reencoded")):
        actual = cached.get_action_and_value_cached(obs, action=action, cached_embeddings=cache)
    for expected_value, actual_value in zip(expected, actual[:4]):
        torch.testing.assert_close(expected_value, actual_value, rtol=0, atol=0)
    for result in (expected, actual):
        loss = _loss(result)
        assert torch.isfinite(loss)
        loss.backward()
    actual_parameters = dict(cached.named_parameters())
    for name, parameter in fresh.named_parameters():
        other = actual_parameters[name]
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
    for parameter in (fresh.backbone.joint_graph_encoder.layers[0].edge_value,
                      fresh.backbone.decoder.resource_decoder.edge_action_bias.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
def test_evolved_directed_edge_row_reaches_logits_without_resource_branch(tmp_path, problem):
    torch.manual_seed(713)
    cfg = _recipe(tmp_path, problem)
    agent = _agent(cfg)
    env = _env(cfg)
    env.reset(seed=9)
    obs, _, _, _, _ = env.step([1, 1])
    adapter = agent.backbone.decoder.resource_decoder
    # The new readout starts at zero. Activate only one output coefficient so
    # this check proves a usable path after learning, not merely its presence.
    with torch.no_grad():
        adapter.edge_action_bias.weight[0, 0] = .25
    cache = agent.backbone.encode(obs)
    learned_edges = cache[5]["edge_relations"]
    learned_edges.retain_grad()
    with patch.object(adapter, "forward", wraps=adapter.forward) as call:
        logits, _ = agent.backbone.decode(obs, cache)
    assert call.call_args.kwargs["edge_relations"] is learned_edges
    assert not adapter.use_resources and adapter.use_edge_relations
    assert obs["action_mask"][:, 2].all()
    assert not torch.allclose(learned_edges[0, 1, 2], learned_edges[0, 2, 1])

    def edge_change(source, destination):
        changed = learned_edges.clone()
        changed[0, source, destination, 0] += 4.
        altered = (*cache[:5], {**cache[5], "edge_relations": changed})
        return agent.backbone.decode(obs, altered)[0]

    feasible = torch.from_numpy(obs["action_mask"])[None].bool()
    changed = edge_change(1, 2)
    expected = logits.clone()
    expected[..., 2] += 1.
    torch.testing.assert_close(changed[feasible], expected[feasible], rtol=1e-6, atol=1e-6)
    # Reverse edge is not the outgoing edge of this decision state.
    reversed_change = edge_change(2, 1)
    torch.testing.assert_close(reversed_change[feasible], logits[feasible], rtol=0, atol=0)
    (-logits.log_softmax(-1)[..., 2].mean()).backward()
    assert learned_edges.grad is not None and torch.isfinite(learned_edges.grad).all()
    assert learned_edges.grad[0, 1, 2].abs().sum() > 0
    assert torch.isneginf(changed[~feasible]).all()


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
def test_core_and_reference_preserve_masks_edge_units_and_episode_physics(tmp_path, problem):
    cfgs = [_recipe(tmp_path, problem, preset) for preset in ("core", "reference")]
    assert cfgs[0]["env"] == cfgs[1]["env"]
    assert cfgs[0]["experiment_protocol"]["input_normalization_signature"] == cfgs[1]["experiment_protocol"]["input_normalization_signature"]
    models, envs = [_agent(cfg) for cfg in cfgs], [_env(cfg) for cfg in cfgs]
    observations = [env.reset(seed=19)[0] for env in envs]
    sequence = (1, 5, 2, 0, 3, 4, 0) if problem == "evrptw" else (1, 2, 0, 3, 4, 0)
    for destination in sequence:
        _assert_observation_equal(*observations)
        for model, env, obs in zip(models, envs, observations):
            before = deepcopy(obs)
            with torch.no_grad(), patch.object(model.backbone, "_build_distance_matrix", side_effect=AssertionError("Euclidean fallback")):
                logits, _ = model.backbone(obs)
            _assert_observation_equal(obs, before)
            feasible = StateWrapper(obs, "cpu").states["action_mask"].bool()
            assert torch.isneginf(logits[~feasible]).all()
            assert torch.isfinite(logits[feasible]).all()
            assert (logits.softmax(-1)[~feasible] == 0).all()
            np.testing.assert_allclose(obs["edge_distance"] * env.observation_distance_scale_km, env.distance_km, rtol=1e-6)
            np.testing.assert_allclose(obs["edge_time"] * env.horizon_s, env.instance.travel_time_matrix_s, rtol=1e-6)
            np.testing.assert_allclose(obs["edge_energy"] * env.battery_capacity_kwh, env.instance.energy_matrix_kwh, rtol=1e-6)
            assert env.travel_time_source == "provided_travel_time_matrix_s"
            assert obs["action_mask"][:, destination].all()
        transitions = [env.step([destination, destination]) for env in envs]
        for position in (1, 2, 3):
            np.testing.assert_array_equal(transitions[0][position], transitions[1][position])
        for attribute in ("current_time_s", "load_cm3", "battery_used_kwh", "objective_distance_km"):
            np.testing.assert_array_equal(getattr(envs[0], attribute), getattr(envs[1], attribute))
        observations = [step[0] for step in transitions]
    assert transitions[0][2].all() and not transitions[0][3].any()


@pytest.mark.parametrize("missing", ["edge_distance", "edge_time", "edge_energy"])
def test_core_requires_all_authoritative_edge_matrices(tmp_path, missing):
    cfg = _recipe(tmp_path)
    agent = _agent(cfg)
    obs, _ = _env(cfg).reset()
    del obs[missing]
    with pytest.raises(KeyError, match=missing):
        agent.backbone.encode(obs)
