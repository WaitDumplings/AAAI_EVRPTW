"""Physical decision predictions are checked against both actual environments."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from caliroute.plugins.physical_decision import candidate_transitions, ResourceDecisionAdapter, _safe_attention
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.decoder import Decoder
from test_input_normalization import _env, _instance


def make_env(fast=False, kind='evrptw', charging_mode='fixed_full'):
    instance = _instance()
    if kind != 'evrptw':
        instance.vehicle['consumption_kwh_per_km'] = 0.
        instance.vehicle['full_charge_time_s'] = 0.
        instance.metadata['charging_constraint'] = False
        instance = replace(instance, charging_stations=np.empty((0, 2), dtype=np.float32),
                           distance_matrix_km=instance.distance_matrix_km[:5, :5],
                           cs_time_to_depot_s=np.empty(0, dtype=np.float32))
    if kind == 'cvrp':
        instance.metadata['time_window_constraint'] = False
        instance.tw_s[:] = [1000, 3000]
        instance.service_time_s[:] = 0.
    return _env(fast, instance=instance, observation_coordinate_mode='depot_fixed',
                observation_input_context=True, observation_distance_scale_km=5., charging_mode=charging_mode)


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('kind', ['cvrp', 'vrptw', 'evrptw'])
@pytest.mark.parametrize('charging_mode', ['fixed_full', 'proportional_full'])
def test_typed_predictions_match_actual_environment_multiroute(fast, kind, charging_mode):
    env = make_env(fast, kind, charging_mode)
    obs, _ = env.reset()
    emb = torch.randn(1, env.num_nodes, 32)
    sequence = [1, 5, 2, 0, 3, 4, 0] if kind == 'evrptw' else [1, 2, 0, 3, 4, 0]
    for k, destination in enumerate(sequence):
        state = StateWrapper(obs, 'cpu')
        features = candidate_transitions(state, emb)
        assert obs['action_mask'][:, destination].all()
        expected_time = features['next_time'][0, :, destination].numpy()
        expected_load = features['load_after'][0, :, destination].numpy()
        expected_battery = features['battery_after'][0, :, destination].numpy()
        if destination == 5:
            assert features['charge_time'][0, 0, 5] >= 0
            assert features['battery_after'][0, 0, 5] == 0
        if destination == 0:
            assert features['terminal'][0, 0, 0] == (k == len(sequence) - 1)
            assert features['next_route'][0, 0, 0] == (k != len(sequence) - 1)
        obs, _, terminated, truncated, _ = env.step([destination] * 2)
        assert not truncated.any()
        np.testing.assert_allclose(obs['current_time'], expected_time, atol=2e-7)
        np.testing.assert_allclose(obs['current_load'], expected_load, atol=2e-7)
        np.testing.assert_allclose(obs['current_battery'], expected_battery, atol=2e-7)
    assert terminated.all()


def test_customer_due_constrains_start_not_departure_and_cs_ignores_customer_windows():
    obs, _ = make_env().reset()
    obs['time_window'][1] = [0.2, 0.21]
    obs['service_time'][1] = 0.05
    obs['time_window'][5] = [0.8, 0.81]
    feat = candidate_transitions(StateWrapper(obs, 'cpu'), torch.randn(1, 6, 32))
    torch.testing.assert_close(feat['service_start'][..., 1], torch.full((1, 2), .2))
    torch.testing.assert_close(feat['start_due_slack'][..., 1], torch.full((1, 2), .01))
    assert (feat['departure'][..., 1] > .21).all()
    torch.testing.assert_close(feat['service_start'][..., 5], feat['arrival'][..., 5])


def test_future_observation_separates_feasibility_visitation_and_cs_route_reuse():
    obs, _ = make_env().reset()
    obs['action_mask'][:, 2] = False  # capacity/time may block now, still useful to observe
    obs['customer_visited'][:, 1] = True
    obs['cs_visited_current_route'][:, 5] = True
    obs['instance_mask'] = np.array([[False, False, False, False, True, False]])
    feat = candidate_transitions(StateWrapper(obs, 'cpu'), torch.randn(1, 6, 32))
    assert feat['future_mask'][..., 2].all()
    assert not feat['future_mask'][..., 1].any()
    assert not feat['future_mask'][..., 4].any()
    assert not feat['future_mask'][..., 5].any()
    assert feat['future_mask'][..., 0].all()


def decoder(enabled=False, edges=False):
    return Decoder(32, 32, 4, None, 10, use_dynamic_decision_encoder=True,
                   dynamic_decision_heads=4, dynamic_decision_delta_k=False,
                   dynamic_decision_delta_v=False, use_resource_decoder=enabled,
                   decoder_observation_mode='dual' if enabled else 'feasible',
                   use_edge_relation_encoder=edges)


def test_zero_initial_decoder_exact_equivalence_rng_hardmask_and_gradients():
    obs, _ = make_env().reset()
    state = StateWrapper(obs, 'cpu')
    torch.manual_seed(981)
    old = decoder()
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(981)
    new = decoder(True)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    embeddings = torch.randn(1, 7, 32)
    old_out, old_glimpse = old.advance(old._precompute(embeddings), state)
    new_out, new_glimpse = new.advance(new._precompute(embeddings), state)
    torch.testing.assert_close(old_out, new_out, rtol=0, atol=0)
    torch.testing.assert_close(old_glimpse, new_glimpse, rtol=0, atol=0)
    mask = ~torch.as_tensor(obs['action_mask'])[None]
    assert torch.equal(new_out.softmax(-1)[mask], torch.zeros(mask.sum()))
    (-new_out.log_softmax(-1)[..., 1].mean()).backward()
    for name in ['action_key_out', 'action_bias_out', 'readout_out']:
        grad = getattr(new.resource_decoder, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_dual_readout_and_glimpse_empty_masks_are_finite_without_changing_action_mask():
    obs, _ = make_env().reset()
    obs['action_mask'][:] = False
    obs['instance_mask'] = np.ones((1, 6), bool)
    state = StateWrapper(obs, 'cpu')
    net = decoder(True)
    embeddings = torch.randn(1, 7, 32)
    logits, glimpse = net.advance(net._precompute(embeddings), state)
    assert torch.isfinite(glimpse).all()
    assert torch.isneginf(logits).all()  # empty action set stays empty
    weights = _safe_attention(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    assert torch.equal(weights, torch.zeros_like(weights))


def test_missing_edge_matrix_is_rejected_and_nonfinite_edge_is_unobservable():
    obs, _ = make_env().reset()
    state = StateWrapper(obs, 'cpu')
    del state.states['edge_time']
    with pytest.raises(KeyError, match='edge_time'):
        candidate_transitions(state, torch.randn(1, 6, 32))
    obs['edge_distance'][0, 2] = np.inf
    feat = candidate_transitions(StateWrapper(obs, 'cpu'), torch.randn(1, 6, 32))
    assert not feat['future_mask'][..., 2].any()


@pytest.mark.parametrize('autocast', [False, True])
def test_nonzero_plugin_is_batch_independent_and_amp_finite(autocast):
    obs, _ = make_env().reset()
    state = StateWrapper({k: np.stack([v, v]) for k, v in obs.items()}, 'cpu')
    node = torch.randn(2, 6, 32)
    query = torch.randn(2, 2, 32)
    module = ResourceDecisionAdapter(32, observation_mode='dual')
    for output in (module.action_key_out, module.action_bias_out, module.readout_out):
        torch.nn.init.normal_(output.weight, std=.03)
    module.diagnostics_enabled = True
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        together = module(node, query, state)
        for b in range(2):
            separate = module(node[b:b+1], query[b:b+1], StateWrapper(obs, 'cpu'))
            for joint, single in zip(together, separate):
                assert torch.isfinite(joint).all()
                torch.testing.assert_close(joint[b:b+1], single, rtol=.015 if autocast else 1e-5, atol=.003 if autocast else 1e-6)
    for value in module.diagnostics().values():
        assert not value.requires_grad and torch.isfinite(value)


def test_edge_row_residual_uses_current_direction_and_is_zero_initialized():
    obs, _ = make_env().reset()
    obs['last_node_idx'][:] = [1, 3]
    state = StateWrapper(obs, 'cpu')
    module = ResourceDecisionAdapter(32, use_resources=False, use_edge_relations=True)
    node, query = torch.randn(1, 6, 32), torch.randn(1, 2, 32)
    edge = torch.randn(1, 6, 6, 16, requires_grad=True)
    initial = module(node, query, state, edge_relations=edge)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in initial)
    torch.nn.init.normal_(module.edge_action_key.weight, std=.1)
    _, key, _ = module(node, query, state, edge_relations=edge)
    expected = module.edge_action_key(edge[:, [1, 3]])
    torch.testing.assert_close(key, expected)
    key.sum().backward()
    assert edge.grad[:, [1, 3]].abs().sum() > 0
    assert edge.grad[:, [0, 2, 4, 5]].abs().sum() == 0


def test_inactive_resource_dummies_cannot_change_learned_decisions():
    obs, _ = make_env(kind='cvrp').reset()
    node, query = torch.randn(1, 5, 32), torch.randn(1, 2, 32)
    adapter = ResourceDecisionAdapter(32, observation_mode='dual')
    for output in (adapter.action_key_out, adapter.action_bias_out, adapter.readout_out):
        torch.nn.init.normal_(output.weight, std=.03)
    original = adapter(node, query, StateWrapper(obs, 'cpu'))
    obs['edge_time'][:] = np.inf
    obs['edge_energy'][:] = np.nan
    obs['current_time'][:] = 1e20
    obs['current_battery'][:] = np.inf
    changed = adapter(node, query, StateWrapper(obs, 'cpu'))
    for left, right in zip(original, changed):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_float32_physics_and_compression_before_half_conversion():
    obs, _ = make_env().reset()
    obs['edge_distance'][:] *= 1e8
    obs['edge_time'][:] *= 1e8
    obs['edge_energy'][:] *= 1e8
    node, query = torch.randn(1, 6, 32).half(), torch.randn(1, 2, 32).half()
    adapter = ResourceDecisionAdapter(32, observation_mode='dual').half()
    for output in (adapter.action_key_out, adapter.action_bias_out, adapter.readout_out):
        torch.nn.init.normal_(output.weight, std=.01)
    features = candidate_transitions(StateWrapper(obs, 'cpu'), node)
    assert features['arrival'].dtype == torch.float32
    assert features['travel_distance'].max() > 65504
    for value in adapter(node, query, StateWrapper(obs, 'cpu')):
        assert torch.isfinite(value).all()


def test_nonfinite_customer_inputs_are_not_future_observations():
    obs, _ = make_env().reset()
    obs['time_window'][1] = [np.nan, np.nan]
    obs['demand'][2] = np.inf
    features = candidate_transitions(StateWrapper(obs, 'cpu'), torch.randn(1, 6, 32))
    assert not features['future_mask'][..., 1:3].any()


def test_stateless_replay_slicing_trajectories_preserves_decisions_and_gradients():
    from offline2online.observation_storage import STATIC_OBSERVATION_KEYS
    obs, _ = make_env().reset()
    obs['current_time'][:] = [.1, .3]
    obs['last_node_idx'][:] = [1, 3]
    obs['action_mask'][0, 2] = False
    node, query = torch.randn(1, 6, 32), torch.randn(1, 2, 32)
    adapter = ResourceDecisionAdapter(32, observation_mode='dual')
    for output in (adapter.action_key_out, adapter.action_bias_out, adapter.readout_out):
        torch.nn.init.normal_(output.weight, std=.03)
    together = adapter(node, query, StateWrapper(obs, 'cpu'))
    total = sum(value.square().sum() for value in together)
    total.backward()
    full_grads = {name: parameter.grad.clone() for name, parameter in adapter.named_parameters()}
    adapter.zero_grad()
    separate_loss = 0
    for t in range(2):
        single_obs = {key: value if key in STATIC_OBSERVATION_KEYS else value[t:t+1]
                      for key, value in obs.items()}
        one = adapter(node, query[:, t:t+1], StateWrapper(single_obs, 'cpu'))
        for joint, single in zip(together, one):
            torch.testing.assert_close(joint[:, t:t+1], single, rtol=1e-5, atol=1e-6)
        separate_loss = separate_loss + sum(value.square().sum() for value in one)
    separate_loss.backward()
    for name, parameter in adapter.named_parameters():
        torch.testing.assert_close(full_grads[name], parameter.grad, rtol=1e-4, atol=2e-6)
