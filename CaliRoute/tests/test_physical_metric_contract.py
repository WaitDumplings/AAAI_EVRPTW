"""Physical costs and model-state semantics cannot silently change on transfer."""
import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest
import torch

from evrptw_core.physical import resolve_physical_edge_matrices
from evrptw_core.schema import EVRPTWInstance
from evrptw_core.validation import validate_instance_structure
from offline2online.instance_adapter import adapt_instance_payload, iter_adapted_instances, CLASSICAL_BUNDLE_FORMAT
from offline2online import input_normalization, model_integration
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.decoder import DynamicGraphKVEncoder
from test_input_normalization import _instance, _env
from test_instance_adapter import _base_payload
from test_model_design_optimizations import agent


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _vrptw_payload():
    payload = _base_payload('VRPTW')
    payload.update(working_start_s=0, working_end_s=1000,
                   tw_s=np.tile([0., 1000.], (3, 1)), service_time_s=np.zeros(3),
                   speed_profile={'effective_speed_kmh': 3600.})
    return payload


def test_strict_road_mode_rejects_fallback_but_legacy_accepts_it(tmp_path):
    payload = _base_payload()
    del payload['distance_matrix_km']
    assert adapt_instance_payload(payload, problem_type='cvrp').distance_matrix_km.shape == (4, 4)
    with pytest.raises(ValueError, match='Euclidean fallback'):
        adapt_instance_payload(payload, problem_type='cvrp', strict_road_metric=True)
    path = tmp_path/'instances.pkl'
    with path.open('wb') as stream:
        pickle.dump({'format': CLASSICAL_BUNDLE_FORMAT, 'num_instances': 1}, stream)
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match='Euclidean fallback'):
        list(iter_adapted_instances(path, problem_type='cvrp', strict_road_metric=True))
    payload = _base_payload()
    payload['metadata'] = {'distance_metric': 'euclidean'}
    with pytest.raises(ValueError, match='non-road'):
        adapt_instance_payload(payload, problem_type='cvrp', strict_road_metric=True)


def test_nonuniform_time_is_preserved_instead_of_replaced_by_median_speed():
    payload = _vrptw_payload()
    travel = payload['distance_matrix_km'].copy() * 7
    travel[0, 1] = 73.
    payload['travel_time_matrix_s'] = travel
    instance = adapt_instance_payload(payload, problem_type='vrptw', strict_road_metric=True)
    np.testing.assert_array_equal(instance.travel_time_matrix_s, travel)
    explicit = resolve_physical_edge_matrices(instance, prefer_explicit_edge_matrices=True)
    np.testing.assert_array_equal(explicit['travel_time_s'], travel)
    old = resolve_physical_edge_matrices(instance)
    np.testing.assert_allclose(old['travel_time_s'], payload['distance_matrix_km'])
    assert explicit['travel_time_source'] == 'provided_travel_time_matrix_s'
    assert old['travel_time_source'] == 'distance_over_effective_speed'


def test_schema_retains_energy_and_invalid_matrix_cannot_pass_validation():
    payload = copy.deepcopy(_instance().raw)
    energy = np.asarray(payload['distance_matrix_km'], dtype=np.float32) * .7
    energy[0, 1] = 1.75
    payload['energy_matrix_kwh'] = energy
    instance = EVRPTWInstance.from_dict(payload)
    np.testing.assert_array_equal(resolve_physical_edge_matrices(instance, prefer_explicit_edge_matrices=True)['energy_kwh'], energy)
    broken = replace(instance, energy_matrix_kwh=energy[:-1])
    assert not validate_instance_structure(broken).success
    with pytest.raises(ValueError, match='shape'):
        resolve_physical_edge_matrices(broken, prefer_explicit_edge_matrices=True)


def _features(encoder, state, embedding):
    captured = []
    hook = encoder.action_bias_proj.register_forward_pre_hook(lambda module, values: captured.append(values[0].detach()))
    try:
        encoder(embedding, torch.zeros(1, 1, embedding.size(-1)),
                torch.zeros(1, state.get_current_node().size(1), embedding.size(-1)), state)
    finally:
        hook.remove()
    assert len(captured) == 1
    return captured[0]


def _encoder(**options):
    return DynamicGraphKVEncoder(embedding_dim=16, enabled=True, enable_delta_k=False,
                                 enable_delta_v=False, enable_delta_action_key=False,
                                 optimize_dynamic_projections=True, **options)


def test_agda_physical_channels_match_actual_depot_and_charge_post_states():
    env = _env(observation_coordinate_mode='depot_fixed', observation_distance_scale_km=5.)
    obs, _ = env.reset()
    obs, *_ = env.step([1, 1])
    state = StateWrapper(obs, 'cpu')
    emb = torch.zeros(1, env.num_nodes, 16)
    features = _features(_encoder(agda_physical_candidate_features=True), state, emb)
    assert features.shape == (1, 2, env.num_nodes, 30)
    # Trajectory 0 closes a route; trajectory 1 recharges at a physical station.
    after, _, _, truncated, _ = env.step([0, 5])
    assert not truncated.any()
    for trajectory, node in enumerate([0, 5]):
        assert float(features[0, trajectory, node, 17]) == pytest.approx(float(after['current_load'][trajectory]), abs=2e-7)
        assert float(features[0, trajectory, node, 20]) == pytest.approx(float(after['current_time'][trajectory]), abs=2e-7)
        assert float(features[0, trajectory, node, 24]) == pytest.approx(float(after['current_battery'][trajectory]), abs=2e-7)
    old = _features(_encoder(), state, emb)
    assert old[0, 0, 0, 17] > 0  # Historical feature was pre-reset load.
    assert old[0, 1, 5, 24] > 0  # Historical feature was pre-charge consumption.


def test_physical_due_slack_tracks_service_start_not_finish():
    env = _env(observation_distance_scale_km=5.)
    obs, _ = env.reset()
    obs['time_window'][1] = [.2, .21]
    obs['service_time'][1] = .05
    feat = _features(_encoder(agda_physical_candidate_features=True), StateWrapper(obs, 'cpu'), torch.zeros(1, 6, 16))
    assert float(feat[0, 0, 1, 22]) == pytest.approx(.01, abs=1e-7)
    assert float(feat[0, 0, 1, 23]) == pytest.approx(.75, abs=1e-7)


def test_distance_features_remain_distinguishable_beyond_training_range():
    env = _env(observation_distance_scale_km=5.)
    obs, _ = env.reset()
    obs['edge_distance'][0, 1:4] = [3., 5., 9.]
    state = StateWrapper(obs, 'cpu')
    emb = torch.zeros(1, env.num_nodes, 16)
    legacy = _features(_encoder(), state, emb)
    smooth = _features(_encoder(agda_smooth_distance_features=True), state, emb)
    assert torch.equal(legacy[0, 0, 1:4, 12], torch.full((3,), 2.))
    assert torch.all(torch.diff(smooth[0, 0, 1:4, 12]) > 0)
    assert torch.isfinite(smooth).all()


def test_disabled_feature_flags_preserve_weights_and_rng():
    torch.manual_seed(417)
    first = agent()
    random = torch.rand(3)
    torch.manual_seed(417)
    explicit = agent(agda_physical_candidate_features=False, agda_smooth_distance_features=False)
    torch.testing.assert_close(torch.rand(3), random, atol=0, rtol=0)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize('flag', ['agda_physical_candidate_features', 'agda_smooth_distance_features'])
def test_feature_semantics_changes_require_weights_only_migration(flag):
    old_config = {'model': {}}
    old = agent()
    model_integration.configure(old, old_config)
    checkpoint = {'config': old_config, **model_integration.checkpoint_metadata(old, old_config)}
    # Simulate a v1 checkpoint written before these parameter-free flags existed.
    for key in model_integration.FEATURE_FLAGS:
        checkpoint['model_integration_signature'].pop(key)
    target_config = {'model': {flag: True}}
    target = agent(**target_config['model'])
    model_integration.configure(target, target_config)
    with pytest.raises(ValueError, match='changed on resume'):
        model_integration.load_checkpoint_profile(target, checkpoint, resume=True)
    assert model_integration.load_checkpoint_profile(target, checkpoint, resume=False)['migrated']
    assert model_integration.checkpoint_profile(checkpoint) == model_integration.signature(old_config)


@pytest.mark.parametrize('section,flag', [('data', 'strict_road_metric'), ('env', 'prefer_explicit_edge_matrices')])
def test_input_contract_tracks_physical_cost_semantics(section, flag):
    old_config = {'model': {}}
    previous = input_normalization.signature(old_config)
    checkpoint = {'config': old_config, 'input_normalization_signature': dict(previous)}
    for key in ('strict_road_metric', 'prefer_explicit_edge_matrices'):
        checkpoint['input_normalization_signature'].pop(key)
    assert input_normalization.checkpoint_profile(checkpoint) == previous
    changed = {section: {flag: True}, 'model': {}}
    target = agent()
    input_normalization.configure(target, changed)
    with pytest.raises(ValueError, match='changed on resume'):
        input_normalization.load_checkpoint_profile(target, checkpoint, resume=True)


@pytest.mark.parametrize('problem', ['cvrp', 'vrptw', 'evrptw'])
@pytest.mark.parametrize('n', [15, 50, 100])
def test_physical_contract_supports_task_and_customer_size_transfer(problem, n):
    count = n + 1 + (1 if problem == 'evrptw' else 0)
    idx = np.arange(count)
    distance = (np.abs(idx[:, None] - idx[None, :]) + 1.).astype(np.float32)
    np.fill_diagonal(distance, 0.)
    payload = dict(instance_id='sizes', problem_class=problem, working_start_s=0,
                   working_end_s=1000, depot=np.zeros(2), customers=np.column_stack([np.arange(n), np.ones(n)]),
                   charging_stations=np.zeros((1 if problem == 'evrptw' else 0, 2)),
                   distance_matrix_km=distance, travel_time_matrix_s=distance*2,
                   energy_matrix_kwh=distance*.002, demands_cm3=np.ones(n),
                   package_counts=np.ones(n), service_time_s=np.zeros(n), tw_s=np.tile([0, 1000], (n, 1)),
                   cs_time_to_depot_s=np.zeros(1 if problem == 'evrptw' else 0),
                   vehicle={'cargo_capacity_cm3': 1000., 'battery_capacity_kwh': 100., 'consumption_kwh_per_km': .4},
                   speed_profile={'effective_speed_kmh': 3600.})
    instance = adapt_instance_payload(payload, problem_type=problem, strict_road_metric=True)
    env = _env(instance=instance, observation_distance_scale_km=43.638668,
               prefer_explicit_edge_matrices=True)
    obs, _ = env.reset()
    feat = _features(_encoder(agda_physical_candidate_features=True, agda_smooth_distance_features=True),
                     StateWrapper(obs, 'cpu'), torch.zeros(1, count, 16))
    next_obs, *_ = env.step([1, 1])
    np.testing.assert_allclose(feat[0, :, 1, 17].numpy(), next_obs['current_load'], atol=2e-7)
    np.testing.assert_allclose(feat[0, :, 1, 20].numpy(), next_obs['current_time'], atol=2e-7)
    np.testing.assert_allclose(feat[0, :, 1, 24].numpy(), next_obs['current_battery'], atol=2e-7)


def test_explicit_energy_stays_active_when_nominal_consumption_is_zero():
    from caliroute.input_normalization import build_input_context
    energy = np.array([[0., 1.], [2., 0.]])
    ctx = build_input_context(distance_km=np.array([[0., 3.], [4., 0.]]),
        travel_time_s=np.array([[0., 30.], [40., 0.]]), energy_kwh=energy,
        distance_scale_km=10., horizon_s=1000., cargo_capacity_cm3=1e6,
        battery_capacity_kwh=100., speed_kmh=40., energy_per_km=0.,
        full_charge_time_s=100., charging_mode='fixed_full')
    assert ctx['graph_input_context'][6] == 1
    assert ctx['node_input_context'][0, 6] > 0


def test_historical_evrptw_solver_rejects_nonproportional_matrices_without_license():
    pytest.importorskip('gurobipy')
    from EVRPTW_Benchmark.Exact.Gurobi_Solver.EVRPTW.gurobi_solver import GurobiEVRPTWSolver
    instance = _instance()
    matrices = resolve_physical_edge_matrices(instance)
    equivalent = replace(instance, travel_time_matrix_s=matrices['travel_time_s'].astype(np.float32),
                         energy_matrix_kwh=matrices['energy_kwh'].astype(np.float32))
    GurobiEVRPTWSolver._check_physical_matrices(equivalent)
    for field in ('travel_time_matrix_s', 'energy_matrix_kwh'):
        changed = getattr(equivalent, field).copy()
        changed[0, 1] *= 2
        with pytest.raises(ValueError, match='refusing to discard'):
            GurobiEVRPTWSolver._check_physical_matrices(replace(equivalent, **{field: changed}))


def test_independent_vrptw_validator_uses_selected_explicit_times():
    from offline2online import trainer
    from test_vrptw_eval_validation import instance as audit_instance, row as audit_row
    instance = audit_instance()  # Deliberately supports lightweight dataset hosts.
    travel = np.array(instance.distance_matrix_km, copy=True)
    travel[0, 3] = 3.  # Due=1002; true travel now reaches this customer too late.
    instance.travel_time_matrix_s = travel
    assert trainer._validate_vrptw_eval_route(instance, audit_row())['valid']
    actual = trainer._validate_vrptw_eval_route(instance, audit_row(), prefer_explicit_edge_matrices=True)
    assert not actual['valid'] and not actual['time_windows_valid']
    assert actual['travel_time_source'] == 'provided_travel_time_matrix_s'
    assert actual['time_window_violations'][0]['customer'] == 3
    assert actual['time_window_violations'][0]['service_start_s'] == 1003.
    payload = dict(vars(instance), problem_class='VRPTW', depot=[0., 0.],
                   customers=[[1., 0.], [0., 1.], [1., 1.]])
    adapted = adapt_instance_payload(payload, problem_type='vrptw', strict_road_metric=True)
    env = _env(instance=adapted, prefer_explicit_edge_matrices=True)
    obs, _ = env.reset()
    assert not obs['action_mask'][:, 3].any()  # Independent audit and env agree.


def test_unreachable_edges_are_consistent_without_inactive_energy_nan():
    instance = _instance()
    distance = instance.distance_matrix_km.copy()
    distance[0, 2] = np.inf
    vehicle = dict(instance.vehicle, consumption_kwh_per_km=0.)
    instance = replace(instance, distance_matrix_km=distance, vehicle=vehicle)
    derived = resolve_physical_edge_matrices(instance)
    assert not np.isnan(derived['energy_kwh']).any()
    assert np.isposinf(derived['energy_kwh'][0, 2])
    explicit = replace(instance, travel_time_matrix_s=derived['travel_time_s'], energy_matrix_kwh=derived['energy_kwh'])
    assert validate_instance_structure(explicit).success
    resolve_physical_edge_matrices(explicit, prefer_explicit_edge_matrices=True)
    inconsistent = derived['energy_kwh'].copy()
    inconsistent[0, 2] = 0.
    with pytest.raises(ValueError, match='reachability'):
        resolve_physical_edge_matrices(replace(explicit, energy_matrix_kwh=inconsistent), prefer_explicit_edge_matrices=True)
