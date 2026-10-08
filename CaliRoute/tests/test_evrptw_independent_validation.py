"""Independent complete-route physics, without importing a trainer or model."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evrptw_independent_validation', ROOT/'scripts/original_eval_validation.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def instance():
    return SimpleNamespace(instance_id='ev', num_customers=2, num_charging_stations=1,
        num_terminals=4, demands_cm3=np.ones(2), service_time_s=np.ones(2),
        tw_s=np.array([[100., 130.], [100., 130.]]), working_start_s=100., working_end_s=130.,
        distance_matrix_km=2. * (np.ones((4, 4)) - np.eye(4)),
        depot=np.zeros(2), customers=np.zeros((2, 2)), charging_stations=np.zeros((1, 2)),
        metadata={}, speed_profile={'effective_speed_kmh': 3600.},
        vehicle=dict(cargo_capacity_cm3=2., battery_capacity_kwh=5.,
                     consumption_kwh_per_km=1., full_charge_time_s=5.))


def row(routes=None, data=None):
    data = instance() if data is None else data
    routes = [[0, 1, 3, 2, 0]] if routes is None else routes
    return dict(routes=routes, objective_distance_km=float(sum(
        data.distance_matrix_km[a, b] for route in routes for a, b in zip(route, route[1:]))))


def test_fixed_full_charge_has_correct_units_and_resets_only_energy():
    checked = validation.validate_evrptw_route(instance(), row())
    assert checked['valid']
    assert checked['route_return_times_s'] == [115.]
    assert checked['route_loads_cm3'] == [2.]
    assert checked['route_max_battery_used_kwh'] == [4.]
    assert checked['route_charge_counts'] == [1]
    assert checked['charge_events'] == [dict(route_index=0, station=3, arrival_time_s=105.,
        departure_time_s=110., battery_used_before_charge_kwh=4., battery_used_after_charge_kwh=0., charge_time_s=5.)]
    assert checked['travel_time_source'] == 'distance_over_effective_speed'
    assert checked['energy_source'] == 'distance_times_consumption'


def test_battery_must_reach_station_before_recharge_and_depot_before_reset():
    data = instance()
    data.energy_matrix_kwh = data.distance_matrix_km.copy()
    data.energy_matrix_kwh[1, 3] = 4.
    checked = validation.validate_evrptw_route(data, row())
    assert not checked['valid'] and not checked['battery_valid']
    assert checked['battery_violations'][0]['to'] == 3
    data.energy_matrix_kwh[1, 3] = 2.
    data.energy_matrix_kwh[2, 0] = 4.
    checked = validation.validate_evrptw_route(data, row())
    assert not checked['battery_valid'] and checked['battery_violations'][0]['to'] == 0


def test_full_charge_time_is_not_proportional_even_when_no_energy_was_used():
    data = instance()
    data.vehicle['consumption_kwh_per_km'] = 0.
    data.working_end_s = 114.
    checked = validation.validate_evrptw_route(data, row())
    assert checked['charge_events'][0]['battery_used_before_charge_kwh'] == 0.
    assert checked['charge_events'][0]['charge_time_s'] == 5.
    assert checked['route_return_times_s'] == [115.]
    assert not checked['valid'] and not checked['depot_return_valid']


def test_station_does_not_unload_or_reset_route_clock():
    data = instance()
    data.vehicle['cargo_capacity_cm3'] = 1.
    checked = validation.validate_evrptw_route(data, row())
    assert not checked['capacity_valid']
    assert checked['route_return_times_s'] == [115.]
    data.vehicle['cargo_capacity_cm3'] = 2.
    data.tw_s[1, 1] = 111.
    assert not validation.validate_evrptw_route(data, row())['time_windows_valid']


def test_each_vehicle_starts_with_fresh_clock_load_battery_and_station_availability():
    data = instance()
    data.vehicle['cargo_capacity_cm3'] = 1.
    data.tw_s[:, 1] = 103.
    data.working_end_s = 112.
    checked = validation.validate_evrptw_route(data, row([[0, 1, 3, 0], [0, 2, 3, 0]], data))
    assert checked['valid']
    assert checked['route_return_times_s'] == [112., 112.]
    assert checked['route_loads_cm3'] == [1., 1.]
    assert checked['route_charge_counts'] == [1, 1]


def test_service_start_windows_waiting_and_final_completion_are_distinct():
    data = instance()
    data.tw_s[0] = [104., 104.]
    checked = validation.validate_evrptw_route(data, row())
    assert checked['valid'] and checked['route_waiting_times_s'] == [2.]
    assert checked['route_return_times_s'] == [117.]
    data.service_time_s[1] = 50.
    checked = validation.validate_evrptw_route(data, row())
    assert checked['time_windows_valid']
    assert not checked['service_completion_valid'] and not checked['depot_return_valid']


def test_cs_revisits_are_physical_events_not_implicit_gurobi_copy_limits():
    checked = validation.validate_evrptw_route(instance(), row([[0, 1, 3, 2, 3, 0]]))
    assert checked['valid'] and checked['route_charge_counts'] == [2]
    assert len(checked['charge_events']) == 2
    assert checked['validation_scope'].endswith('environment_action_masks_checked_separately')


@pytest.mark.parametrize('location', ['attribute', 'raw', 'sidecar'])
def test_authoritative_directed_time_and_energy_are_preserved(location):
    data = instance()
    travel = data.distance_matrix_km.copy()
    energy = data.distance_matrix_km.copy()
    travel[3, 2] = 20.
    energy[1, 3] = 4.
    edges = dict(travel_time_matrix_s=travel, energy_matrix_kwh=energy)
    kwargs = {}
    if location == 'attribute':
        data.travel_time_matrix_s, data.energy_matrix_kwh = travel, energy
    elif location == 'raw':
        data.raw = edges
    else:
        kwargs['raw_payload'] = edges
    checked = validation.validate_evrptw_route(data, row(), **kwargs)
    assert not checked['valid'] and not checked['battery_valid'] and not checked['depot_return_valid']
    assert checked['travel_time_source'] == 'provided_travel_time_matrix_s'
    assert checked['energy_source'] == 'provided_energy_matrix_kwh'
    assert data.distance_matrix_km[1, 3] == 2.  # No mutation or median reconstruction.
    assert checked['recomputed_distance_km'] == 8.


@pytest.mark.parametrize('kind', ['time', 'energy'])
@pytest.mark.parametrize('fault', ['shape', 'negative', 'nan', 'reachability'])
def test_malformed_explicit_edges_fail_instead_of_falling_back(kind, fault):
    matrix = instance().distance_matrix_km.copy()
    if fault == 'shape':
        matrix = matrix[:-1]
    elif fault == 'negative':
        matrix[0, 1] = -1.
    elif fault == 'nan':
        matrix[0, 1] = np.nan
    else:
        matrix[0, 1] = np.inf
    key = 'travel_time_matrix_s' if kind == 'time' else 'energy_matrix_kwh'
    with pytest.raises(ValueError, match='Authoritative'):
        validation.validate_evrptw_route(instance(), row(), raw_payload={key: matrix})


def test_unreachable_edges_are_rejected_and_unused_unreachable_edges_are_legal():
    data = instance()
    data.distance_matrix_km[0, 3] = np.inf
    assert validation.validate_evrptw_route(data, row())['valid']
    data.distance_matrix_km[1, 3] = np.inf
    checked = validation.validate_evrptw_route(data, row())
    assert not checked['valid'] and not checked['traversed_edges_reachable']
    assert checked['unreachable_edges'] == [dict(route_index=0, **{'from': 1, 'to': 3})]
    data.vehicle['consumption_kwh_per_km'] = 0.
    with np.errstate(invalid='raise'):
        energy, _ = validation.resolve_validation_energy(data)
    assert np.isinf(energy[1, 3]) and energy[0, 1] == 0.


@pytest.mark.parametrize('routes', [[], [[0, 1, 3, 0]], [[0, 1, 1, 3, 2, 0]],
    [[0, 1, 3, 2]], [[0, 1, 3, 2, 0], [0, 3, 0]], [[0, 1, 0, 2, 0]],
    [[0, 1, 3, 2, 0, 0]], [[0, 1, True, 2, 0]], [[0, 1, 4, 2, 0]], [[0, 1., 3, 2, 0]]])
def test_route_coverage_indices_and_endpoints_are_not_inferred_or_repaired(routes):
    checked = validation.validate_evrptw_route(instance(), dict(routes=routes, objective_distance_km=8.))
    assert not checked['valid']


def test_directed_distance_matches_actual_arcs_and_wrong_objective_is_rejected():
    data = instance()
    data.distance_matrix_km[3, 1] = 9.  # Reverse edge is not part of the route.
    assert validation.validate_evrptw_route(data, row())['valid']
    data.distance_matrix_km[1, 3] = 2.25
    data.vehicle['consumption_kwh_per_km'] = .5
    assert not validation.validate_evrptw_route(data, row())['distance_matches']
    assert validation.validate_evrptw_route(data, row(data=data))['valid']


@pytest.mark.parametrize('mode', ['proportional_full', 'partial', None])
def test_unsupported_charging_is_explicit(mode):
    with pytest.raises(ValueError, match='fixed_full only'):
        validation.validate_evrptw_route(instance(), row(), charging_mode=mode)


@pytest.mark.parametrize('n', [15, 50, 100])
def test_task_sizes_with_physical_stations_and_vehicle_resets(n):
    data = instance()
    data.num_customers = n
    data.num_terminals = n + 2
    data.demands_cm3 = np.ones(n)
    data.service_time_s = np.ones(n)
    data.tw_s = np.array([[100., 130.]] * n)
    data.distance_matrix_km = 2. * (np.ones((n + 2, n + 2)) - np.eye(n + 2))
    routes = [[0, customer, n + 1, 0] for customer in range(1, n + 1)]
    checked = validation.validate_evrptw_route(data, row(routes, data))
    assert checked['valid'] and checked['recomputed_vehicle_count'] == n
    assert checked['recomputed_distance_km'] == n * 6.
    assert checked['route_return_times_s'] == [112.] * n


def test_shared_fixed_full_transition_matches_environment_cpu_replay():
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env import EVRPTWVectorEnv
    data = instance()
    data.travel_time_matrix_s = data.distance_matrix_km.copy()
    data.travel_time_matrix_s[3, 2] = 3.
    data.energy_matrix_kwh = .5 * data.distance_matrix_km
    env = EVRPTWVectorEnv(instance=data, n_traj=1, prefer_explicit_edge_matrices=True,
                         charging_mode='fixed_full')
    observation, _ = env.reset(seed=3)
    for action in [1, 3, 2]:
        assert observation['action_mask'][0, action]
        observation, _, _, _, _ = env.step(np.array([action]))
    return_time = float(env.current_time_s[0] + data.travel_time_matrix_s[2, 0])
    assert observation['action_mask'][0, 0]
    _, _, _, _, info = env.step(np.array([0]))
    assert info['success'][0]
    candidate = dict(routes=env.get_routes()[0], objective_distance_km=info['objective_distance_km'][0])
    checked = validation.validate_evrptw_route(data, candidate)
    assert checked['valid'] and checked['route_return_times_s'] == [return_time]
    assert checked['charge_events'][0]['battery_used_before_charge_kwh'] == 2.
