"""Physical time invariants and no silent CVRP fallback in native VRPTW."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from e1.native_vrptw import convert_vrptw_instances,radar_environment
from e1.validator import validate_routes


def test_encoder_recomputation_preserves_outputs_gradients_and_state_keys():
    import copy
    from torch import nn
    from e1.native_vrptw import checkpoint_rrnco_encoder

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4)
            self.dropout = nn.Dropout(.2)

        def forward(self, left, right):
            return self.dropout(self.projection(left)).tanh(), self.projection(right).sin()

    reference = nn.Module()
    reference.encoder = nn.Module()
    reference.encoder.net = nn.Module()
    reference.encoder.net.layers = nn.ModuleList([Layer(), Layer()])
    recomputed = copy.deepcopy(reference)
    checkpoint_rrnco_encoder(recomputed)
    assert reference.state_dict().keys() == recomputed.state_dict().keys()
    inputs = [torch.randn(3, 4), torch.randn(3, 4)]
    outputs = []
    for model in (reference, recomputed):
        torch.manual_seed(3011)
        left, right = [x.clone().requires_grad_() for x in inputs]
        for layer in model.encoder.net.layers:
            left, right = layer(left, right)
        (left.square().sum() + right.square().sum()).backward()
        outputs.append((left.detach(), right.detach()))
    for expected, actual in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(reference.parameters(), recomputed.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    with torch.no_grad():
        torch.manual_seed(3011)
        expected = reference.encoder.net.layers[0](*inputs)
        torch.manual_seed(3011)
        actual = recomputed.encoder.net.layers[0](*inputs)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)


def instance():
    return SimpleNamespace(num_customers=3,num_charging_stations=0,instance_id='x',
        distance_matrix_km=np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]],dtype=np.float32),
        travel_time_matrix_s=np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]],dtype=np.float32),
        demands_cm3=np.ones(3,dtype=np.float32),vehicle={'cargo_capacity_cm3':10.},
        depot=np.array([0.,0.]),customers=np.array([[1.,0.],[2.,0.],[3.,0.]]),
        tw_s=np.array([[10.,12.],[10.,29.],[10.,29.]],dtype=np.float32),service_time_s=np.ones(3,dtype=np.float32),
        working_start_s=10.,working_end_s=30.,metadata={'time_window_constraint':True},region_id='r')


def test_time_order_return_and_waiting_independently_checked():
    i=instance()
    assert validate_routes(i,[[0,1,2,3,0]])['feasible']
    assert 'time_window_exceeded' in validate_routes(i,[[0,3,2,1,0]])['reasons']
    assert validate_routes(i,[[0,3,2,0],[0,1,0]])['feasible'] # reset at depot per vehicle
    i.working_end_s=15.
    assert 'depot_closing_exceeded' in validate_routes(i,[[0,1,2,3,0]])['reasons']


def test_shared_time_units_preserve_physics_and_fingerprint():
    i=instance();a,ids=convert_vrptw_instances([i],3)
    np.testing.assert_allclose(a['duration_matrix'][0]*20,i.travel_time_matrix_s)
    np.testing.assert_allclose(a['time_windows'][0,1:]*20+10,i.tw_s)
    np.testing.assert_array_equal(a['distance_matrix'][0],i.distance_matrix_km)
    i.service_time_s[0]+=1
    _,changed=convert_vrptw_instances([i],3)
    assert ids[0]['semantic_sha256']!=changed[0]['semantic_sha256']
    i.tw_s[0,1]=10.5
    with pytest.raises(ValueError,match='round trip'):convert_vrptw_instances([i],3)


def test_radar_mask_closes_late_customer_but_reopens_after_depot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/native_sources/radar/acvrp'))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/native_sources/radar'))
    from ACVRPEnv import ACVRPEnv
    env=radar_environment(ACVRPEnv)(node_cnt=3,pomo_size=1)
    a,_=convert_vrptw_instances([instance()],3)
    env.load_problems_manual(torch.from_numpy(a['distance_matrix']),torch.from_numpy(a['demand']/a['capacity'][:,None]))
    env.set_time_data(a,0,1);env.reset()
    env.step(torch.tensor([[0]]));env.step(torch.tensor([[3]]))
    assert torch.isneginf(env.ninf_mask[0,0,1])
    with pytest.raises(ValueError,match='infeasible'):env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    assert env.current_time[0,0]==0 and torch.isfinite(env.ninf_mask[0,0,1])
    env.step(torch.tensor([[1]]))
    assert float(env.current_time[0,0])==pytest.approx(.1)


def test_rrnco_mask_uses_directed_time_and_depot_reset():
    from e1.native_vrptw import rrnco_environment
    a,_=convert_vrptw_instances([instance()],3)
    td=dict(current_node=torch.tensor([3]),current_time=torch.tensor([[.2]]),
        duration_matrix=torch.from_numpy(a['duration_matrix']),
        time_windows=torch.from_numpy(a['time_windows']),service_time=torch.from_numpy(a['service_time']),
        demand_linehaul=torch.tensor([[0.,.1,.1,.1]]),used_capacity_linehaul=torch.tensor([[.1]]),
        vehicle_capacity=torch.ones(1,1),visited=torch.tensor([[False,False,False,True]]))
    env=rrnco_environment(object)
    mask=env.get_action_mask(td)
    assert mask[0,0] and not mask[0,1] and mask[0,2]
    td.update(current_node=torch.tensor([0]),current_time=torch.zeros(1,1),used_capacity_linehaul=torch.zeros(1,1))
    mask=env.get_action_mask(td)
    assert not mask[0,0] and mask[0,1] and mask[0,2]
    # Directed return time can invalidate a customer despite short outbound time.
    td['duration_matrix'][0,2,0]=2.
    assert not env.get_action_mask(td)[0,2]


def test_waiting_counts_toward_return_and_rejects_invalid_horizon():
    i=instance()
    i.tw_s[0]=[25.,29.]
    # Arrival at 11 must wait to 25; service and remaining route miss closing.
    assert 'depot_closing_exceeded' in validate_routes(i,[[0,1,2,3,0]])['reasons']
    assert validate_routes(i,[[0,1,0],[0,2,3,0]])['feasible']
    i.working_end_s=float('nan')
    assert 'invalid_time_arrays' in validate_routes(i,[[0,1,2,3,0]])['reasons']
    with pytest.raises(ValueError,match='positive horizon'):convert_vrptw_instances([i],3)
