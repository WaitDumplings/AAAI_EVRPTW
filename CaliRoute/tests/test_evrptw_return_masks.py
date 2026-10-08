"""Charging reachability must agree with executable station masks."""
import numpy as np
import pytest
from evrptw_core.schema import EVRPTWInstance
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env


def instance(distance, capacity=6., end=1000.):
    d=np.asarray(distance,dtype=float);ncs=len(d)-2
    return EVRPTWInstance.from_dict(dict(instance_id='return-mask', working_start_s=0.,working_end_s=end,
        depot=[0.,0.],customers=[[1.,0.]],charging_stations=[[float(i+2),0.] for i in range(ncs)],
        distance_matrix_km=d,demands_cm3=[1.],package_counts=[1.],service_time_s=[1.],tw_s=[[0.,end]],
        cs_time_to_depot_s=d[2:,0],vehicle=dict(cargo_capacity_cm3=10.,battery_capacity_kwh=capacity,
        consumption_kwh_per_km=1.,full_charge_time_s=1.),speed_profile=dict(effective_speed_kmh=3600.)))


@pytest.mark.parametrize('fast,jit',[(False,False),(True,False),(True,True)])
def test_last_customer_can_charge_before_finishing(fast,jit):
    env=make_terran_env(instance=instance([[0,4,3],[6,0,2],[4,5,0]]),n_traj=1,
        use_fast_env=fast,use_jit_mask=jit)
    obs,_=env.reset(seed=1)
    for action in (1,2,0):
        assert obs['action_mask'][0,action]
        obs,_,terminated,truncated,info=env.step([action])
        assert not truncated[0]
    assert terminated[0] and info['success'][0]
    assert env.unwrapped.get_routes()==[[[0,1,2,0]]]
    assert env.unwrapped.objective_distance_km[0]==10.


@pytest.mark.parametrize('fast,jit',[(False,False),(True,False),(True,True)])
def test_return_path_cannot_use_a_previously_visited_station_later(fast,jit):
    d=np.full((4,4),100.);np.fill_diagonal(d,0.)
    d[0,1]=1.;d[1,2]=4.;d[2,3]=4.;d[3,0]=1.
    env=make_terran_env(instance=instance(d,capacity=5.),n_traj=1,use_fast_env=fast,use_jit_mask=jit)
    obs,_=env.reset(seed=1);raw=env.unwrapped
    assert obs['action_mask'][0,1]  # 1 -> unused CS 2 -> unused CS 3 -> depot
    raw.cs_visited_current_route[0,3]=True
    assert not raw._can_return_to_depot(1,2.,1.,traj_idx=0)
    assert not raw._compute_action_mask()[0,1]


@pytest.mark.parametrize('fast,jit',[(False,False),(True,False),(True,True)])
def test_direct_time_failure_does_not_hide_feasible_charge_detour(fast,jit):
    # Authoritative edge time need not have the same shortest path as distance.
    obj=instance([[0,1,1],[1,0,1],[1,1,0]],capacity=100.,end=20.)
    obj.raw['travel_time_matrix_s']=np.array([[0.,1.,1.],[100.,0.,1.],[1.,1.,0.]])
    env=make_terran_env(instance=obj,n_traj=1,use_fast_env=fast,use_jit_mask=jit,
        prefer_explicit_edge_matrices=True)
    obs,_=env.reset(seed=1)
    for action in (1,2,0):
        assert obs['action_mask'][0,action]
        obs,_,terminated,truncated,info=env.step([action])
        assert not truncated[0]
    assert info['success'][0]


def test_jit_and_reference_agree_with_route_specific_station_restrictions():
    rng=np.random.default_rng(5209)
    for _ in range(8):
        d=rng.uniform(1.,9.,size=(7,7));np.fill_diagonal(d,0.)
        obj=instance(d,capacity=6.,end=30.)
        reference=make_terran_env(instance=obj,n_traj=1,use_fast_env=False)
        accelerated=make_terran_env(instance=obj,n_traj=1,use_fast_env=True,use_jit_mask=True)
        reference.reset();accelerated.reset()
        for _ in range(15):
            last=int(rng.integers(1,7));served=int(rng.integers(0,2))
            used=rng.random(7)<.4;used[:2]=False
            if last>=2:used[last]=True
            now=float(rng.uniform(0.,32.));energy=float(rng.uniform(0.,6.))
            for wrapped in (reference,accelerated):
                env=wrapped.unwrapped;env.last[0]=last
                env.current_time_s[0]=now;env.battery_used_kwh[0]=energy
                env.cs_visited_current_route[0]=used
                env.visited[0,1]=bool(served);env.served_customers[0]=served
                env.route_has_customer[0]=True
            np.testing.assert_array_equal(reference.unwrapped._compute_action_mask(),
                                          accelerated.unwrapped._compute_action_mask())
