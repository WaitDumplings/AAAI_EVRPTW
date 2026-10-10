"""Independent CVRP route checks in physical kilometres, plus environment replay."""
from __future__ import annotations

import json
import math
import re

import numpy as np

DISTANCE_ATOL_KM = 1e-5
DISTANCE_RTOL = 1e-6


def validate_routes(instance, routes, reported_cost=None):
    """Require complete depot-to-depot routes; never repair an incomplete route."""
    n = int(instance.num_customers)
    d = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    demand = np.asarray(instance.demands_cm3, dtype=np.float64)
    capacity = float(instance.vehicle['cargo_capacity_cm3'])
    reasons, loads, coverage = [], [], np.zeros(n, dtype=np.int64)
    distance = 0.
    if not isinstance(routes, (list, tuple)) or not routes:
        return dict(valid=False, feasible=False, reasons=['missing_routes'], recomputed_distance_km=None, num_routes=0, action_count=0, distance_matches=False)
    if d.shape != (n+1,n+1) or demand.shape != (n,) or not np.isfinite(demand).all() or (demand<0).any() or not math.isfinite(capacity) or capacity<=0:
        return dict(valid=False, feasible=False, reasons=['invalid_instance_arrays'], recomputed_distance_km=None, num_routes=0, action_count=0, distance_matches=False)
    for route in routes:
        if not isinstance(route,(list,tuple)) or len(route)<3:
            reasons.append('invalid_route_shape'); continue
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 0<=x<=n for x in route):
            reasons.append('invalid_node_id'); continue
        if route[0]!=0 or route[-1]!=0 or 0 in route[1:-1]:
            reasons.append('missing_or_internal_depot')
        nodes=[int(x) for x in route if x!=0]
        for node in nodes: coverage[node-1]+=1
        load=float(sum(demand[node-1] for node in nodes)); loads.append(load)
        if load>capacity+1e-6*max(1.,capacity): reasons.append('capacity_exceeded')
        for left,right in zip(route,route[1:]):
            edge=float(d[left,right])
            if not math.isfinite(edge) or edge<0: reasons.append('invalid_road_edge')
            else: distance+=edge
    if np.any(coverage==0): reasons.append('missing_customers')
    if np.any(coverage>1): reasons.append('repeated_customers')
    feasible=not reasons
    error=None; matches=None
    if reported_cost is not None:
        try: objective=float(reported_cost)
        except (TypeError,ValueError): objective=float('nan')
        matches=math.isfinite(objective) and math.isclose(distance,objective,rel_tol=DISTANCE_RTOL,abs_tol=DISTANCE_ATOL_KM)
        error=distance-objective if math.isfinite(objective) else None
        if not matches: reasons.append('recorded_cost_mismatch')
    return dict(valid=not reasons, feasible=feasible, reasons=sorted(set(reasons)),
        recomputed_distance_km=distance if feasible else None, distance_matches=matches,
        distance_error_km=error, distance_atol_km=DISTANCE_ATOL_KM,distance_rtol=DISTANCE_RTOL,
        num_routes=len(routes), action_count=sum(max(0,len(route)-1) for route in routes if isinstance(route,(list,tuple))),
        route_loads_cm3=loads,missing_customers=(np.where(coverage==0)[0]+1).tolist(),
        repeated_customers=(np.where(coverage>1)[0]+1).tolist())


def routes_from_sequence(sequence):
    if not isinstance(sequence,list) or len(sequence)<3 or sequence[0]!=0 or sequence[-1]!=0:
        raise ValueError('route sequence must explicitly start and finish at depot')
    routes=[]; current=[0]
    for node in sequence[1:]:
        current.append(node)
        if node==0:
            if len(current)<3: raise ValueError('empty route in route sequence')
            routes.append(current); current=[0]
    return routes


def recover_routes_from_arcs(arcs):
    """Recover directed binary x[i,j] routes, without solving or guessing edges.

    Accepted payloads: [[i,j], ...], [[i,j,value], ...], dictionaries whose keys
    are 'i,j' or 'x[i,j]', or objects {from,to,value}. Vehicle-indexed variables
    need an explicit upstream conversion; ambiguous index orders are rejected.
    """
    edges=[]
    if isinstance(arcs,dict):
        for key,value in arcs.items():
            match=re.fullmatch(r'(?:x\[)?\s*(\d+)\s*,\s*(\d+)\s*\]?',str(key))
            if not match: raise ValueError('unsupported arc variable key')
            edges.append((int(match[1]),int(match[2]),value))
    elif isinstance(arcs,list):
        for item in arcs:
            if isinstance(item,dict): edges.append((item['from'],item['to'],item.get('value',1)))
            elif isinstance(item,list) and len(item) in (2,3): edges.append((*item[:2],item[2] if len(item)==3 else 1))
            else: raise ValueError('unsupported arc record')
    else: raise ValueError('unsupported arc payload')
    selected=[]
    for left,right,value in edges:
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (left,right)):
            raise ValueError('invalid arc node')
        value=float(value)
        if not math.isfinite(value) or min(abs(value),abs(value-1))>1e-6: raise ValueError('non-binary arc value')
        if value>.5:
            if left==right or (left,right) in selected: raise ValueError('self or duplicate arc')
            selected.append((left,right))
    successors={}; incoming={}
    for left,right in selected:
        successors.setdefault(left,[]).append(right); incoming[right]=incoming.get(right,0)+1
    if not successors.get(0): raise ValueError('no selected depot departures')
    nondepot=set(successors)|set(incoming); nondepot.discard(0)
    if any(len(successors.get(node,[]))!=1 or incoming.get(node)!=1 for node in nondepot):
        raise ValueError('ambiguous or incomplete customer arcs')
    routes=[]; used=set()
    for first in sorted(successors[0]):
        route=[0,first]; used.add((0,first)); node=first
        while node!=0:
            nxt=successors[node][0]
            if (node,nxt) in used: raise ValueError('cycle without depot')
            used.add((node,nxt)); route.append(nxt); node=nxt
        routes.append(route)
    if used!=set(selected): raise ValueError('disconnected subtour')
    return routes


def extract_routes(row):
    for key in ('routes_json','routes','route_json','solution_routes'):
        if row.get(key) not in (None,'','nan','NaN','None'):
            value=json.loads(row[key]) if isinstance(row[key],str) else row[key]
            if isinstance(value,dict): value=value.get('routes',value.get('solution'))
            if not isinstance(value,list): raise ValueError('invalid routes payload')
            return value,'routes'
    for key in ('route_sequence_json','route_sequence'):
        if row.get(key): return routes_from_sequence(json.loads(row[key]) if isinstance(row[key],str) else row[key]),'sequence'
    for key in ('arcs_json','arc_variables_json','x_json'):
        if row.get(key): return recover_routes_from_arcs(json.loads(row[key]) if isinstance(row[key],str) else row[key]),'arcs'
    raise ValueError('no complete routes or recoverable arc variables')


def teacher_force_routes(instance,routes,env_config=None):
    """Replay complete actions against the same factory used for expert training."""
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
    kwargs=dict(use_fast_env=True,use_jit_mask=False,info_level='light',normalize_reward=False,max_steps_factor=4)
    kwargs.update(env_config or {})
    kwargs.pop('n_traj',None)
    scale_mode=str(kwargs.get('reward_distance_scale_mode',''))
    if scale_mode.startswith('dataset_'): kwargs['reward_distance_scale_mode']=scale_mode[len('dataset_'):]
    env=make_terran_env(instance=instance,n_traj=1,pbrs_config=None,**kwargs)
    actions=[int(node) for route in routes for node in route[1:]]
    try:
        obs,info=env.reset(); steps=0; terminated=truncated=False
        for index,action in enumerate(actions):
            if terminated or truncated:
                return dict(valid=False,reason='terminated_before_last_action',step=index)
            mask=np.asarray(obs['action_mask'],dtype=bool)
            if action<0 or action>=mask.shape[1] or not mask[0,action]:
                return dict(valid=False,reason='expert_action_mask_violation',step=index,action=action)
            obs,_,term,trunc,info=env.step(np.asarray([action],dtype=np.int64)); steps+=1
            terminated=bool(np.asarray(term).reshape(-1)[0]); truncated=bool(np.asarray(trunc).reshape(-1)[0])
        success=bool(np.asarray(info.get('success',[False])).reshape(-1)[0])
        cost=float(np.asarray(info.get('objective_distance_km',[float('nan')])).reshape(-1)[0])
        valid=success and terminated and not truncated and math.isfinite(cost)
        return dict(valid=valid,reason='' if valid else 'expert_replay_incomplete_or_failed',
            completed=terminated,feasible=success,truncated=truncated,action_count=steps,
            recomputed_cost_km=cost if math.isfinite(cost) else None)
    finally:
        env.close()
