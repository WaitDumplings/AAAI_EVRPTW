"""EVRPTW physical audit and independent fixed-full-charge route validation."""
import hashlib
import json
import math
import numpy as np
from evrptw_core.physical import resolve_physical_edge_matrices
from evrptw_core.validation import validate_instance_structure


def physical_contract(instance):
    checked=validate_instance_structure(instance)
    if not checked.success:raise ValueError(str(checked.errors))
    edges=resolve_physical_edge_matrices(instance,prefer_explicit_edge_matrices=True)
    arrays=[instance.depot,instance.customers,instance.charging_stations,
            instance.demands_cm3,instance.tw_s,instance.service_time_s]
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite node/resource inputs')
    vehicle=instance.vehicle
    for key in ('cargo_capacity_cm3','battery_capacity_kwh','full_charge_time_s'):
        if not math.isfinite(float(vehicle[key])) or float(vehicle[key])<=0:raise ValueError('Invalid '+key)
    if instance.metadata.get('charging_constraint') is False or instance.metadata.get('time_window_constraint') is False:
        raise ValueError('EVRPTW resources must be active')
    return edges


def export_audit_dataset(path, output_path, *, split, expected_customers, problem_type='evrptw'):
    """Physical/identity audit only; never loads test reference solutions."""
    from offline2online.instance_adapter import iter_adapted_instances
    matrices=[];identities=[];seen=set()
    for i in iter_adapted_instances(path,problem_type='evrptw',strict_road_metric=True):
        if i.num_customers!=expected_customers or i.num_charging_stations!={15:3,50:10,100:20}[expected_customers]:
            raise ValueError('Unexpected customer/physical charging-station count')
        edges=physical_contract(i);h=hashlib.sha256()
        values=dict(depot=i.depot,customers=i.customers,stations=i.charging_stations,
                    demand=i.demands_cm3,tw=i.tw_s,service=i.service_time_s,
                    window=[i.working_start_s,i.working_end_s],
                    resources=[i.vehicle[k] for k in ('cargo_capacity_cm3','battery_capacity_kwh','full_charge_time_s')],
                    **{k:edges[k] for k in ('distance_km','travel_time_s','energy_kwh')})
        for k,a in sorted(values.items()):
            a=np.asarray(a,dtype='<f8');h.update(k.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        if i.instance_id in seen:raise ValueError('Duplicate instance identity')
        seen.add(i.instance_id)
        identities.append(dict(instance_id=i.instance_id,semantic_sha256=h.hexdigest(),
            territory_id=str(i.metadata.get('service_territory_id') or i.region_id)))
        matrices.append(edges['distance_km'])
    if not matrices:raise ValueError('Empty EVRPTW split')
    np.savez(output_path,distance_matrix=np.stack(matrices))
    return dict(count=len(matrices),instances=identities)


def validate_evrptw_routes(instance,routes,reported_cost=None):
    reasons=[];n=instance.num_customers;m=instance.num_charging_stations
    def invalid(reason):
        return dict(valid=False,feasible=False,reasons=[reason],recomputed_distance_km=None,
                    num_routes=0,action_count=0,distance_matches=False)
    try:edges=physical_contract(instance)
    except (ValueError,KeyError,TypeError):return invalid('invalid_evrptw_physical_contract')
    if not isinstance(routes,(list,tuple)) or not routes:return invalid('missing_routes')
    distance=0.;coverage=np.zeros(n,dtype=int);loads=[];actions=0
    capacity=float(instance.vehicle['cargo_capacity_cm3']);battery=float(instance.vehicle['battery_capacity_kwh'])
    charge=float(instance.vehicle['full_charge_time_s']);closing=float(instance.working_end_s)
    for route in routes:
        if not isinstance(route,(list,tuple)) or len(route)<3:
            reasons.append('invalid_route_shape');continue
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 0<=x<=n+m for x in route):
            reasons.append('invalid_node_id');continue
        if route[0]!=0 or route[-1]!=0 or 0 in route[1:-1]:reasons.append('missing_or_internal_depot')
        if not any(1<=x<=n for x in route):reasons.append('route_without_customer')
        clock=float(instance.working_start_s);used=0.;load=0.;stations=set()
        for left,right in zip(route,route[1:]):
            actions+=1
            d,t,e=(float(edges[k][left,right]) for k in ('distance_km','travel_time_s','energy_kwh'))
            if not np.isfinite([d,t,e]).all():reasons.append('unreachable_edge');continue
            distance+=d;clock+=t;used+=e
            if used>battery+1e-6:reasons.append('battery_exceeded')
            if 1<=right<=n:
                coverage[right-1]+=1;load+=float(instance.demands_cm3[right-1])
                clock=max(clock,float(instance.tw_s[right-1,0]))
                if clock>float(instance.tw_s[right-1,1])+.01:reasons.append('time_window_exceeded')
                clock+=float(instance.service_time_s[right-1])
            elif right>n:
                if right in stations:reasons.append('repeated_station_in_route')
                stations.add(right);clock+=charge;used=0.
            if clock>closing+.01:reasons.append('depot_closing_exceeded')
        loads.append(load)
        if load>capacity+1e-6*max(1.,capacity):reasons.append('capacity_exceeded')
    if (coverage==0).any():reasons.append('missing_customers')
    if (coverage>1).any():reasons.append('repeated_customers')
    feasible=not reasons;matches=None;error=None
    if reported_cost is not None:
        try:objective=float(reported_cost)
        except (ValueError,TypeError):objective=float('nan')
        matches=math.isfinite(objective) and math.isclose(distance,objective,rel_tol=1e-6,abs_tol=1e-5)
        error=distance-objective if math.isfinite(objective) else None
        if not matches:reasons.append('recorded_cost_mismatch')
    return dict(valid=not reasons,feasible=feasible,reasons=sorted(set(reasons)),
        recomputed_distance_km=distance if feasible else None,distance_matches=matches,
        distance_error_km=error,num_routes=len(routes),action_count=actions,route_loads_cm3=loads,
        missing_customers=(np.where(coverage==0)[0]+1).tolist(),repeated_customers=(np.where(coverage>1)[0]+1).tolist())
