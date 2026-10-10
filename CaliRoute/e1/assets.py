"""Read-only E1 CVRP asset audit and one validated train-only expert pool.

No solver, model, GPU, test reward, or test solution is invoked. Test instances
are inspected solely for identity, split separation and the data contract.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from offline2online.instance_adapter import iter_instance_payloads,adapt_instance_payload
from e1.validator import validate_routes,extract_routes,teacher_force_routes,DISTANCE_ATOL_KM,DISTANCE_RTOL


SCHEMA='aaai_e1_cvrp_assets_v1'


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def _write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def _write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def _finite_cost(value):
    try: value=float(value)
    except (ValueError,TypeError): return None
    return value if math.isfinite(value) and value>=0 else None


def _fingerprint(instance):
    h=hashlib.sha256()
    for name,array in [('distance_matrix_km',instance.distance_matrix_km),('demands_cm3',instance.demands_cm3),
        ('depot',instance.depot),('customers',instance.customers),('capacity',[instance.vehicle['cargo_capacity_cm3']])]:
        values=np.asarray(array,dtype='<f8');h.update(name.encode());h.update(str(values.shape).encode());h.update(values.tobytes())
    return h.hexdigest()


def _data_contract(raw,instance,customers):
    failures=[]
    if instance.num_customers!=customers: failures.append('customer_count_mismatch')
    if instance.num_charging_stations!=0: failures.append('cvrp_has_charging_stations')
    distance=np.asarray(raw['distance_matrix_km'],dtype=np.float32)
    if distance.shape!=(customers+1,customers+1) or not np.array_equal(distance,instance.distance_matrix_km):
        failures.append('road_matrix_changed_by_adapter')
    if not np.array_equal(np.asarray(raw['depot'],dtype=np.float32),instance.depot): failures.append('depot_changed_by_adapter')
    demands=np.asarray(raw.get('demands_cm3',raw.get('demands')),dtype=np.float32).reshape(-1)
    if not np.array_equal(demands,instance.demands_cm3): failures.append('demands_changed_by_adapter')
    capacity=float(raw.get('vehicle',{}).get('cargo_capacity_cm3',raw.get('capacity',float('nan'))))
    if not math.isfinite(capacity) or capacity<=0 or capacity!=float(instance.vehicle['cargo_capacity_cm3']): failures.append('capacity_changed_or_invalid')
    if not np.isfinite(demands).all() or (demands<0).any(): failures.append('invalid_demands')
    if instance.metadata.get('charging_constraint') is not False or instance.metadata.get('time_window_constraint') is not False:
        failures.append('inactive_cvrp_constraints_not_disabled')
    if float(instance.vehicle['consumption_kwh_per_km'])!=0 or np.any(instance.service_time_s!=0): failures.append('inactive_energy_or_service_cost')
    finite=distance[np.isfinite(distance)]
    max_distance=float(finite.max()) if finite.size else float('inf')
    max_route_time=max_distance*(2*customers+1)*3600/float(instance.speed_profile['effective_speed_kmh'])
    if max_route_time>=instance.working_end_s-instance.working_start_s: failures.append('surrogate_time_horizon_may_bind_cvrp')
    if raw.get('metadata',{}).get('objective_unit','km')!='km': failures.append('non_km_objective_unit')
    return failures


def _expert_candidates(train_dir):
    by_id=defaultdict(list); seen={}; raw_counts={}; missing=[]
    route_keys=('routes_json','routes','route_json','solution_routes','route_sequence_json','route_sequence','arcs_json','arc_variables_json','x_json')
    for name in ('expert_solutions.csv','gurobi_summary.csv'):
        path=train_dir/name
        if not path.is_file(): missing.append(str(path));continue
        count=0
        with path.open(newline='',encoding='utf-8-sig') as handle:
            for index,row in enumerate(csv.DictReader(handle),2):
                count+=1;identity=str(row.get('instance_id',''))
                key=(identity,row.get('objective_distance_km',''),*(row.get(k,'') for k in route_keys))
                source={'path':str(path),'line':index}
                if key in seen: seen[key]['sources'].append(source);continue
                item={'row':row,'sources':[source]};seen[key]=item;by_id[identity].append(item)
        raw_counts[name]=count
    return by_id,raw_counts,missing


def audit_assets(data_root,output_dir,*,teacher_force=True,expected_customers=100,env_config=None,progress=None):
    """Audit all split identities and every available train expert candidate.

    The output directory must be new. ``teacher_force=False`` is an incomplete
    inspection and never produces a training-ready expert pool. This function
    works on tiny fixture scales for tests; the E1 runner selects Cus100 only.
    """
    root=Path(data_root).expanduser().resolve();out=Path(output_dir).expanduser().resolve()
    if root.name=='dataset': raise ValueError('data_root must contain dataset/ and test_release/')
    out.mkdir(parents=True,exist_ok=False)
    paths={'train':root/'dataset/cvrp/train'/f'Cus{expected_customers}',
           'val':root/'dataset/cvrp/val'/f'Cus{expected_customers}',
           'test':root/'test_release/cvrp/test'/f'Cus{expected_customers}'}
    report={'schema':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),'problem':'cvrp',
        'customers':expected_customers,'data_root':str(root),'output_dir':str(out),'splits':{},
        'expert_pool':{},'split_overlaps':{},'errors':[],
        'test_policy':'Test read only for identity, split overlap and physical data contract; no test solutions or Gurobi results are opened.',
        'teacher_forcing':{'enabled':bool(teacher_force),'env_config':env_config or {},'factory':'make_terran_env','n_traj':1},
        'distance_tolerance':{'atol_km':DISTANCE_ATOL_KM,'rtol':DISTANCE_RTOL}}
    candidates,source_rows,missing_experts=_expert_candidates(paths['train'])
    indices=[];expert_rows=[];pool_rows=[];split_ids={};split_fps={};split_source_keys={};counters=Counter();reasons=Counter()
    visited_train=set();incumbent_ids=set();route_ids=set();independent_ids=set();tf_ids=set();cost_ids=set()
    for split,path in paths.items():
        data={'path':str(path),'present':path.is_dir(),'count':0,'files':{},'territory_counts':{},'contract_failures':{},'duplicate_instance_ids':[]}
        report['splits'][split]=data;split_ids[split]=set();split_fps[split]=set();split_source_keys[split]=set()
        if not (path/'instances.pkl').is_file():
            data['valid']=False;report['errors'].append(f'{split}: missing instances.pkl');continue
        metadata={}
        if (path/'metadata.json').is_file(): metadata=json.loads((path/'metadata.json').read_text())
        data['metadata']=metadata;data['dataset_version']=metadata.get('dataset_version');data['declared_count']=metadata.get('num_instances')
        file_names=['instances.pkl','metadata.json']
        if split!='test': file_names+=['public_metadata.json','expert_solutions.csv','gurobi_summary.csv']
        for name in file_names:
            file=path/name
            if file.is_file(): data['files'][name]={'path':str(file),'sha256':_hash(file),'bytes':file.stat().st_size}
        declared_hash=metadata.get('bundle_sha256')
        data['bundle_hash_matches_metadata']=declared_hash is None or declared_hash==data['files']['instances.pkl']['sha256']
        if not data['bundle_hash_matches_metadata']: report['errors'].append(f'{split}: metadata bundle SHA256 mismatch')
        territory=Counter();contract_failures=Counter();duplicates=[]
        for index,raw in enumerate(iter_instance_payloads(path)):
            data['count']+=1
            try:
                instance=adapt_instance_payload(raw,problem_type='cvrp',strict_road_metric=True)
                identity=str(instance.instance_id)
                if not identity or identity=='adapted_instance': raise ValueError('missing original instance_id')
                failures=_data_contract(raw,instance,expected_customers)
            except Exception as error:
                contract_failures[f'{type(error).__name__}:{error}']+=1;continue
            for failure in failures:contract_failures[failure]+=1
            if identity in split_ids[split]: duplicates.append(identity)
            split_ids[split].add(identity)
            meta=instance.metadata
            area=str(meta.get('service_territory_id') or meta.get('source_territory_id') or instance.region_id or 'missing')
            territory[area]+=1;fp=_fingerprint(instance);split_fps[split].add(fp)
            source_key=json.dumps([meta.get('source_split'),meta.get('source_territory_id',area),meta.get('source_instance_id')])
            if meta.get('source_instance_id') is not None:split_source_keys[split].add(source_key)
            indices.append(dict(split=split,instance_id=identity,territory_id=area,instance_key=meta.get('instance_key',''),
                semantic_sha256=fp,source_key=source_key,dataset_version=meta.get('dataset_version',data['dataset_version']),contract_valid=not failures))
            if split!='train':continue
            visited_train.add(identity);accepted=[]
            for item in candidates.get(identity,[]):
                row=item['row'];cost=_finite_cost(row.get('objective_distance_km'));audit={'instance_id':identity,
                    'sources_json':json.dumps(item['sources'],sort_keys=True),'recorded_cost_km':cost,'recovered_cost':cost is None,
                    'route_source':'','independent_valid':False,'teacher_forcing_valid':False,'cost_consistent':False,
                    'accepted':False,'reasons_json':'[]'}
                counters['candidate_rows']+=1
                if cost is not None: incumbent_ids.add(identity)
                fail=list(failures)
                try:
                    routes,source=extract_routes(row);audit['route_source']=source;route_ids.add(identity)
                    checked=validate_routes(instance,routes,cost);audit['independent_valid']=checked.get('feasible',False)
                    audit['cost_consistent']=checked.get('distance_matches') is True
                    audit['recomputed_cost_km']=checked.get('recomputed_distance_km')
                    if checked.get('feasible'):independent_ids.add(identity)
                    if checked.get('distance_matches') is True:cost_ids.add(identity)
                    fail.extend(checked['reasons'])
                    if not fail and teacher_force:
                        tf=teacher_force_routes(instance,routes,env_config)
                        audit['teacher_forcing_valid']=tf['valid'];audit['teacher_forcing_cost_km']=tf.get('recomputed_cost_km')
                        if tf['valid']:
                            tf_ids.add(identity)
                            if not math.isclose(tf['recomputed_cost_km'],checked['recomputed_distance_km'],abs_tol=DISTANCE_ATOL_KM,rel_tol=DISTANCE_RTOL):fail.append('teacher_forcing_cost_mismatch')
                        else:fail.append(tf['reason'])
                    elif not teacher_force:fail.append('teacher_forcing_not_requested')
                    if not fail:
                        clean=[[int(node) for node in route] for route in routes]
                        canonical=json.dumps(clean,separators=(',',':'))
                        value=checked['recomputed_distance_km']
                        accepted.append(dict(instance_id=identity,status_name='VERIFIED',feasible=True,verified_feasible=True,
                            objective_distance_km=value,recorded_objective_distance_km=cost,vehicle_count=len(clean),
                            routes_json=canonical,route_sequence_json=json.dumps([0]+[node for route in clean for node in route[1:]],separators=(',',':')),
                            territory_id=area,source_split='train',instance_semantic_sha256=fp,
                            audit_sources_json=audit['sources_json'],recovered_cost=cost is None,route_source=source))
                        audit['accepted']=True
                except Exception as error:fail.append(f'{type(error).__name__}:{error}')
                audit['reasons_json']=json.dumps(sorted(set(fail)));reasons.update(set(fail));expert_rows.append(audit)
            if accepted:pool_rows.append(min(accepted,key=lambda row:(row['objective_distance_km'],row['routes_json'])))
            if progress and (index+1)%100==0:progress(dict(split=split,instances=index+1,accepted_experts=len(pool_rows)))
        data.update(territory_counts=dict(sorted(territory.items())),contract_failures=dict(contract_failures),duplicate_instance_ids=duplicates)
        data['count_matches_metadata']=data['declared_count'] is None or data['count']==data['declared_count']
        data['valid']=data['bundle_hash_matches_metadata'] and data['count_matches_metadata'] and not contract_failures and not duplicates
        if not data['valid']:report['errors'].append(f'{split}: failed data contract, IDs, hash or count')
    for identity,items in candidates.items():
        if identity not in visited_train:
            for item in items:
                expert_rows.append(dict(instance_id=identity,sources_json=json.dumps(item['sources']),accepted=False,
                    reasons_json=json.dumps(['expert_id_not_in_train'])))
                reasons['expert_id_not_in_train']+=1
    overlap_found=False
    for left,right in itertools.combinations(paths,2):
        values={'instance_ids':sorted(split_ids[left]&split_ids[right]),
                'semantic_sha256':sorted(split_fps[left]&split_fps[right]),
                'source_keys':sorted(split_source_keys[left]&split_source_keys[right])}
        report['split_overlaps'][f'{left}:{right}']={key:{'count':len(value),'values':value} for key,value in values.items()}
        if values['instance_ids'] or values['semantic_sha256'] or values['source_keys']:overlap_found=True
    if overlap_found:report['errors'].append('Split identity/semantic/source overlap detected')
    training_data_ready=all(report['splits'][s].get('valid',False) for s in ('train','val')) and not overlap_found
    if not training_data_ready:pool_rows=[]  # Never publish usable experts for a contaminated or invalid split.
    pool_rows.sort(key=lambda x:x['instance_id'])
    pool_path=out/'train_expert_pool.csv'
    pool_fields=['instance_id','status_name','feasible','verified_feasible','objective_distance_km','recorded_objective_distance_km',
        'vehicle_count','routes_json','route_sequence_json','territory_id','source_split','instance_semantic_sha256','audit_sources_json','recovered_cost','route_source']
    _write_csv(pool_path,pool_rows,pool_fields)
    _write_csv(out/'instance_index.csv',indices,['split','instance_id','territory_id','instance_key','semantic_sha256','source_key','dataset_version','contract_valid'])
    _write_csv(out/'expert_audit.csv',expert_rows,['instance_id','sources_json','recorded_cost_km','recomputed_cost_km','recovered_cost','route_source',
        'independent_valid','teacher_forcing_valid','teacher_forcing_cost_km','cost_consistent','accepted','reasons_json'])
    total=report['splits']['train']['count']
    counts={'train_instances':total,'with_recorded_incumbent':len(incumbent_ids),'with_complete_or_recovered_route_payload':len(route_ids),
        'independent_feasible':len(independent_ids),'teacher_forcing_feasible':len(tf_ids),'recorded_cost_consistent':len(cost_ids),
        'accepted_unique_experts':len(pool_rows),'cost_recovered_from_complete_route':sum(x['recovered_cost'] for x in pool_rows)}
    report['expert_pool']={'path':str(pool_path),'sha256':_hash(pool_path),'accepted_count':len(pool_rows),
        'coverage':len(pool_rows)/total if total else 0.,'train_only':True,'counts':counts,
        'fractions':{key:value/total if total else 0. for key,value in counts.items() if key!='train_instances'},
        'source_row_counts':source_rows,'missing_assets':missing_experts,'failure_reasons':dict(reasons),
        'selection':'One lowest independently validated finite-cost complete train route per instance; route JSON provides deterministic tie-break.',
        'teacher_forcing_required_for_acceptance':True}
    report.update(training_data_ready=training_data_ready,expert_methods_ready=training_data_ready and bool(pool_rows) and bool(teacher_force),
        test_data_ready=report['splits']['test'].get('valid',False) and not overlap_found,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    report['outputs']={name:{'path':str(out/name),'sha256':_hash(out/name)} for name in ('instance_index.csv','expert_audit.csv','train_expert_pool.csv')}
    _write_json(out/'assets_audit.json',report)
    return report
