"""Controlled VRPTW/EVRPTW campaign preparation; reuses E1 models, losses and lifecycle.

No modifications to existing frozen campaigns. Native baselines keep their
separate launchers. Test is inspected for identity only during preparation.
"""
from pathlib import Path
import csv
import json
import math
import shutil

import numpy as np
import yaml
from e1.configs import ROOT, CONTROLLED, EXPERT_METHODS, build_config, content_hash, digest
from e1.evaluation import atomic_json
from e1.runner import _runtime, verify_campaign


def configuration(method, data, run, unit, backup, customers, world_size, ppo_chunk, expert_chunk, epochs=1500, *, problem_type="vrptw"):
    cfg = build_config(method, data_root=data, output_dir=run, distance_unit_km=unit,
        backup_root=backup, customers=customers, world_size=world_size, epochs=epochs,
        expert_pool=run.parent.parent/'assets/train_expert_pool.csv' if method in EXPERT_METHODS else None,
        ppo_chunk=ppo_chunk, expert_chunk=expert_chunk)
    if problem_type not in ('vrptw','evrptw'):raise ValueError('Unsupported routing task')
    cfg['dataset_name'] = f'Geo-{problem_type.upper()}-v1'
    cfg['run_name'] = f'{run.parent.parent.name}_{method}'
    cfg['data'].update(problem_type=problem_type, train_dataset_path=str(data/f'dataset/{problem_type}/train/Cus{customers}'))
    cfg['evaluation']['eval_path'] = str(data/f'dataset/{problem_type}/val/Cus{customers}')
    if method in EXPERT_METHODS:
        cfg['offline']['expert_dataset_path'] = cfg['data']['train_dataset_path']
    if problem_type=='evrptw':
        if method not in CONTROLLED:raise ValueError('Unsupported EVRPTW method')
        if method in ('awbc','dapg') and customers>=50:
            cfg['offline']['expert_microbatch_size']=64
            cfg['experiment_protocol']['expert_microbatch_size']=64
        stations={15:3,50:10,100:20}[customers]
        cap=4*(1+customers+stations)
        cfg['data']['num_charging_stations']=stations
        cfg['training']['rollout_steps']=cap
        cfg['evaluation']['eval_max_steps']=cap
        cfg['experiment_protocol'].update(decode_cap=cap,charging_mode='fixed_full',physical_charging_stations=stations)
    if method == 'ppo_base':
        cfg['model']['e1_no_road_task'] = problem_type
        cfg['experiment_protocol']['base_preserves'] = ['coordinates','demand','capacity','time_windows','service_time','current_clock','history','physical_feasibility_mask']
        if problem_type=='evrptw':
            cfg['experiment_protocol']['base_preserves'] += ['charging_station_coordinates','current_battery','remaining_battery','battery_capacity','full_charge_time','fixed_full_charge']
    cfg['experiment_protocol'].update(schema=f'aaai_e1_{problem_type}{customers}_v1', task=f'{problem_type}{customers}',
                                     actor_sha256=content_hash(cfg['model']))
    return cfg


def audit(data, folder, customers, include_experts, *, expected_counts=None, problem_type="vrptw"):
    from e1.native import export_native_dataset
    if problem_type=="evrptw":
        from e1.evrptw_contract import export_audit_dataset as export_native_dataset
    from e1.assets import _expert_candidates, _write_csv
    from e1.validator import extract_routes, validate_routes, teacher_force_routes
    from offline2online.instance_adapter import iter_adapted_instances
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(schema=f'aaai_e1_{problem_type}_assets_v1', problem=problem_type, customers=customers,
                  data_root=str(data), splits={}, test_policy='Identity/physical contract audit only; no test solutions read')
    indices=[]; snapshots={}; inputs=[]
    expected_counts = expected_counts or {'train':5000,'val':1000,'test':1000}
    for split, expected in expected_counts.items():
        path=data/('test_release' if split=='test' else 'dataset')/problem_type/split/f'Cus{customers}'
        if not (path/'instances.pkl').is_file():raise FileNotFoundError(path/'instances.pkl')
        temporary=folder/f'{split}.npz'
        item=export_native_dataset(path,temporary,split=split,expected_customers=customers,problem_type=problem_type)
        if item['count'] != expected:raise ValueError(f'{split}: expected {expected} instances, got {item["count"]}')
        if split=='train':
            with np.load(temporary) as arr:
                d=arr['distance_matrix'];unit=float(np.median(d[:,0,1:customers+1]+d[:,1:customers+1,0]))
            if not np.isfinite(unit) or unit<=0:raise ValueError('Invalid train-only distance unit')
        temporary.unlink()
        files={}
        for name in ('instances.pkl','metadata.json') + (('expert_solutions.csv','gurobi_summary.csv') if split=='train' and include_experts else ()):
            f=path/name
            if f.exists():files[name]=dict(path=str(f),sha256=digest(f));inputs.append(files[name])
        report['splits'][split]=dict(path=str(path),count=item['count'],valid=True,files=files)
        snapshots[split]=item
        indices.extend(dict(split=split,**r) for r in item['instances'])
    for left,right in [('train','val'),('train','test'),('val','test')]:
        for key in ('instance_id','semantic_sha256'):
            if {r[key] for r in snapshots[left]['instances']} & {r[key] for r in snapshots[right]['instances']}:
                raise ValueError(f'{left}/{right} overlap by {key}')
    pool=[]; audit_rows=[]
    if include_experts:
        train=Path(report['splits']['train']['path']); candidates,counts,missing=_expert_candidates(train)
        known=set()
        identities={r['instance_id']:r for r in snapshots['train']['instances']}
        for index,instance in enumerate(iter_adapted_instances(train,problem_type=problem_type,strict_road_metric=True),1):
            identity=str(instance.instance_id);known.add(identity);accepted=[]
            for item in candidates.get(identity,[]):
                raw=item['row'];entry=dict(instance_id=identity,accepted=False,sources_json=json.dumps(item['sources']))
                try:
                    value=raw.get('objective_distance_km');cost=float(value) if value not in (None,'') else None
                    if cost is not None and (not math.isfinite(cost) or cost<0):raise ValueError('Invalid expert cost')
                    routes,_=extract_routes(raw);checked=validate_routes(instance,routes,cost)
                    if not checked['valid']:raise ValueError(str(checked['reasons']))
                    forced=teacher_force_routes(instance,routes,dict(normalize_reward=True,reward_contract='legacy',reward_mode='distance',max_steps_factor=4,**({'prefer_explicit_edge_matrices':True,'charging_mode':'fixed_full'} if problem_type=='evrptw' else {})))
                    if not forced['valid'] or not math.isclose(forced['recomputed_cost_km'],checked['recomputed_distance_km'],rel_tol=1e-6,abs_tol=1e-5):
                        raise ValueError('Expert environment replay/cost mismatch')
                    accepted.append(dict(instance_id=identity,status_name='VERIFIED',feasible=True,verified_feasible=True,
                        objective_distance_km=checked['recomputed_distance_km'],routes_json=json.dumps(routes,separators=(',',':')),
                        route_sequence_json=json.dumps([0]+[n for route in routes for n in route[1:]]),
                        source_split='train',instance_semantic_sha256=identities[identity]['semantic_sha256']))
                    entry['accepted']=True
                except (ValueError,TypeError,KeyError,RuntimeError) as error:entry['reason']=str(error)
                audit_rows.append(entry)
            if accepted:pool.append(min(accepted,key=lambda r:(r['objective_distance_km'],r['routes_json'])))
            if index%100==0:print(json.dumps(dict(stage='expert_audit',customers=customers,instances=index,accepted=len(pool))),flush=True)
        for identity in set(candidates)-known:
            audit_rows.append(dict(instance_id=identity,accepted=False,reason='expert_id_not_in_train'))
        if not pool:raise ValueError('No independently validated and teacher-forced train experts')
        report['expert_sources']=dict(counts=counts,missing=missing)
    _write_csv(folder/'train_expert_pool.csv',sorted(pool,key=lambda r:r['instance_id']),['instance_id','status_name','feasible','verified_feasible','objective_distance_km','routes_json','route_sequence_json','source_split','instance_semantic_sha256'])
    _write_csv(folder/'expert_audit.csv',audit_rows,['instance_id','accepted','reason','sources_json'])
    _write_csv(folder/'instance_index.csv',indices,['split','instance_id','semantic_sha256','territory_id'])
    report.update(training_data_ready=True,test_data_ready=True,expert_methods_ready=bool(pool),
        expert_pool=dict(path=str(folder/'train_expert_pool.csv'),sha256=digest(folder/'train_expert_pool.csv'),accepted_count=len(pool),coverage=len(pool)/report['splits']['train']['count'],train_only=True,teacher_forcing_required_for_acceptance=True),
        distance_unit=dict(km=unit,fit_split='train'),
        outputs={name:dict(path=str(folder/name),sha256=digest(folder/name)) for name in ('instance_index.csv','expert_audit.csv','train_expert_pool.csv')})
    atomic_json(folder/'assets_audit.json',report)
    return report, inputs


def prepare(folder, data, customers, server, assignments, chunks, expert_chunk, *, problem_type="vrptw"):
    if folder.exists():
        manifest=verify_campaign(folder)
        actual={m:s['gpus'] for m,s in manifest['methods'].items()}
        if actual!=assignments or manifest['schema']!=f'aaai_e1_{problem_type}{customers}_v1':raise ValueError('Existing campaign placement/scale differs')
        return manifest
    if customers not in (15,50,100) or not assignments or any(m not in CONTROLLED for m in assignments):raise ValueError('Invalid scale/methods')
    cards=[g for gs in assignments.values() for g in gs]
    if len(cards)!=len(set(cards)) or min(cards)<0:raise ValueError('Concurrent methods require distinct GPUs')
    folder.mkdir(parents=True)
    report,inputs=audit(data,folder/'assets',customers,any(m in EXPERT_METHODS for m in assignments),problem_type=problem_type)
    code=folder/'source/CaliRoute';source=_runtime().source_snapshot(code,include_initialization_assets=False)
    (code/'results').unlink();(folder/'artifacts').mkdir();(code/'results').symlink_to(folder/'artifacts',target_is_directory=True)
    backup=folder.parent/'checkpoint_backups'/folder.name;methods={}
    for method,gpus in assignments.items():
        run=folder/'runs'/method;run.mkdir(parents=True)
        cfg=configuration(method,data,run,report['distance_unit']['km'],backup,customers,len(gpus),chunks[method],expert_chunk,problem_type=problem_type)
        cfg['experiment_protocol'].update(source_content_sha256=source['content_sha256'],data_audit_sha256=digest(folder/'assets/assets_audit.json'),
            expert_pool_sha256=report['expert_pool']['sha256'] if method in EXPERT_METHODS else None)
        config=run/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        methods[method]=dict(config=str(config),config_sha256=digest(config),run_dir=str(run),gpus=gpus,server=server,
                             requires_expert=method in EXPERT_METHODS,ready=method not in EXPERT_METHODS or report['expert_methods_ready'])
    for name in ('assets_audit.json','instance_index.csv','expert_audit.csv','train_expert_pool.csv'):
        f=folder/'assets'/name;inputs.append(dict(path=str(f),sha256=digest(f)))
    manifest=dict(schema=f'aaai_e1_{problem_type}{customers}_v1',campaign=str(folder),code_root=str(code),artifacts_root=str(folder/'artifacts'),
        source=source,training_seed=3009,initialization='scratch',methods=methods,backup_root=str(backup),inputs=inputs,
        audit_path=str(folder/'assets/assets_audit.json'),distance_unit=report['distance_unit'],
        server_plan={server:[[m,gs] for m,gs in assignments.items()]},
        comparison_contract=dict(global_batch=64,n_traj=50,ppo_passes=4,online_loops=1500,eval_K=50,
            total_instance_exposures=96000,task=problem_type,customers=customers,training_seed=3009,
            train_sha256=report['splits']['train']['files']['instances.pkl']['sha256']))
    atomic_json(folder/'manifest.json',manifest)
    return manifest
