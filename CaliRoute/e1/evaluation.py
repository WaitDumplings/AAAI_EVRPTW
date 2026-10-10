"""Independent K-attempt evaluation shared by E1 methods; failures remain rows."""
from __future__ import annotations
import copy
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from e1.validator import validate_routes, routes_from_sequence


def _finite(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (ValueError, TypeError):
        return None


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def select_candidates(instance, candidates, *, method, run_id, training_seed,
                      inference_seed, checkpoint_id, requested_k=50, decode_mode='sample',
                      timing_batch_size=1, batch_runtime_s=0.):
    if not candidates or len(candidates) > requested_k:
        raise ValueError('Candidate attempts must be nonempty and not exceed requested K')
    feasible = []; reasons = []
    for index, item in enumerate(candidates):
        checked = validate_routes(instance, item.get('routes', []), item.get('reported_cost'))
        ok = bool(item.get('completed') and not item.get('truncated') and checked.get('feasible'))
        # A cost mismatch is an implementation error, not a reason to hide a
        # feasible physical route. Preserve it and report the recomputed cost.
        if ok:
            feasible.append((checked['recomputed_distance_km'], index, item, checked))
        else:
            reasons.extend(checked.get('reasons', []))
            if item.get('truncated'): reasons.append('decode_cap_or_environment_truncation')
            if not item.get('completed'): reasons.append('incomplete_attempt')
            if item.get('failure_reason'): reasons.append(item['failure_reason'])
    best = min(feasible, key=lambda x: (x[0], x[1])) if feasible else None
    meta = instance.metadata or {}
    result = dict(method=method, run_id=run_id, training_seed=training_seed,
        inference_seed=inference_seed, instance_id=str(instance.instance_id),
        territory_id=str(meta.get('service_territory_id') or meta.get('source_territory_id') or instance.region_id),
        checkpoint_id=checkpoint_id, requested_K=requested_k, actual_K=len(candidates), decode_mode=decode_mode,
        completed=bool(best), feasible=bool(best),
        truncated=bool(not best and any(c.get('truncated') for c in candidates)),
        failure_reason=None if best else ';'.join(sorted(set(reasons))) or 'no_feasible_candidate',
        reported_cost=_finite(best[2].get('reported_cost')) if best else None,
        recomputed_cost_km=best[0] if best else None, num_routes=best[3]['num_routes'] if best else None,
        action_count=best[3]['action_count'] if best else None,
        runtime=float(batch_runtime_s)/timing_batch_size, batch_runtime_s=float(batch_runtime_s),
        timing_batch_size=timing_batch_size,
        timing_scope='in-memory environment/model preprocessing, encoding/SVD, all decoding attempts and route validation; excludes file IO, raw instance deserialization/adaptation and checkpoint loading',
        runtime_kind='single_instance_latency' if timing_batch_size==1 else 'batch_amortized',
        selected_route=best[2]['routes'] if best else None, selected_candidate=best[1] if best else None,
        candidate_feasible_count=len(feasible), candidate_completed_count=sum(bool(c.get('completed')) for c in candidates),
        candidate_truncated_count=sum(bool(c.get('truncated')) for c in candidates),
        selected_cost_matches_reported=best[3].get('distance_matches') if best else None,
        cost_mismatch_candidate_count=sum(c[3].get('distance_matches') is False for c in feasible))
    return result


def summarize_rows(rows):
    good = [r for r in rows if r['feasible']]
    costs = [r['recomputed_cost_km'] for r in good]
    return dict(instances=len(rows), feasible_count=len(good), failure_count=len(rows)-len(good),
        feasible_rate=len(good)/len(rows) if rows else 0.,
        mean_distance_km=float(np.mean(costs)) if costs else None,
        mean_vehicle_count=float(np.mean([r['num_routes'] for r in good])) if good else None,
        mean_runtime_s=float(np.mean([r['runtime'] for r in rows])) if rows else None,
        requested_K=sorted(set(r['requested_K'] for r in rows)), actual_K=sorted(set(r['actual_K'] for r in rows)),
        cost_mismatch_candidates=sum(r['cost_mismatch_candidate_count'] for r in rows),
        training_seeds=sorted(set(r['training_seed'] for r in rows)),
        statistical_scope='Single training seed. No between-training-seed standard deviation is estimated.')


def _sync(device):
    import torch
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def controlled_rows(agent, cfg, *, seed, checkpoint_id, device='cpu', split='val'):
    import torch
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations, encode_static_rollout, sample_actions
    from offline2online.instance_adapter import iter_adapted_instances
    evaluation = cfg['evaluation']; batch_size = evaluation['eval_batch_size']; k = evaluation['eval_n_traj']
    limit = evaluation.get('eval_limit'); cap = evaluation['eval_max_steps']
    if k != 50 and not cfg.get('experiment_protocol', {}).get('smoke_only'):
        raise ValueError('Formal E1 evaluation requires exactly K=50')
    data = iter_adapted_instances(evaluation['eval_path'], problem_type='cvrp', strict_road_metric=True, limit=limit)
    batches_limit = evaluation.get('eval_num_batches')
    seen = 0; batch_index = 0
    while True:
        instances = list(itertools.islice(data, batch_size))
        if not instances or (batches_limit is not None and batch_index >= batches_limit): break
        batch_index += 1
        _sync(device); started = time.perf_counter()
        envs = []
        try:
            for instance in instances:
                envs.append(make_terran_env(instance=instance, n_traj=k, **cfg['env']))
            observations=[];infos=[]
            for index,env in enumerate(envs):
                obs,info=env.reset(seed=seed+seen+index);observations.append(obs);infos.append(info)
            done=np.zeros((len(envs),k),dtype=bool);terminated=done.copy();truncated=done.copy()
            sequences=[[[0] for _ in range(k)] for _ in envs]
            static_cache={}; encoded=None
            with torch.no_grad():
                for step in range(cap):
                    obs=stack_observations(observations, static_cache=static_cache)
                    if step==0: encoded=encode_static_rollout(agent,obs)
                    actions,_,_,_,_=sample_actions(agent,obs,decode_mode=evaluation.get('eval_decode_mode','sample'),device=device,cached_embeddings=encoded)
                    actions=actions.cpu().numpy()
                    for index,env in enumerate(envs):
                        for trajectory in np.flatnonzero(~done[index]):
                            sequences[index][trajectory].append(int(actions[index,trajectory]))
                        obs,_,term,trunc,info=env.step(actions[index])
                        observations[index]=obs;infos[index]=info
                        terminated[index]|=np.asarray(term,dtype=bool);truncated[index]|=np.asarray(trunc,dtype=bool)
                    done=terminated|truncated
                    if done.all():break
            rows=[]
            for index,instance in enumerate(instances):
                costs=np.asarray(infos[index].get('objective_distance_km',[float('nan')]*k)).reshape(-1)
                candidates=[]
                for trajectory,sequence in enumerate(sequences[index]):
                    try:routes=routes_from_sequence(sequence)
                    except ValueError:routes=[]
                    candidates.append(dict(routes=routes, completed=bool(terminated[index,trajectory]),
                        truncated=bool(truncated[index,trajectory] or not done[index,trajectory]),
                        reported_cost=_finite(costs[trajectory])))
                row=select_candidates(instance,candidates,method=cfg['experiment_protocol']['method'],
                    run_id=cfg['run_name'],training_seed=cfg['experiment_protocol']['training_seed'],
                    inference_seed=seed,checkpoint_id=checkpoint_id,requested_k=k,
                    decode_mode=evaluation.get('eval_decode_mode','sample'),timing_batch_size=len(envs))
                row['split']=split;rows.append(row)
            _sync(device);elapsed=time.perf_counter()-started
            for row in rows:
                row['runtime']=elapsed/len(envs);row['batch_runtime_s']=elapsed
                yield row
            seen+=len(instances)
        finally:
            for env in envs:env.close()


def evaluate_controlled_agent(agent, cfg, seed, epoch, device):
    """Trainer hook; caller isolates inference RNG and restores agent mode."""
    agent.eval()
    evaluation=cfg['evaluation'];infer_seed=evaluation.get('eval_seed',17000000+seed)
    rows=list(controlled_rows(agent,cfg,seed=infer_seed,checkpoint_id=f'online_epoch_{epoch}',device=device))
    for row in rows:row['epoch']=epoch
    path=Path(evaluation['eval_output_dir'])/f'epoch_{epoch:04d}.jsonl'
    write_rows(path,rows)
    summary=summarize_rows(rows)
    def number(key):return summary[key] if summary[key] is not None else float('nan')
    trajs=sum(r['candidate_feasible_count'] for r in rows)
    return dict(eval_status='ok' if rows else 'no_instances',eval_num_instances=len(rows),
        eval_n_traj=evaluation['eval_n_traj'],eval_batch_size=evaluation['eval_batch_size'],
        eval_num_batches=math.ceil(len(rows)/evaluation['eval_batch_size']),
        eval_decode_mode=evaluation.get('eval_decode_mode','sample'),eval_info_level='independent_all_candidates',eval_save_routes=True,
        eval_feasible_rate=summary['feasible_rate'],eval_avg_objective_distance_km=number('mean_distance_km'),
        eval_avg_min_objective_distance_km=number('mean_distance_km'),
        eval_avg_median_objective_distance_km=float('nan'),
        eval_avg_vehicle_count=number('mean_vehicle_count'),eval_avg_min_vehicle_count=number('mean_vehicle_count'),
        eval_avg_median_vehicle_count=float('nan'),eval_avg_runtime_s=number('mean_runtime_s'),
        eval_traj_feasible_rate=trajs/(len(rows)*evaluation['eval_n_traj']) if rows else 0.,
        eval_avg_feasible_traj_count=trajs/len(rows) if rows else 0.,eval_instances_path=str(path))


def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as handle:
        for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    temporary.replace(path)
    atomic_json(path.with_suffix('.summary.json'),summarize_rows(rows))


def evaluate_checkpoint(config, checkpoint, output_path, *, test_path, device='cpu', batch_size=1):
    """Final controlled test, loading the selected checkpoint without training."""
    import torch
    from offline2online.models import Agent
    from offline2online.input_normalization import configure as configure_inputs
    from offline2online.model_integration import configure as configure_integration
    from offline2online.trainer import _load_agent_checkpoint, _isolated_eval_rng
    cfg=copy.deepcopy(config)
    cfg['evaluation'].update(eval_path=str(test_path),eval_batch_size=batch_size,eval_n_traj=50,eval_limit=None,eval_num_batches=None)
    parameters={k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters}
    agent=Agent(device=device,**parameters).to(device)
    configure_inputs(agent,cfg);configure_integration(agent,cfg)
    _load_agent_checkpoint(agent,checkpoint,device=device,strict=True)
    agent.eval()
    checkpoint_id=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    seed=cfg['evaluation']['eval_seed']
    with _isolated_eval_rng(seed,device):
        rows=list(controlled_rows(agent,cfg,seed=seed,checkpoint_id=checkpoint_id,device=device,split='test'))
    write_rows(output_path,rows)
    return summarize_rows(rows)
