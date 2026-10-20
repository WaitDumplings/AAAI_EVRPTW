"""Prepare, verify and execute E2-B B01/B04, with separate smoke evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import torch
from caliroute.methods import method_preset
from e1.native import atomic_json,file_hash,json_hash
from e1.native_worker import evaluate_runner,now
from e2.rrnco_native import prepare_variant,verify_run
from e2.rrnco_ppo import PPOExperiment,COMMON

BASE=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2/runs/rrnco')
EXPERT=Path('/data/Maojie/AAAI/AAAI_Dataset/dataset/vrptw/train/Cus100/expert_solutions.csv')


def prepare(root,campaign,method,*,preflight=False,small=False):
    run=prepare_variant(root,BASE,campaign,method,preflight=preflight)
    cfg=json.loads((run/'native_resolved.json').read_text());manifest=json.loads((run/'native_manifest.json').read_text())
    protocol=deepcopy(COMMON)
    if small:protocol.update(batch_size=4,n_traj=3,num_minibatches=2,microbatch_instances=2)
    cfg.update(e2b_method=method,run_id=f'E2B_RRNCO_VRPTW100_{method.upper()}_S3009',
        ppo_protocol=protocol,batch_size=protocol['batch_size'],n_traj=protocol['n_traj'],
        expert_sha256=None,preflight_result=str(campaign/'preflight'/method/'preflight.json'))
    cfg['experiment_protocol'].update(panel='E2-B',modules=[],training_objective='per_action_PPO_GAE',
        online_rollout_iterations=1500,total_instance_exposures=96000,ppo_passes=4)
    if method in {'slppo','awbc','dapg'}:
        shutil.copy2(EXPERT,run/'expert_pool.csv');cfg['expert_sha256']=file_hash(run/'expert_pool.csv')
        preset=method_preset(method);offline=preset.offline_config()
        offline.update(solution_reference_contract='e1_cost_incumbent_v1',policy_replay_enabled=False,
            branch_exploration_enabled=False,exploration_enabled=False,use_priority_sampler=False,strict_replay=True)
        if method in {'awbc','dapg'}:offline.update(expert_microbatch_size=32)
        cfg['e1_method_config']=dict(offline=offline,advantage=preset.advantage_config(),
            training=dict(reward_norm_mode='legacy'),data=dict(num_customers=100))
    worker=root/'scripts/run_e2b_rrnco_worker.py';dest=run/'native_harness/scripts'/worker.name
    shutil.copy2(worker,dest)
    manifest['harness']['files'].append(dict(path='scripts/'+worker.name,sha256=file_hash(dest)))
    manifest['harness']['sha256']=json_hash(manifest['harness']['files']);manifest['config_sha256']=json_hash(cfg)
    atomic_json(run/'native_resolved.json',cfg);atomic_json(run/'native_manifest.json',manifest)
    return run


def evaluate(exp,*,split='val',small=False,checkpoint_id='preflight'):
    cfg=deepcopy(exp.cfg)
    cfg['eval_batch_size']=1 if split=='test' else 8
    if small:
        cfg.update(eval_limit=2,eval_k=3,eval_batch_size=2)
        cfg['experiment_protocol'].update(smoke_only=True,hardware_preflight_only=True)
    target=exp.run/('test' if split=='test' else 'evaluations');target.mkdir(exist_ok=True)
    output=target/('instances.jsonl' if split=='test' else f'iteration_{exp.iteration:04d}_{checkpoint_id[:12]}.jsonl')
    result=evaluate_runner(exp.runner,cfg,exp.run,split=split,checkpoint_id=checkpoint_id,
        output=output,inference_seed=17003009)
    exp.evaluation_wall_s+=result['wall_time_s'];exp.total_active_wall_s+=result['wall_time_s']
    return result


def preflight(run):
    cfg,manifest=verify_run(run);exp=PPOExperiment(cfg,run,manifest)
    expert_audit=None
    if exp.experts is not None:
        from e1.native_worker import rng_state,restore_rng
        from e2.expert_audit import audit_expert_completion
        saved_rng=rng_state()
        expert_audit=audit_expert_completion(exp)
        restore_rng(saved_rng)
    imitation_audit=exp.imitation.audit() if exp.imitation else None
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    warmup_recovery=None
    if exp.warmup_target():
        exp.warmup_step();warm_checkpoint=exp.save('warmup.pt')
        w1=exp.warmup_step();weights={k:v.clone() for k,v in exp.runner.model.state_dict().items()}
        exp.restore(warm_checkpoint['path']);w2=exp.warmup_step()
        difference=max(float((v-exp.runner.model.state_dict()[k]).abs().max()) for k,v in weights.items())
        if w1['expert_indices_sha256']!=w2['expert_indices_sha256'] or difference>2e-5:raise ValueError('Warmup resume differs')
        warmup_recovery=dict(passed=True,parameter_max_difference=difference)
        while exp.warmup_updates<exp.warmup_target():exp.warmup_step()
    first=exp.iteration_step(audit=True);saved=exp.save('last.pt')
    # Continue once, then restore exactly and repeat the same online sampling,
    # minibatch order, external neighborhood samples and method state update.
    second=exp.iteration_step(audit=False)
    state_after={k:v.clone() for k,v in exp.runner.model.state_dict().items()}
    memory_after=dict(exp.incumbent)
    exp.restore(saved['path']);repeat=exp.iteration_step(audit=False)
    if second['instance_indices_sha256']!=repeat['instance_indices_sha256'] or memory_after!=exp.incumbent:
        raise ValueError('Resume sampler/incumbent differs')
    if exp.imitation and second['method_diagnostics']['expert_sample_indices_sha256']!=repeat['method_diagnostics']['expert_sample_indices_sha256']:raise ValueError('Expert sampler resume differs')
    maxdiff=max(float((v-exp.runner.model.state_dict()[k]).abs().max()) for k,v in state_after.items())
    # Native CUDA indexing backward has small nondeterministic reductions.
    if maxdiff>2e-5:raise ValueError(('Resume parameter difference',maxdiff))
    result=evaluate(exp,small=True)
    if result['feasible_instances']!=2 or result['cost_mismatch_candidates']:
        raise ValueError('Physical preflight validation failed')
    exp.best=(result['feasible_instances'],-result['mean_distance_km'])
    checkpoint=exp.save('best.pt');exp.restore(checkpoint['path'])
    again=evaluate(exp,small=True,checkpoint_id='reload')
    if result['mean_distance_km']!=again['mean_distance_km']:raise ValueError('Reload validation differs')
    report=dict(passed=True,method=cfg['e2b_method'],protocol=exp.p,imitation_audit=imitation_audit,
        warmup_recovery=warmup_recovery,warmup_updates=exp.warmup_updates,warmup_expert_steps=exp.warmup_expert_steps,warmup_wall_s=exp.warmup_wall_s,
        harness_sha256=manifest['harness']['sha256'],expert_sha256=cfg['expert_sha256'],
        actor_initial_sha256=exp.actor_initial,critic_initial_sha256=exp.critic_initial,
        first_iteration=first,second_iteration=second,resumed_iteration=repeat,
        resume_parameter_max_difference=maxdiff,validation=result,reload_validation=again,
        max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved(),
        elapsed_s=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),finished_at=now())
    if exp.experts is not None:
        report['e1_unchanged_source']=exp.e1_evidence
        report['expert_completion_audit']=expert_audit
    atomic_json(run/'preflight.json',report);print(json.dumps(report,indent=2),flush=True)


def formal(run,resume=False):
    cfg,manifest=verify_run(run)
    pre=json.loads(Path(cfg['preflight_result']).read_text())
    if not pre['passed'] or pre['protocol']!=cfg['ppo_protocol'] or pre['harness_sha256']!=manifest['harness']['sha256']:
        raise ValueError('Matching full protocol preflight required')
    if not resume and (run/'train.jsonl').exists():raise FileExistsError('Explicit resume required')
    exp=PPOExperiment(cfg,run,manifest)
    if (exp.actor_initial,exp.critic_initial)!=(pre['actor_initial_sha256'],pre['critic_initial_sha256']):raise ValueError('Initialization changed')
    if resume:exp.restore(run/'native_checkpoints/last.pt')
    resumed_active_s=exp.total_active_wall_s;active_started=time.perf_counter()
    atomic_json(run/'architecture.json',exp.runner.architecture)
    while exp.warmup_updates<exp.warmup_target():
        row=exp.warmup_step()
        with (run/'warmup.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        exp.total_active_wall_s=resumed_active_s+time.perf_counter()-active_started
        exp.save('last.pt')
        atomic_json(run/'status.json',dict(state='bc_warmup',method=cfg['e2b_method'],warmup_update=exp.warmup_updates,warmup_target=exp.warmup_target(),training_complete=False,test_complete=False,updated_at=now()))
        print(f"[{cfg['e2b_method']}] BC warmup={exp.warmup_updates}/{exp.warmup_target()}",flush=True)
    while exp.iteration<exp.p['online_rollout_iterations']:
        row=exp.iteration_step()
        with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if exp.iteration%exp.p['eval_every_iterations']==0:
            val=evaluate(exp,checkpoint_id=f'iter{exp.iteration}')
            metric=(val['feasible_instances'],-val['mean_distance_km'] if val['mean_distance_km'] is not None else -1e300)
            val.update(iteration=exp.iteration,instance_exposures=exp.iteration*exp.p['batch_size'])
            with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(val,allow_nan=False)+'\n')
            if exp.best is None or metric>tuple(exp.best):exp.best=metric;exp.save('best.pt')
        exp.total_active_wall_s=resumed_active_s+time.perf_counter()-active_started
        if exp.iteration%exp.p['checkpoint_every_iterations']==0:exp.save('last.pt')
        status=dict(state='training',method=cfg['e2b_method'],training_complete=False,test_complete=False,
            counters=exp.method_state(),updated_at=now())
        # Full scalar incumbent mapping stays in the checkpoint, not status.
        status['counters'].pop('incumbent');status['counters'].pop('online_rng');status['counters'].pop('minibatch_rng')
        atomic_json(run/'status.json',status)
        print(f"[{cfg['e2b_method']}] iteration={exp.iteration}/1500 optimizer_steps={exp.optimizer_steps} best_train_km={row['mean_best_cost_km']:.4f} KL={row['kl_max']:.6f}",flush=True)
    status.update(state='testing',training_complete=True,
        active_training_including_validation_s=resumed_active_s+time.perf_counter()-active_started)
    atomic_json(run/'status.json',status)
    best=run/'native_checkpoints/best.pt';exp.restore(best)
    result=evaluate(exp,split='test',checkpoint_id=file_hash(best))
    target=run/'test';rows=[json.loads(x) for x in (target/'instances.jsonl').read_text().splitlines()]
    if len(rows)!=1000 or len({r['instance_id'] for r in rows})!=1000 or any(r['actual_K']!=50 or r['cost_mismatch_candidate_count'] for r in rows):
        raise ValueError('Incomplete or invalid test inventory')
    atomic_json(target/'instances.summary.json',result)
    atomic_json(target/'manifest.json',dict(accepted=True,method=cfg['e2b_method'],panel='E2-B',
        training_complete=True,checkpoint=str(best),checkpoint_sha256=file_hash(best),
        instances_sha256=file_hash(target/'instances.jsonl'),test_data_sha256=manifest['datasets']['test']['source_sha256'],
        protocol_sha256=json_hash(exp.p),result=result))
    status.update(state='completed',test_complete=True,test_accepted=True,test_result=result,updated_at=now())
    atomic_json(run/'status.json',status)
    update_table(run.parent.parent)


def update_table(campaign):
    import csv,fcntl
    from io import StringIO
    lock=(campaign/'table.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    completed={}
    for method in ['ppo','awbc','dapg','slppo']:
        run=campaign/'runs'/method;p=run/'test/manifest.json'
        if p.exists():
            m=json.loads(p.read_text())
            if m.get('accepted'):
                instances={r['instance_id']:r for r in [json.loads(x) for x in (run/'test/instances.jsonl').read_text().splitlines()]}
                completed[method]=(m,instances,json.loads((run/'status.json').read_text()))
    rows=[]
    for number,method in [('B01','ppo'),('B02','awbc'),('B03','dapg'),('B04','slppo')]:
        row=dict(id=number,method=method,status='TBD',feasible='TBD',mean_km='TBD',
            common_with_ppo='N/A' if method=='ppo' else 'TBD',common_ppo_km='N/A' if method=='ppo' else 'TBD',
            common_method_km='N/A' if method=='ppo' else 'TBD',paired_improvement_pct='N/A' if method=='ppo' else 'TBD',
            inference_s_per_instance='TBD',training_h='TBD')
        if method in completed:
            m,instances,status=completed[method];result=m['result']
            row.update(status='test accepted',feasible=result['feasible_instances'],mean_km=result['mean_distance_km'],
                inference_s_per_instance=result['mean_runtime_s'],training_h=status['active_training_including_validation_s']/3600)
            if method!='ppo' and 'ppo' in completed:
                base=completed['ppo'][1]
                common=[i for i,x in instances.items() if x['feasible'] and base[i]['feasible'] and base[i]['recomputed_cost_km']>0]
                if common:
                    row.update(common_with_ppo=len(common),common_ppo_km=sum(base[i]['recomputed_cost_km'] for i in common)/len(common),
                        common_method_km=sum(instances[i]['recomputed_cost_km'] for i in common)/len(common),
                        paired_improvement_pct=sum(100*(base[i]['recomputed_cost_km']-instances[i]['recomputed_cost_km'])/base[i]['recomputed_cost_km'] for i in common)/len(common))
        rows.append(row)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary=campaign/'E2B_results.csv.tmp';temporary.write_text(stream.getvalue());os.replace(temporary,campaign/'E2B_results.csv')
    lock.close()
