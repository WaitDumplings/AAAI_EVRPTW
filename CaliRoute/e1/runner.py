"""Explicit E1 lifecycle: audit/prepare/smoke/queue/resume/test/summary.

Preparation and smoke do not launch long jobs. Launch defaults to a dry run;
--execute starts a detached scheduler which waits for idle GPUs without killing
other jobs. The frozen campaign is the source of truth after preparation.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import time

import yaml
from e1.configs import ROOT, METHODS, CONTROLLED, EXPERT_METHODS, SERVER_PLANS, SCHEMA, build_config, digest, fit_train_distance_unit, content_hash
from e1.evaluation import atomic_json


def _read(path):return json.loads(Path(path).read_text())
def _yaml(path):return yaml.safe_load(Path(path).read_text())
def _utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def _dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(value,sort_keys=False))


def _runtime():
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_reward_norm_comparison
    return run_reward_norm_comparison


def verify_campaign(path, *, data=True):
    path=Path(path).resolve();manifest=_read(path/'manifest.json')
    if manifest['schema']!=SCHEMA:raise ValueError('Unexpected E1 campaign schema')
    for name,expected in manifest['source']['files'].items():
        if digest(Path(manifest['code_root'])/name)!=expected:raise ValueError(f'Frozen source changed: {name}')
    for method,spec in manifest['methods'].items():
        if digest(spec['config'])!=spec['config_sha256']:raise ValueError(f'Frozen config changed: {method}')
    if data:
        for item in manifest['inputs']:
            if digest(item['path'])!=item['sha256']:raise ValueError(f'Frozen data/expert asset changed: {item["path"]}')
    return manifest


def prepare(args):
    report=_read(args.audit)
    if not report.get('training_data_ready'):raise ValueError('Train/validation data audit has not passed')
    if report['customers']!=100 or report['problem']!='cvrp':raise ValueError('This E1 definition is CVRP100 only')
    output=Path(args.campaign).resolve()
    if output.exists():raise FileExistsError('Use a new campaign directory; frozen configs are never overwritten')
    data=Path(report['data_root']).resolve()
    unit=fit_train_distance_unit(data/'dataset/cvrp/train/Cus100')
    inputs=[]
    # Test payload hashes record split identity only. No test reference/route file
    # enters this manifest or is opened by training.
    for split in ('train','val','test'):
        for name,item in report['splits'][split]['files'].items():
            if split=='test' and name not in ('instances.pkl','metadata.json'):continue
            if digest(item['path'])!=item['sha256']:raise ValueError(f'Audit input changed: {item["path"]}')
            inputs.append(item)
    pool=report['expert_pool']
    if digest(pool['path'])!=pool['sha256']:raise ValueError('Audited expert pool changed')
    output.mkdir(parents=True)
    assets=output/'assets';assets.mkdir()
    shutil.copy2(args.audit,assets/'assets_audit.json')
    for asset_name in ('instance_index.csv', 'expert_audit.csv'):
        original = Path(args.audit).parent / asset_name
        expected = report['outputs'][asset_name]['sha256']
        if digest(original) != expected: raise ValueError(f'Audit artifact changed: {asset_name}')
        shutil.copy2(original, assets/asset_name)
        inputs.append(dict(path=str(assets/asset_name),sha256=expected))
    shutil.copy2(pool['path'],assets/'train_expert_pool.csv')
    expert=assets/'train_expert_pool.csv'
    inputs.append(dict(path=str(expert),sha256=digest(expert)))
    inputs.append(dict(path=str(assets/'assets_audit.json'),sha256=digest(assets/'assets_audit.json')))
    atomic_json(assets/'distance_unit.json',unit)
    inputs.append(dict(path=str(assets/'distance_unit.json'),sha256=digest(assets/'distance_unit.json')))
    code=output/'source/CaliRoute'
    source=_runtime().source_snapshot(code,include_initialization_assets=False)
    # Isolate checkpoints/logs even if the shared snapshot utility normally links
    # all old launchers to the repository-level results tree.
    (code/'results').unlink();(output/'artifacts').mkdir();(code/'results').symlink_to(output/'artifacts',target_is_directory=True)
    backup=Path(args.backup_root).resolve() if args.backup_root else output.parent/'checkpoint_backups'/output.name
    if backup==output or output in backup.parents:raise ValueError('Checkpoint backup directory must be independent of the campaign tree')
    methods={}
    placement={method:(server,gpus) for server,items in SERVER_PLANS.items() for method,gpus in items}
    for method in METHODS:
        server,gpus=placement[method];run=output/'runs'/method;run.mkdir(parents=True)
        cfg=build_config(method,data_root=data,output_dir=run,expert_pool=expert,
            distance_unit_km=unit['km'],seed=args.seed,world_size=len(gpus),epochs=args.epochs,
            backup_root=backup,ppo_chunk=args.ppo_chunk,expert_chunk=args.expert_chunk,eval_batch=args.eval_batch)
        # Trainer names must remain unique across campaigns and remote results copies.
        unique=f'{output.name}_{method}'
        if method in CONTROLLED:cfg['run_name']=unique
        else:cfg['run_id']=unique
        cfg['experiment_protocol']['source_content_sha256']=source['content_sha256']
        cfg['experiment_protocol']['data_audit_sha256']=digest(assets/'assets_audit.json')
        cfg['experiment_protocol']['expert_pool_sha256']=pool['sha256'] if method in EXPERT_METHODS else None
        path=run/'config.yaml';_dump(path,cfg)
        methods[method]=dict(config=str(path),config_sha256=digest(path),run_dir=str(run),
            server=server,gpus=gpus,requires_expert=method in EXPERT_METHODS,
            ready=method not in EXPERT_METHODS or report['expert_methods_ready'])
    manifest=dict(schema=SCHEMA,created_at=_utc(),campaign=str(output),code_root=str(code),artifacts_root=str(output/'artifacts'),source=source,
        initialization='scratch',training_seed=args.seed,inputs=inputs,methods=methods,
        audit_path=str(assets/'assets_audit.json'),distance_unit=unit,backup_root=str(backup),
        server_plan=SERVER_PLANS,stage='prepared',description='No training launched; each method needs smoke and full hardware preflight')
    with expert.open() as handle:
        expert_semantics=sorted((r['instance_id'],float(r['objective_distance_km']),json.loads(r['routes_json'])) for r in csv.DictReader(handle))
    manifest['comparison_contract']=dict(schema=SCHEMA,training_seed=args.seed,online_loops=args.epochs,global_batch=64,
        total_instance_exposures=args.epochs*64,n_traj=50,controlled_ppo_passes=4,gamma=.99,eval_K=50,
        source_content_sha256=source['content_sha256'],expert_pool_semantic_sha256=content_hash(expert_semantics),
        dataset_sha256={split:report['splits'][split]['files']['instances.pkl']['sha256'] for split in ('train','val','test')})
    atomic_json(output/'manifest.json',manifest)
    print(json.dumps(dict(campaign=str(output),stage='prepared',methods=list(methods),server_plan=SERVER_PLANS),indent=2))
    return manifest


def _worker_command(manifest,method,config,*,gpus=None,resume=None,device='cpu'):
    command=[sys.executable]
    if gpus and len(gpus)>1:
        command+=['-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpus))]
    command+=['-m','e1.worker','--config',str(config),'--method',method,'--device',device]
    if resume:command+=['--resume',str(resume)]
    return command


def _process_env(manifest,gpus=None):
    return dict(os.environ,PYTHONPATH=manifest['code_root'],CUDA_VISIBLE_DEVICES=','.join(map(str,gpus or [])),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def _checkpoint_dir(manifest,cfg):
    return Path(manifest['code_root'])/'results/checkpoints'/f'Cus_{cfg["data"]["num_customers"]}_CS_0'/cfg['run_name']/f'seed_{cfg["experiment_protocol"]["training_seed"]}'


def _small_assets(manifest,method):
    from offline2online.instance_adapter import iter_instance_payloads
    run=Path(manifest['methods'][method]['run_dir'])/'smoke'
    run.mkdir(exist_ok=False)
    cfg=_yaml(manifest['methods'][method]['config'])
    train=cfg.get('data',{}).get('train_dataset_path',cfg.get('train_path'))
    val=cfg.get('evaluation',{}).get('eval_path',cfg.get('val_path'))
    identities=[]
    for split,path,count in (('train',train,8),('val',val,2)):
        folder=run/split;folder.mkdir();rows=[]
        for row in iter_instance_payloads(path):
            rows.append(row)
            if len(rows)==count:break
        with (folder/'instances.pkl').open('wb') as stream:pickle.dump({'instances':rows},stream)
        if split=='train':identities=[str(r['instance_id']) for r in rows]
    expert=run/'train_expert_pool.csv'
    if method in EXPERT_METHODS:
        with (Path(manifest['campaign'])/'assets/train_expert_pool.csv').open() as src,expert.open('w') as dst:
            reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
            for row in reader:
                if row['instance_id'] in identities:writer.writerow(row)
    if method in CONTROLLED:
        warmup=cfg['offline'].get('bc_warmup_epochs',0)
        cfg['run_name']+='_SMOKE_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        cfg['data']['train_dataset_path']=str(run/'train')
        cfg['model'].update(embedding_dim=32)
        cfg['training'].update(epochs=warmup+2,num_envs_per_gpu=4,n_traj=2,ppo_step_chunk_size=8,
            checkpoint_interval=1,latest_checkpoint_interval=1,debug_log_every=1,monitor_interval=1,
            post_update_kl_interval=0,monitor_output_dir=str(run/'monitoring'))
        cfg['env']['use_jit_mask']=False
        cfg['evaluation'].update(eval_path=str(run/'val'),eval_interval=1,eval_batch_size=2,eval_n_traj=2,eval_output_dir=str(run/'validation'))
        cfg['offline'].update(expert_dataset_path=str(run/'train'),expert_solution_path=str(expert)) if method in EXPERT_METHODS else None
        if method in ('awbc','dapg'):cfg['offline'].update(bc_batch_size=8,bc_updates_per_epoch=2)
        cfg['offline']['sl_expert_logprob_chunk_size']=16
        cfg['advantage']['sl_expert_logprob_chunk_size']=16
        cfg['experiment_protocol'].update(smoke_only=True,world_size=1,global_batch=4,training_seed=manifest['training_seed'])
    else:
        cfg.update(train_path=str(run/'train'),val_path=str(run/'val'),instance_exposures=2,
            batch_size=1,eval_interval_exposures=1,eval_batch_size=1,run_id=cfg['run_id']+'_SMOKE_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
        cfg['checkpoint_backup_dir']=str(Path(cfg['checkpoint_backup_dir']).with_name(Path(cfg['checkpoint_backup_dir']).name+'_smoke'))
        cfg['experiment_protocol']['smoke_only']=True
    _dump(run/'config.yaml',cfg)
    return run,cfg


def smoke(args):
    manifest=verify_campaign(args.campaign)
    methods=METHODS if args.method=='all' else [args.method]
    results={}
    for method in methods:
        spec=manifest['methods'][method]
        if not spec['ready']:
            results[method]=dict(state='blocked',reason='missing audited training expert trajectories');continue
        old=Path(spec['run_dir'])/'smoke'
        if old.exists():
            if not args.retry: raise FileExistsError(f'{old} exists; inspect its result and use --retry for a failed attempt')
            result_path=old/'smoke_result.json'
            if result_path.exists() and _read(result_path).get('state')=='passed':
                results[method]=_read(result_path);continue
            old.rename(old.with_name('smoke_attempt_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())))
        run,cfg=_small_assets(manifest,method)
        try:
            if method not in CONTROLLED:
                from e1.native import train
                first=train(cfg,run,resume=False,device='cpu',max_updates=1)
                second=train(cfg,run,resume=True,device='cpu')
                result=dict(state='passed',initial=first,resumed=second)
            else:
                commands=[]
                for resume in (None,_checkpoint_dir(manifest,cfg)/f'checkpoint_epoch_{cfg["training"]["epochs"]-1:04d}.pt'):
                    command=_worker_command(manifest,method,run/'config.yaml',resume=resume)
                    if resume is None: command += ['--stop-after-epoch', str(cfg['training']['epochs']-1)]
                    commands.append(command)
                    with (run/('initial.log' if resume is None else 'resume.log')).open('w') as log:
                        subprocess.run(command,cwd=manifest['code_root'],env=_process_env(manifest),stdout=log,stderr=subprocess.STDOUT,check=True)
                import torch
                last=_checkpoint_dir(manifest,cfg)/'checkpoint_final.pt'
                checkpoint=torch.load(last,map_location='cpu',weights_only=False)
                result=dict(state='passed',commands=commands,last_checkpoint=str(last),last_sha256=digest(last),
                    epoch=checkpoint['epoch'],config_sha256=digest(run/'config.yaml'),
                    evidence='Real CVRP100 subset, real optimizer/expert path, validation, save, new process full resume')
            atomic_json(run/'smoke_result.json',result);results[method]=result
        except Exception as error:
            result=dict(state='failed',reason=f'{type(error).__name__}: {error}',logs=str(run))
            atomic_json(run/'smoke_result.json',result);results[method]=result
        print(json.dumps(dict(method=method,**results[method])),flush=True)
    atomic_json(Path(args.campaign)/f'smoke_{args.method}.json',results)
    return results


def _smoke_passed(spec):
    path=Path(spec['run_dir'])/'smoke/smoke_result.json'
    return path.exists() and _read(path).get('state')=='passed'


def launch(args):
    manifest=verify_campaign(args.campaign)
    methods=[m for m,_ in SERVER_PLANS[args.server]] if args.method=='all' else [args.method]
    if not methods:print('This server is reserved; no E1 jobs assigned.');return
    plan=[]
    for method in methods:
        spec=manifest['methods'][method]
        from e1.lifecycle import assert_can_start
        assert_can_start(manifest,method)
        gpus=list(map(int,args.gpus.split(','))) if args.gpus else spec['gpus']
        if len(set(gpus))!=len(gpus) or any(g<0 for g in gpus):raise ValueError('GPU indices must be unique nonnegative integers')
        cfg=_yaml(spec['config'])
        if len(gpus)!=cfg['experiment_protocol']['world_size']:raise ValueError('GPU count must match frozen world size')
        plan.append(dict(method=method,gpus=gpus,smoke_passed=_smoke_passed(spec),assets_ready=spec['ready']))
    print(json.dumps(dict(execute=args.execute,server=args.server,plan=plan),indent=2))
    if not args.execute:return
    if any(not row['smoke_passed'] or not row['assets_ready'] for row in plan):raise ValueError('Every requested method must first pass its own smoke and asset checks')
    gpu_list=[g for row in plan for g in row['gpus']]
    if len(gpu_list)!=len(set(gpu_list)):raise ValueError('Concurrent methods may not share assigned GPUs')
    scheduler=Path(args.campaign)/('scheduler_'+args.server)
    scheduler.mkdir(exist_ok=True)
    plan_path=scheduler/'plan.json'
    if (scheduler/'pid.json').exists():
        old=_read(scheduler/'pid.json');proc=Path('/proc')/str(old.get('pid',0))/'cmdline'
        if proc.exists() and str(scheduler).encode() in proc.read_bytes():raise ValueError('This scheduler is already live')
    atomic_json(plan_path,plan)
    command=[sys.executable,'-m','e1.runner','_supervise','--campaign',str(Path(args.campaign).resolve()),'--scheduler',str(scheduler.resolve())]
    with (scheduler/'supervisor.log').open('a') as log:
        process=subprocess.Popen(command,cwd=manifest['code_root'],env=dict(os.environ,PYTHONPATH=manifest['code_root']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(scheduler/'pid.json',dict(pid=process.pid,started_at=_utc(),command=command))
    print(f'Queued supervisor PID {process.pid}; status: {scheduler / "status.json"}')


def _hardware_preflight(manifest,method):
    spec=manifest['methods'][method];cfg=_yaml(spec['config']);run=Path(spec['run_dir'])/'hardware_preflight'
    run.mkdir(exist_ok=False)
    if method in CONTROLLED:
        warmup=cfg['offline'].get('bc_warmup_epochs',0)
        cfg['run_name']+='_HARDWARE_PREFLIGHT';cfg['training'].update(epochs=warmup+2,checkpoint_interval=1,latest_checkpoint_interval=1,
            monitor_output_dir=str(run/'monitoring'))
        cfg['evaluation'].update(eval_interval=1,eval_limit=cfg['evaluation']['eval_batch_size'],eval_output_dir=str(run/'validation'))
    else:
        cfg.update(instance_exposures=cfg['batch_size']*2,eval_interval_exposures=cfg['batch_size'],
                   eval_limit=cfg['eval_batch_size'],run_id=cfg['run_id']+'_HARDWARE_PREFLIGHT')
        cfg['checkpoint_backup_dir']=str(Path(cfg['checkpoint_backup_dir']).with_name(Path(cfg['checkpoint_backup_dir']).name+'_hardware_preflight'))
    cfg['experiment_protocol']['hardware_preflight_only']=True
    _dump(run/'config.yaml',cfg)
    return run/'config.yaml'


def supervise(args):
    manifest=verify_campaign(args.campaign);folder=Path(args.scheduler)
    lock=(folder/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=_runtime();plan=_read(folder/'plan.json');active={};finished={};pending=list(plan)
    status=dict(created_at=_utc(),pid=os.getpid(),state='waiting_for_idle_gpus',methods={})
    try:
        while pending or active:
            for method,item in list(active.items()):
                rc=item['process'].poll()
                if rc is None:continue
                item['log'].close()
                if rc==0 and item['stage']=='hardware_preflight':
                    verify_campaign(args.campaign)
                    spec=manifest['methods'][method]
                    from e1.lifecycle import assert_can_start
                    assert_can_start(manifest,method)
                    # Formal training gets freshly initialized weights and RNG.
                    cmd=_worker_command(manifest,method,spec['config'],gpus=item['gpus'],device='cuda')
                    log=(Path(spec['run_dir'])/'train.log').open('a')
                    process=subprocess.Popen(cmd,cwd=manifest['code_root'],env=_process_env(manifest,item['gpus']),stdout=log,stderr=subprocess.STDOUT)
                    item.update(process=process,log=log,stage='training',command=cmd)
                    status['methods'][method]=dict(state='training',pid=process.pid,gpus=item['gpus'],started_at=_utc())
                else:
                    for gpu_lock in item['locks']:gpu_lock.close()
                    finished[method]=dict(state='completed' if rc==0 else 'failed',stage=item['stage'],returncode=rc,updated_at=_utc())
                    status['methods'][method]=finished[method];del active[method]
            if pending:
                cards=runtime.gpu_snapshot()
                for item in list(pending):
                    method=item['method'];gpus=item['gpus'];runtime.validate_requested_gpus(gpus,cards)
                    if any(not runtime.idle_gpu(cards[g]) for g in gpus):
                        status['methods'][method]=dict(state='waiting_for_idle_gpus',gpus=gpus);continue
                    locks=[]
                    for gpu in gpus:
                        held=runtime.acquire_gpu_lock(cards[gpu]['uuid'])
                        if held is None:break
                        locks.append(held)
                    current=runtime.gpu_snapshot()
                    if len(locks)!=len(gpus) or any(not runtime.idle_gpu(current[g]) for g in gpus):
                        for held in locks:held.close()
                        continue
                    verify_campaign(args.campaign)
                    from e1.lifecycle import assert_can_start
                    assert_can_start(manifest,method)
                    config=_hardware_preflight(manifest,method)
                    cmd=_worker_command(manifest,method,config,gpus=gpus,device='cuda')
                    log=(config.parent/'preflight.log').open('w')
                    process=subprocess.Popen(cmd,cwd=manifest['code_root'],env=_process_env(manifest,gpus),stdout=log,stderr=subprocess.STDOUT)
                    active[method]=dict(process=process,log=log,locks=locks,stage='hardware_preflight',gpus=gpus,command=cmd)
                    status['methods'][method]=dict(state='hardware_preflight',pid=process.pid,gpus=gpus,started_at=_utc())
                    pending.remove(item)
            status.update(updated_at=_utc(),state='running' if active else 'waiting_for_idle_gpus' if pending else 'finished')
            try:
                hardware=runtime.gpu_snapshot()
                with (folder/'hardware.jsonl').open('a') as h:h.write(json.dumps(dict(timestamp=_utc(),gpus=hardware))+'\n')
            except (OSError,subprocess.SubprocessError):pass
            atomic_json(folder/'status.json',status)
            if pending or active:time.sleep(30)
    finally:
        # Do not kill training processes on scheduler interruption; leave evidence.
        for item in active.values():
            item['log'].close()
            for held in item['locks']:held.close()
        lock.close()


def resume(args):
    manifest=verify_campaign(args.campaign);spec=manifest['methods'][args.method];cfg=_yaml(spec['config'])
    gpus=list(map(int,(args.gpus or ','.join(map(str,spec['gpus']))).split(',')))
    if len(gpus)!=cfg['experiment_protocol']['world_size'] or len(set(gpus))!=len(gpus) or any(g<0 for g in gpus):
        raise ValueError('Resume must preserve rank count and use unique nonnegative GPUs')
    checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    from e1.lifecycle import assert_checkpoint_owner
    assert_checkpoint_owner(manifest,args.method,checkpoint)
    cmd=_worker_command(manifest,args.method,spec['config'],gpus=gpus,resume=checkpoint,device='cuda')
    print(shlex.join(cmd))
    if not args.execute:return
    runtime=_runtime();cards=runtime.gpu_snapshot();runtime.validate_requested_gpus(gpus,cards)
    if any(not runtime.idle_gpu(cards[g]) for g in gpus):raise ValueError('Requested resume GPUs are busy; existing processes were left untouched')
    locks=[]
    try:
        for gpu in gpus:
            held=runtime.acquire_gpu_lock(cards[gpu]['uuid'])
            if held is None:raise ValueError('GPU is reserved by another scheduler')
            locks.append(held)
        if any(not runtime.idle_gpu(runtime.gpu_snapshot()[g]) for g in gpus):raise ValueError('GPU became busy')
        subprocess.run(cmd,cwd=manifest['code_root'],env=_process_env(manifest,gpus),check=True)
    finally:
        for held in locks:held.close()


def evaluate(args):
    manifest=verify_campaign(args.campaign);method=args.method;spec=manifest['methods'][method]
    cfg=_yaml(spec['config']);report=_read(manifest['audit_path'])
    test=report['splits']['test']['path']
    if not report.get('test_data_ready'):raise ValueError('Independent test data audit has not passed')
    checkpoint=Path(args.checkpoint).resolve()
    from e1.lifecycle import assert_test_ready
    acceptance=assert_test_ready(manifest,method,checkpoint)
    selected = (_checkpoint_dir(manifest,cfg)/'checkpoint_best.pt' if method in CONTROLLED
                else Path(spec['run_dir'])/'native_checkpoints/best.pt')
    if not selected.exists() or digest(checkpoint)!=digest(selected):
        raise ValueError('Final E1 test must use the validation-selected best checkpoint (or its identical backup)')
    output=Path(spec['run_dir'])/'test'
    if output.exists():raise FileExistsError('Final test output already exists; inspect it before any explicit retry')
    gpu_lock=None
    try:
        if args.device.startswith('cuda'):
            runtime=_runtime();cards=runtime.gpu_snapshot()
            index=int(args.device.split(':')[1]) if ':' in args.device else 0
            runtime.validate_requested_gpus([index],cards)
            if not runtime.idle_gpu(cards[index]):raise ValueError('Requested evaluation GPU is busy; use an idle GPU or CPU')
            gpu_lock=runtime.acquire_gpu_lock(cards[index]['uuid'])
            if gpu_lock is None or not runtime.idle_gpu(runtime.gpu_snapshot()[index]):raise ValueError('Evaluation GPU became reserved/busy')
        output.mkdir(exist_ok=False)
        if method in CONTROLLED:
            from e1.evaluation import evaluate_checkpoint
            result=evaluate_checkpoint(cfg,checkpoint,output/'instances.jsonl',test_path=test,device=args.device,batch_size=args.batch_size)
        else:
            from e1.native import evaluate as native_evaluate
            cfg['test_path']=test
            result=native_evaluate(cfg,spec['run_dir'],checkpoint,split='test',device=args.device,output_path=output/'instances.jsonl',eval_batch_size=args.batch_size)
    finally:
        if gpu_lock is not None:gpu_lock.close()
    rows=[json.loads(line) for line in (output/'instances.jsonl').read_text().splitlines() if line.strip()]
    with (Path(args.campaign)/'assets/instance_index.csv').open() as handle:
        expected={r['instance_id'] for r in csv.DictReader(handle) if r['split']=='test'}
    actual=[r['instance_id'] for r in rows]
    if len(actual)!=1000 or len(set(actual))!=1000 or set(actual)!=expected:
        raise ValueError('Incomplete or mismatched E1 test IDs; rows retained but not accepted as a final result')
    if any(r['actual_K']!=50 or r['requested_K']!=50 for r in rows):
        raise ValueError('Every final test instance must record exactly 50 total candidate attempts')
    atomic_json(output/'manifest.json',dict(accepted=True,instances_sha256=digest(output/'instances.jsonl'),
        **acceptance,checkpoint=str(checkpoint),test_data_sha256=report['splits']['test']['files']['instances.pkl']['sha256'],result=result))
    print(json.dumps(result,indent=2))


def summarize(args):
    from e1.summary import summarize_campaign
    result=summarize_campaign(args.campaign,gurobi_path=args.gurobi,additional_campaigns=args.additional_campaign)
    print(json.dumps(result,indent=2))


def parser():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('native-status');p.add_argument('--method',choices=('rrnco','radar','all'),default='all')
    p=sub.add_parser('preflight');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare');p.add_argument('--audit',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3009);p.add_argument('--epochs',type=int,default=1500)
    p.add_argument('--backup-root',type=Path);p.add_argument('--ppo-chunk',type=int,default=8);p.add_argument('--expert-chunk',type=int,default=32);p.add_argument('--eval-batch',type=int,default=8)
    p=sub.add_parser('smoke');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--method',choices=(*METHODS,'all'),default='all');p.add_argument('--retry',action='store_true')
    p=sub.add_parser('launch');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--server',choices=SERVER_PLANS,required=True);p.add_argument('--method',choices=(*METHODS,'all'),default='all');p.add_argument('--gpus');p.add_argument('--execute',action='store_true')
    p=sub.add_parser('_supervise');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--scheduler',type=Path,required=True)
    p=sub.add_parser('resume');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--gpus');p.add_argument('--execute',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=1)
    p=sub.add_parser('summarize');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--gurobi',type=Path);p.add_argument('--additional-campaign',action='append',type=Path,default=[])
    return parser


def main():
    args=parser().parse_args()
    if getattr(args,'campaign',None) and args.command not in ('prepare','summarize'):
        source=Path(_read(Path(args.campaign)/'manifest.json')['code_root']).resolve()
        if source!=ROOT.resolve():
            env=dict(os.environ,PYTHONPATH=str(source))
            subprocess.run([sys.executable,'-m','e1.runner',*sys.argv[1:]],cwd=source,env=env,check=True)
            return
    if args.command=='native-status':
        from e1.native import native_status
        statuses=[native_status(m) for m in (('rrnco','radar') if args.method=='all' else [args.method])]
        print(json.dumps(statuses,indent=2))
        if any(not s['source_available'] or not s['dependencies_available'] for s in statuses): raise SystemExit(1)
    elif args.command=='preflight':
        from e1.assets import audit_assets
        result=audit_assets(args.data_root,args.output,teacher_force=True,
            env_config=dict(normalize_reward=True,reward_contract='legacy',reward_mode='distance',max_steps_factor=4),progress=lambda r:print(json.dumps(r),flush=True))
        print(json.dumps(result,indent=2))
    else:
        result=globals()[args.command.lstrip('_')](args)
        if args.command=='smoke' and any(v.get('state')!='passed' for v in result.values()):raise SystemExit(1)

if __name__=='__main__':main()
