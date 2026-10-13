#!/usr/bin/env python3
"""Frozen VRPTW100 native-baseline campaign; one GPU per method, no CVRP changes."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from e1.native import atomic_json, file_hash, native_status, train


def configuration(method,data,run,seed):
    return dict(method=method,problem_type='vrptw',run_id=f'{run.parent.parent.name}_{method}',training_seed=seed,
        num_customers=100,world_size=1,train_path=str(data/'dataset/vrptw/train/Cus100'),
        val_path=str(data/'dataset/vrptw/val/Cus100'),batch_size=64,n_traj=50,eval_k=50,
        instance_exposures=1500*64,eval_interval_exposures=50*64,eval_batch_size=8,
        inference_seed=17000000+seed,checkpoint_interval_updates=50,
        backup_dir=str(run.parent.parent.parent/'checkpoint_backups'/run.parent.parent.name/method),
        task_adapter='rrnco_native_rcvrptw_with_road_time_contract' if method=='rrnco' else 'radar_vrptw_time_inputs_and_mask_v1',
        experiment_protocol=dict(schema='aaai_e1_vrptw100_native_v1',method=method,training_seed=seed,
          initialization='scratch',global_batch=64,n_traj=50,nominal_online_loops=1500,total_instance_exposures=96000,
          validation_instances=1000,requested_K=50,selection_rule='feasibility_then_mean_physical_km',
          expert_data_used=False,test_used_for_training_or_selection=False,
          time_unit='all_time_quantities_divided_by_same_working_horizon',distance_objective='directed_road_km'))


def verify(campaign):
    manifest=json.loads((campaign/'manifest.json').read_text())
    for relative,digest in manifest['source']['files'].items():
        if file_hash(Path(manifest['code_root'])/relative)!=digest:raise ValueError('Frozen source changed')
    for item in manifest['inputs']:
        if file_hash(item['path'])!=item['sha256']:raise ValueError('Train/val data changed')
    for spec in manifest['methods'].values():
        if file_hash(spec['config'])!=spec['sha256']:raise ValueError('Frozen config changed')
    return manifest


def prepare(args):
    if args.campaign.exists():return verify(args.campaign)
    readiness={m:native_status(m) for m in ('rrnco','radar')}
    if not all(r['source_available'] and r['dependencies_available'] for r in readiness.values()):
        raise RuntimeError(f'Install native dependencies first: {readiness}')
    inputs=[]
    for split in ['train','val']:
        path=args.data_root/f'dataset/vrptw/{split}/Cus100/instances.pkl'
        inputs.append(dict(path=str(path),sha256=file_hash(path)))
    args.campaign.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_reward_norm_comparison import source_snapshot
    source_root=args.campaign/'source/CaliRoute'
    source=source_snapshot(source_root,include_initialization_assets=False)
    (source_root/'results').unlink()
    (args.campaign/'artifacts').mkdir()
    (source_root/'results').symlink_to(args.campaign/'artifacts',target_is_directory=True)
    methods={}
    for method,gpu in zip(('rrnco','radar'),args.gpus):
        run=args.campaign/'runs'/method;run.mkdir(parents=True)
        cfg=configuration(method,args.data_root,run,args.seed)
        # Point native provenance at the frozen vendored source, not a later pull.
        cfg['native_source_root']=str(source_root/'benchmarks/native_sources'/method)
        cfg['native_python']=readiness[method]['python_executable']
        path=run/'config.json';atomic_json(path,cfg)
        methods[method]=dict(gpu=gpu,run_dir=str(run),config=str(path),sha256=file_hash(path))
    manifest=dict(schema='aaai_e1_vrptw100_native_v1',code_root=str(source_root),source=source,inputs=inputs,methods=methods,
        seed=args.seed,created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_status='separate independent test after training; no test data opened')
    atomic_json(args.campaign/'manifest.json',manifest)
    return manifest


def acceptance(run,expected):
    state=json.loads((run/'native_status.json').read_text())
    rows=[json.loads(x) for x in (run/'native_validation.jsonl').read_text().splitlines()]
    last=rows[-1]
    if state['state']!='completed' or last['instances']!=expected or last['cost_mismatch_candidates'] or last['feasible_instances']!=expected:
        raise RuntimeError(f'Native VRPTW preflight failed validation: {last}')
    if state['counters']['instance_exposures']!=state['total_instance_exposures']:
        raise RuntimeError('Preflight sample budget mismatch')
    return dict(passed=True,validation=last,counters=state['counters'])


def smoke(cfg,run):
    dest=run/'cpu_smoke'
    if (dest/'accepted.json').exists():return
    if dest.exists():raise FileExistsError(f'Failed/partial smoke preserved at {dest}; use a new campaign')
    probe=copy.deepcopy(cfg)
    probe.update(batch_size=2,n_traj=2,eval_k=2,eval_batch_size=2,instance_exposures=4,
                 eval_interval_exposures=4,smoke_train_limit=8,smoke_val_limit=4,checkpoint_interval_updates=1,
                 backup_dir=str(dest/'backup'))
    probe['experiment_protocol']['smoke_only']=True
    train(probe,dest,device='cpu',max_updates=1)
    train(probe,dest,device='cpu',resume=True)
    atomic_json(dest/'accepted.json',acceptance(dest,4))


def arm(args,manifest):
    spec=manifest['methods'][args.arm];cfg=json.loads(Path(spec['config']).read_text());run=Path(spec['run_dir'])
    smoke(cfg,run)
    if args.smoke_only:return
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_reward_norm_comparison as runtime
    gpu=spec['gpu'];held=None
    while held is None:
        cards=runtime.gpu_snapshot();runtime.validate_requested_gpus([gpu],cards)
        if runtime.idle_gpu(cards[gpu]):held=runtime.acquire_gpu_lock(cards[gpu]['uuid'])
        if held is None:time.sleep(10)
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        preflight=run/'hardware_preflight'
        if not (preflight/'accepted.json').exists():
            if preflight.exists():raise FileExistsError('Partial GPU preflight preserved; use a new campaign')
            probe=copy.deepcopy(cfg);probe.update(instance_exposures=2*cfg['batch_size'],eval_interval_exposures=2*cfg['batch_size'],
                eval_limit=8,backup_dir=str(preflight/'backup'))
            probe['experiment_protocol']['hardware_preflight_only']=True
            train(probe,preflight,device='cuda')
            atomic_json(preflight/'accepted.json',acceptance(preflight,8))
        status_path=run/'native_status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status.get('state')=='completed':return
            if not args.resume:raise FileExistsError('Existing formal run needs explicit --resume')
            train(cfg,run,device='cuda',resume=True)
        else:
            train(cfg,run,device='cuda')
    finally:held.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parent/'AAAI_Dataset')
    p.add_argument('--seed',type=int,default=3009)
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--arm',choices=['rrnco','radar'])
    args=p.parse_args();args.campaign=args.campaign.resolve();args.data_root=args.data_root.resolve()
    if min(args.gpus)<0 or len(set(args.gpus))!=2:p.error('Choose two different GPU indices')
    if not (args.execute or args.smoke_only or args.arm):
        print(json.dumps(dict(task='VRPTW100',methods={'rrnco':args.gpus[0],'radar':args.gpus[1]},
            global_batch=64,n_traj=50,nominal_epochs=1500,seed=args.seed,campaign=str(args.campaign),execute=False),indent=2));return
    manifest=prepare(args)
    if args.arm:return arm(args,manifest)
    import fcntl
    lock=(args.campaign/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={};logs=[]
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    state={'pid':os.getpid(),'state':'running','methods':{}}
    try:
        for method,spec in manifest['methods'].items():
            cmd=[sys.executable,str(Path(manifest['code_root'])/'scripts/run_e1_vrptw_native.py'),
                 '--campaign',str(args.campaign),'--arm',method]
            if args.smoke_only:cmd+=['--smoke-only']
            if args.resume:cmd+=['--resume']
            log=(Path(spec['run_dir'])/'launcher.log').open('a');logs.append(log)
            child=subprocess.Popen(cmd,cwd=manifest['code_root'],env=dict(os.environ,PYTHONPATH=manifest['code_root'],
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[method]=child
            state['methods'][method]={'pid':child.pid,'gpu':spec['gpu'],'state':'running'}
        while True:
            for method,child in children.items():
                result=child.poll()
                state['methods'][method]['state']='running' if result is None else ('completed' if result==0 else 'failed')
                state['methods'][method]['exit_code']=result
            state['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            done=all(c.poll() is not None for c in children.values())
            if done:state['state']='completed' if all(c.returncode==0 for c in children.values()) else 'failed'
            atomic_json(args.campaign/'status.json',state)
            if done:break
            time.sleep(5)
        if state['state']=='failed':raise RuntimeError('An arm failed; inspect runs/METHOD/launcher.log')
    except KeyboardInterrupt:
        state['state']='stopped'
        atomic_json(args.campaign/'status.json',state)
        raise
    finally:
        for child in children.values():
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child in children.values():
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        for log in logs:log.close()
        lock.close()

if __name__=='__main__':main()
