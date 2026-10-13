"""Prepare two controlled VRPTW100 arms and hand them off after CVRP completes."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from e1.configs import build_config, content_hash, digest
from e1.evaluation import atomic_json
from e1.runner import _runtime, verify_campaign

METHODS = ('ppo_base', 'ppo_rdi_agda')
SCHEMA = 'aaai_e1_vrptw100_v1'


def configuration(method, data, run, unit, backup, epochs=1500):
    cfg=build_config(method,data_root=data,output_dir=run,distance_unit_km=unit,
        backup_root=backup,epochs=epochs,world_size=2,ppo_chunk=32 if method=='ppo_base' else 8)
    cfg['dataset_name']='Geo-VRPTW-v1'
    cfg['run_name']=f'{run.parent.parent.name}_{method}'
    cfg['data'].update(problem_type='vrptw',train_dataset_path=str(data/'dataset/vrptw/train/Cus100'))
    cfg['evaluation']['eval_path']=str(data/'dataset/vrptw/val/Cus100')
    if method=='ppo_base':cfg['model']['e1_no_road_task']='vrptw'
    cfg['experiment_protocol'].update(schema=SCHEMA,task='vrptw100',actor_sha256=content_hash(cfg['model']),
        base_preserves=['coordinates','demand','capacity','time_windows','service_time','current_clock','history','physical_feasibility_mask'],
        road_input_contract='base excludes explicit road matrices/context; real clock, masks and rewards remain physical')
    return cfg


def prepare(folder,data,predecessors):
    if folder.exists():raise FileExistsError('Frozen campaign exists; use --execute to watch the prepared campaign')
    from e1.native import export_native_dataset
    folder.mkdir(parents=True);assets=folder/'assets';assets.mkdir()
    audit={}
    for split in ('train','val'):
        path=assets/f'{split}.npz'
        audit[split]=export_native_dataset(data/f'dataset/vrptw/{split}/Cus100',path,
            split=split,expected_customers=100,problem_type='vrptw')
        if split=='train':
            with np.load(path) as arrays:
                d=arrays['distance_matrix'];unit=float(np.median(d[:,0,1:]+d[:,1:,0]))
        path.unlink() # keep source hashes and identities, not a duplicate dataset
    if audit['val']['count']!=1000:raise ValueError('Validation must contain 1000 instances')
    for key in ('instance_id','semantic_sha256'):
        if {r[key] for r in audit['train']['instances']} & {r[key] for r in audit['val']['instances']}:
            raise ValueError('Train and validation overlap')
    if not np.isfinite(unit) or unit<=0:raise ValueError('Invalid train-only distance unit')
    audit.update(problem='vrptw',distance_unit_km=unit,distance_unit_fit_split='train',test_opened=False)
    atomic_json(assets/'data_audit.json',audit)
    code=folder/'source/CaliRoute';source=_runtime().source_snapshot(code,include_initialization_assets=False)
    (code/'results').unlink();(folder/'artifacts').mkdir();(code/'results').symlink_to(folder/'artifacts',target_is_directory=True)
    methods={};backup=folder.parent/'checkpoint_backups'/folder.name
    for method,gpus in zip(METHODS,([0,1],[2,3])):
        run=folder/'runs'/method;run.mkdir(parents=True)
        cfg=configuration(method,data,run,unit,backup)
        cfg['experiment_protocol'].update(source_content_sha256=source['content_sha256'],
            data_audit_sha256=digest(assets/'data_audit.json'),expert_pool_sha256=None)
        path=run/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        methods[method]=dict(config=str(path),config_sha256=digest(path),run_dir=str(run),gpus=gpus,
            ready=True,server='2080ti_a',requires_expert=False)
    manifest=dict(schema=SCHEMA,campaign=str(folder),code_root=str(code),artifacts_root=str(folder/'artifacts'),source=source,
        training_seed=3009,initialization='scratch',methods=methods,backup_root=str(backup),
        audit_path=str(assets/'data_audit.json'),distance_unit=dict(km=unit,fit_split='train'),
        inputs=[dict(path=audit[s]['source_path'],sha256=audit[s]['source_sha256']) for s in ('train','val')]+
               [dict(path=str(assets/'data_audit.json'),sha256=digest(assets/'data_audit.json'))],
        server_plan={'2080ti_a':[[m,methods[m]['gpus']] for m in METHODS]})
    atomic_json(folder/'manifest.json',manifest)
    atomic_json(folder/'watch_dependencies.json',{'campaigns':[str(p) for p in predecessors]})
    for method in METHODS:
        subprocess.run([sys.executable,'-m','e1.runner','smoke','--campaign',str(folder),'--method',method],
            cwd=code,env=dict(os.environ,PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),check=True)
    return manifest


def predecessor_state(folder):
    status=folder/'scheduler_2080ti_a/status.json'
    if not status.exists():return 'waiting'
    data=json.loads(status.read_text());state=data.get('methods',{}).get('ppo_base',{}).get('state')
    if state in ('failed','stopped_by_user','interrupted','blocked'):return 'blocked'
    if state!='completed':return 'waiting'
    result=folder/'runs/ppo_base/training_result.json'
    if not result.exists():return 'waiting'
    result=json.loads(result.read_text())
    if result.get('state')!='completed' or not Path(result.get('last_checkpoint','')).is_file():return 'blocked'
    return 'completed'


def watch(folder):
    manifest=verify_campaign(folder)
    lock=(folder/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dependencies=json.loads((folder/'watch_dependencies.json').read_text())['campaigns']
    if not dependencies:raise ValueError('Explicit predecessor campaigns required')
    status=dict(pid=os.getpid(),campaign=str(folder),state='waiting_for_predecessors')
    while True:
        states={p:predecessor_state(Path(p)) for p in dependencies}
        status.update(predecessors=states,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        if 'blocked' in states.values():status['state']='blocked_by_predecessor'
        atomic_json(folder/'watcher_status.json',status)
        if status['state']=='blocked_by_predecessor':raise RuntimeError('Predecessor stopped/failed; no automatic handoff')
        if all(s=='completed' for s in states.values()):break
        time.sleep(30)
    # Check a real final checkpoint, not just a stale status flag.
    import torch
    for previous in dependencies:
        result=json.loads((Path(previous)/'runs/ppo_base/training_result.json').read_text())
        checkpoint=torch.load(result['last_checkpoint'],map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=checkpoint['config']['training']['epochs']:raise ValueError('Predecessor did not reach its training budget')
        del checkpoint
    verify_campaign(folder)
    subprocess.run([sys.executable,'-m','e1.runner','launch','--campaign',str(folder),'--server','2080ti_a','--execute'],
        cwd=manifest['code_root'],env=dict(os.environ,PYTHONPATH=manifest['code_root']),check=True)
    status.update(state='handed_off',scheduler_status=str(folder/'scheduler_2080ti_a/status.json'))
    atomic_json(folder/'watcher_status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--after',type=Path,action='append',default=[])
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--watch',action='store_true')
    args=p.parse_args();folder=args.campaign.resolve()
    if args.prepare:
        if not args.after or not args.data_root:p.error('Preparation requires --data-root and --after campaigns')
        prepare(folder,args.data_root.resolve(),[x.resolve() for x in args.after])
    if args.watch:return watch(folder)
    if args.execute:
        manifest=verify_campaign(folder)
        if (folder/'watcher_pid.json').exists():
            old=json.loads((folder/'watcher_pid.json').read_text())
            proc=Path(f'/proc/{old["pid"]}/cmdline')
            if proc.exists() and str(folder).encode() in proc.read_bytes():raise RuntimeError('Watcher already running')
        cmd=[sys.executable,'-m','e1.vrptw_controlled','--campaign',str(folder),'--watch']
        with (folder/'watcher.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=manifest['code_root'],env=dict(os.environ,PYTHONPATH=manifest['code_root']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(folder/'watcher_pid.json',dict(pid=child.pid,command=cmd))
        print(f'Watcher PID={child.pid}; {folder}/watcher_status.json',flush=True)

if __name__=='__main__':main()
