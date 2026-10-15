#!/usr/bin/env python3
"""Portable controlled VRPTW profiles. Shell detaches audit/smoke/launch/test."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from e1.vrptw_campaign import prepare
from e1.evaluation import atomic_json
from e1.runner import verify_campaign

PROFILES={
 'a6000':[(50,{'slppo':[0]}),(15,{'slppo':[1]})],
 '2080ti_b':[(50,{'awbc':[0],'dapg':[1]}),(15,{'awbc':[2],'dapg':[3]})],
 '2080ti_a':[(15,{'ppo_base':[0],'ppo_rdi_agda':[1]}),(50,{'ppo_base':[0],'ppo_rdi_agda':[1]})],
 '2080ti_3':[(100,{'slppo':[0,1]})],
 '2080ti_a_ev15':[(15,{'ppo_base':[0,1],'ppo_rdi_agda':[2,3]})],
 '2080ti_a_ev100':[(100,{'ppo_base':[0,1],'ppo_rdi_agda':[2,3]})],
}
LABELS={'ppo_base':'PPO (base)','ppo_rdi_agda':'PPO (RDI+AGDA)','awbc':'AWBC','dapg':'DAPG','slppo':'SL-PPO'}
METHODS=[*LABELS.values(),'RRNCO','RADAR','Gurobi (2 h, 1 thread)']
FIELDS=['method','status','feasible_count','test_count','all_feasible_mean_km','common_mean_km','common_gap_to_gurobi_pct','runtime_s_per_instance','training_hours']


def csv_read(path):
    with path.open(newline='') as h:return list(csv.DictReader(h))


def csv_write(path,fields,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def ensure_reports(reports, problem="VRPTW"):
    reports.mkdir(parents=True,exist_ok=True)
    overview=reports/('EVRPTW_test_overview.csv' if problem=='EVRPTW' else 'CVRP_VRPTW_test_overview_20261013.csv')
    with (reports/'.test_report.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=csv_read(overview) if overview.exists() else []
        for scale in (15,50,100):
            path=reports/f'E1_{problem}{scale}_partial_20261013';path.mkdir(exist_ok=True)
            for label in METHODS:
                if not any(r['problem']==problem and r['customers']==str(scale) and r['method']==label for r in rows):
                    r=dict.fromkeys(FIELDS,'TBD');r.update(method=label,status='Independent test result not yet received',test_count='1000')
                    rows.append(dict(problem=problem,customers=str(scale),**r))
            main=path/'main_results_partial.csv'
            if not main.exists():csv_write(main,FIELDS,[{k:r[k] for k in FIELDS} for r in rows if r['problem']==problem and r['customers']==str(scale)])
        csv_write(overview,['problem','customers',*FIELDS],rows)
    return overview


def run_command(manifest,*args):
    env=dict(os.environ,PYTHONPATH=manifest['code_root'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-m','e1.runner',*map(str,args)],cwd=manifest['code_root'],env=env,check=True)



def predecessor_state(campaign):
    """Both Cus15 methods must finish training AND accepted test/reporting."""
    scheduler=campaign/'scheduler_2080ti_a/status.json'
    watcher=campaign/'test_watcher/status.json'
    if not scheduler.exists() or not watcher.exists():return 'waiting_for_cus15'
    training=json.loads(scheduler.read_text()).get('methods',{})
    tests=json.loads(watcher.read_text()).get('jobs',{})
    for method in ('ppo_base','ppo_rdi_agda'):
        a=training.get(method,{}).get('state')
        b=tests.get('vrptw15_'+method,{}).get('state')
        if a in ('failed','blocked','interrupted','stopped_by_user') or b=='blocked':
            return 'blocked_by_cus15'
    if all(training.get(m,{}).get('state')=='completed' and
           tests.get('vrptw15_'+m,{}).get('state')=='completed' for m in ('ppo_base','ppo_rdi_agda')):
        return 'ready'
    return 'waiting_for_cus15'


def wait_for_cus15(args, campaign):
    previous=args.output/'E1_VRPTW15_S3009_E1500_2080TI_A_V1'
    while True:
        state=predecessor_state(previous)
        atomic_json(campaign/'handoff_status.json',dict(state=state,pid=os.getpid(),
            predecessor=str(previous),target=str(campaign),gpus=[0,1],waits_for_cus100=False,
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        if state=='blocked_by_cus15':raise RuntimeError('Cus15 training/test failed; automatic handoff blocked')
        if state=='ready':
            from e1.lifecycle import assert_accepted_test_result
            manifest=verify_campaign(previous)
            for method in ('ppo_base','ppo_rdi_agda'):
                assert_accepted_test_result(manifest,method)
            return
        time.sleep(30)


def run_scale(args, scale, placements):
    problem=getattr(args,'problem','vrptw').upper()
    version='V2' if args.profile=='2080ti_a_ev15' else 'V1'
    server='2080ti_a' if args.profile in ('2080ti_a_ev15','2080ti_a_ev100') else args.profile
    campaign=args.output/f'E1_{problem}{scale}_S3009_E1500_{args.profile.upper()}_{version}'
    defaults={'ppo_base':32,'ppo_rdi_agda':32 if args.profile=='2080ti_a_ev15' else 8 if args.profile=='2080ti_a_ev100' else 14,'awbc':8,'dapg':8,'slppo':32 if args.profile=='a6000' else 8}
    chunks={m:int(os.environ.get('E1_'+m.upper()+'_PPO_CHUNK',defaults[m])) for m in placements}
    expert=int(os.environ.get('E1_EXPERT_CHUNK',128 if args.profile=='a6000' else 32))
    if min([*chunks.values(),expert])<1:raise ValueError('Chunks must be positive')
    print(json.dumps(dict(campaign=str(campaign),scale=scale,placements=placements,chunks=chunks,expert_chunk=expert,
          protocol='scratch seed3009; 1500 online loops; global batch64; n_traj50; PPO passes4; minibatches4; K50; validation every50')),flush=True)
    if not args.execute:return
    manifest=prepare(campaign,args.data_root,scale,server,placements,chunks,expert,problem_type=problem.lower())
    scheduler=campaign/f'scheduler_{server}'
    if not (scheduler/'pid.json').exists():
        for method in placements:
            spec=manifest['methods'][method];run=Path(spec['run_dir'])
            if (run/'train.log').exists() or (run/'hardware_preflight').exists():
                raise RuntimeError('Existing formal artifacts; inspect and explicitly resume rather than restart')
            retry=['--retry'] if (run/'smoke').exists() else []
            run_command(manifest,'smoke','--campaign',campaign,'--method',method,*retry)
        if args.profile=='2080ti_a' and scale==50:
            wait_for_cus15(args,campaign)
        run_command(manifest,'launch','--campaign',campaign,'--server',server,'--execute')
        if args.profile=='2080ti_a' and scale==50:
            atomic_json(campaign/'handoff_status.json',dict(state='handed_off',predecessor=str(args.output/'E1_VRPTW15_S3009_E1500_2080TI_A_V1'),target=str(campaign),gpus=[0,1],waits_for_cus100=False,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    else:
        print(f'Existing scheduler preserved: {scheduler}',flush=True)
    reports=ROOT/'results/e1/reports';overview=ensure_reports(reports,problem)
    watcher=campaign/'test_watcher';watcher.mkdir(exist_ok=True)
    plan=dict(batch_size=8,poll_seconds=60,overview=str(overview),renderer=str(ROOT/'scripts/render_e1_test_table.py'),
              jobs=[dict(id=f'{problem.lower()}{scale}_{m}',campaign=str(campaign),method=m,label=LABELS[m],problem=problem,customers=scale,
                         gpus=gpus,scheduler=str(scheduler),report_dir=str(reports/f'E1_{problem}{scale}_partial_20261013')) for m,gpus in placements.items()])
    status=watcher/'status.json'
    if status.exists():
        s=json.loads(status.read_text());pid=s.get('pid',0);proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists() and b'watch_e1_tests.py' in proc.read_bytes():return
        if s.get('state') in ('completed','finished_with_errors'):
            print(f'Existing terminal test status preserved: {status}',flush=True);return
    atomic_json(watcher/'plan.json',plan)
    subprocess.run([sys.executable,str(ROOT/'scripts/watch_e1_tests.py'),'--plan',str(watcher/'plan.json'),'--execute'],check=True)
    print(f'Training/test queued: {scheduler}/status.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',choices=PROFILES)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--problem',choices=('vrptw','evrptw'),default='vrptw')
    parser.add_argument('--scale',type=int,choices=(15,50,100))
    parser.add_argument('--data-root',type=Path,default=ROOT.parent/'AAAI_Dataset')
    parser.add_argument('--output',type=Path,default=ROOT/'results/e1')
    args=parser.parse_args();args.data_root=args.data_root.resolve();args.output=args.output.resolve()
    if args.problem=='evrptw' and args.profile not in ('a6000','2080ti_a_ev15','2080ti_a_ev100'):parser.error('Unsupported EVRPTW profile')
    if args.profile in ('2080ti_a_ev15','2080ti_a_ev100') and args.problem!='evrptw':parser.error('This profile requires --problem evrptw')
    choices=[(n,g) for n,g in PROFILES[args.profile] if args.scale is None or n==args.scale]
    if not choices:parser.error('Scale is not assigned to this profile')
    if not args.execute:
        for n,g in choices:run_scale(args,n,g)
        return
    launcher=args.output/'launchers'/f'{args.problem.upper()}_{args.profile}_V1';launcher.mkdir(parents=True,exist_ok=True)
    # Independent scale workers allow Cus15 to proceed while Cus50 audits experts.
    children=[]
    for n,g in choices:
        if args.scale is not None:
            with (launcher/f'cus{n}.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_scale(args,n,g)
            return
        log=(launcher/f'cus{n}.log').open('a')
        cmd=[sys.executable,str(Path(__file__).resolve()),args.profile,'--execute','--scale',str(n),
             '--data-root',str(args.data_root),'--output',str(args.output),'--problem',args.problem]
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children.append((child,log))
    codes=[]
    for child,log in children:codes.append(child.wait());log.close()
    if any(codes):raise SystemExit('At least one scale failed; inspect per-scale logs. Other scales may be running.')


if __name__=='__main__':main()
