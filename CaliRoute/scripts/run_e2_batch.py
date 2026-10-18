#!/usr/bin/env python3
"""Preview or prepare E2 server queues. Training is blocked until integration exists."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from e2.server_plans import PROFILES, server_plan, blockers


def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('server',choices=PROFILES)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--execute',action='store_true')
    parser.add_argument('--baseline-policy',choices=('pending','reuse','rerun'),default='pending')
    parser.add_argument('--data-root',type=Path,default=ROOT.parent/'AAAI_Dataset')
    parser.add_argument('--output',type=Path,default=ROOT/'results/e2')
    args=parser.parse_args()
    plan=server_plan(args.server,args.baseline_policy)
    campaign=args.output.resolve()/plan['campaign']
    print(f"Server: {args.server}; seed=3009; Cus100; test K=50; single GPU per job",flush=True)
    for gpu,queue in plan['queues'].items():
        print(f"GPU{gpu}: "+(' -> '.join(t['id'] for t in queue) or 'reserved'),flush=True)
    reasons=blockers(plan)
    print('NOT READY FOR TRAINING: '+str(len(reasons))+' unresolved integration/protocol items',flush=True)
    if args.check or args.execute:
        for reason in reasons:print('  - '+reason,flush=True)
        if reasons:
            print('No training process started. Use --prepare to save the queue and readiness status.',file=sys.stderr)
            return 2
        # No worker is implemented yet; do not offer an override to bypass this.
        raise RuntimeError('No approved E2 execution backend is installed')
    if not args.prepare:
        print('Preview only. --prepare writes the queue; --check reports readiness.');return 0
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=campaign/'server_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing campaign has a different plan; use a new E2_OUTPUT directory')
        write(plan_path,plan)
        tasks=[t for q in plan['queues'].values() for t in q]
        datasets={}
        for problem in sorted({t['problem'] for t in tasks}):
            for split in ('train','val','test'):
                dataset=args.data_root.resolve()/('test_release' if split=='test' else 'dataset')/problem/split/'Cus100/instances.pkl'
                datasets[f'{problem}/{split}']=dict(path=str(dataset),exists=dataset.is_file(),sha256_verified=False)
        for task in tasks:
            path=campaign/'runs'/task['id'];path.mkdir(parents=True,exist_ok=True)
            write(path/'planned_run.json',task)
        now=datetime.now(timezone.utc).isoformat()
        write(campaign/'status.json',dict(state='prepared_not_training',prepared_at=now,pid=os.getpid(),
            formal_training_started=False,launchable=False,blockers=reasons,datasets=datasets,
            missing_data=[key for key,item in datasets.items() if not item['exists']],
            plan=str(plan_path),runs={t['id']:'pending_integration' for t in tasks}))
        print(f'Prepared queue: {plan_path}\nReadiness: {campaign / "status.json"}',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
