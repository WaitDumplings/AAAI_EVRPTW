#!/usr/bin/env python3
import argparse,fcntl,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e1.native import atomic_json
from e2.rrnco_ppo_launch import preflight,formal,now
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','train']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(a.run_dir/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.mode=='preflight':preflight(a.run_dir)
        else:formal(a.run_dir,a.resume)
    except Exception as exc:
        atomic_json(a.run_dir/'worker_failure.json',dict(state='failed',error=repr(exc),updated_at=now()))
        raise
