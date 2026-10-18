#!/usr/bin/env python3
"""Frozen E2-A RRNCO worker (distinct from the prepare-only E2-B shell)."""
import argparse
import fcntl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e1.native import atomic_json
from e2.rrnco_native import run_preflight,run_formal,now

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','train']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(a.run_dir/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.mode=='preflight':run_preflight(a.run_dir)
        else:run_formal(a.run_dir,resume=a.resume)
    except Exception as exc:
        atomic_json(a.run_dir/'worker_failure.json',dict(error=repr(exc),mode=a.mode,updated=now()))
        raise
