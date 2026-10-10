"""One E1 training process (torchrun supplies rank geometry for controlled PPO)."""
from __future__ import annotations
import argparse
import json
from contextlib import nullcontext
import os
from pathlib import Path

import yaml
from e1.configs import CONTROLLED, METHODS
from e1.evaluation import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',type=Path);parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if cfg['experiment_protocol']['method']!=args.method:raise ValueError('Method and frozen config disagree')
    from e1.lifecycle import acquire_run_lock
    guard=acquire_run_lock(args.config.parent) if int(os.environ.get('RANK','0'))==0 else nullcontext()
    with guard:
        if args.method in CONTROLLED:
            from offline2online.trainer import train_from_config
            device=f'cuda:{os.environ.get("LOCAL_RANK","0")}' if args.device.startswith('cuda') else 'cpu'
            overrides={'training':{}}
            if args.resume:overrides['training'].update(resume_checkpoint_path=str(args.resume.resolve()),stop_after_epoch=None)
            if args.stop_after_epoch is not None:overrides['training']['stop_after_epoch']=args.stop_after_epoch
            last=train_from_config(cfg,seed=cfg['experiment_protocol']['training_seed'],device=device,overrides=overrides)
            if int(os.environ.get('RANK','0'))==0:
                atomic_json(args.config.parent/'training_result.json',dict(last_checkpoint=str(last),
                    state='paused' if args.stop_after_epoch is not None else 'completed',
                    resume_checkpoint=str(args.resume) if args.resume else None))
            print(f'E1 checkpoint: {last}',flush=True)
        else:
            if args.resume:
                # Native checkpoints carry full state; no actor-only substitute.
                cfg['resume_checkpoint']=str(args.resume.resolve())
            from e1.native import train
            result=train(cfg,args.config.parent,resume=bool(args.resume),device=args.device)
            print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
