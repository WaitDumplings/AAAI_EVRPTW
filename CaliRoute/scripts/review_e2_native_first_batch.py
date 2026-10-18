#!/usr/bin/env python3
"""Re-execute only the first E1 native batch from its frozen harness."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--method',choices=['radar','rrnco'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
base=ROOT/('results/e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1' if args.method=='radar' else 'results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2')
run=base/'runs'/args.method
sys.path.insert(0,str(run/'native_harness'))
# Import the FROZEN worker, not today's working tree.
from e1.native_worker import RadarRunner,RrncoRunner
import torch,numpy as np
cfg=json.loads((run/'native_resolved.json').read_text());work=args.output/(args.method+'_first_batch');work.mkdir(parents=True,exist_ok=True)
for name in ['native_source','native_data']:
 if not (work/name).exists():(work/name).symlink_to(run/name,target_is_directory=True)
torch.set_num_threads(1);torch.cuda.set_device(0)
random.seed(cfg['training_seed']);np.random.seed(cfg['training_seed']);torch.manual_seed(cfg['training_seed']);torch.cuda.manual_seed_all(cfg['training_seed'])
runner=(RadarRunner if args.method=='radar' else RrncoRunner)(cfg,work,torch.device('cuda:0'))
h=hashlib.sha256()
for k,v in runner.model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
result=runner.train_batch(64)
old=json.loads((run/'native_train.jsonl').read_text().splitlines()[0])
compare={k:dict(recorded=old[k],reproduced=result[k],absolute_error=abs(old[k]-result[k])) for k in ['loss','best_training_cost_km','decision_steps','completed_rollouts']}
output=dict(method=args.method,historical_initial_checkpoint_available=False,reconstructed_initial_sha256=h.hexdigest(),frozen_worker=sys.modules['e1.native_worker'].__file__,batch=64,n_traj=50,comparison=compare,exact_first_batch_match=all(x['absolute_error']==0 for x in compare.values()))
(args.output/(args.method+'_first_batch.json')).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
