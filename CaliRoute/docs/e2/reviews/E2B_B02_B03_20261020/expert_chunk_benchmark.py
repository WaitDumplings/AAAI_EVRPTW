import sys,json,time,gc
from pathlib import Path
sys.path.insert(0,'/data/Maojie/AAAI/AAAI_EVRPTW_e2_review/CaliRoute')
import torch
from torch.utils.checkpoint import checkpoint
from e2.rrnco_ppo import PPOExperiment
from e2.rrnco_native import verify_run
from e1.native import atomic_json
camp=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009')
r=camp/'preflight/awbc';cfg,manifest=verify_run(r);exp=PPOExperiment(cfg,r,manifest);pool=exp.imitation
ix=torch.as_tensor(pool.rng.integers(0,len(pool.actions),size=512));neighbors=[]
with torch.no_grad():
 for i in range(0,512,32):neighbors.append(pool.forward(ix[i:i+32])[3])
nb=torch.cat(neighbors);rows=[];base=None;values=None
for chunk in [32,128,512]:
 exp.runner.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
 outputs=[]
 try:
  def call(j,n):return pool.forward(j,n)[0]
  for i in range(0,512,chunk):outputs.append(checkpoint(call,ix[i:i+chunk],nb[i:i+chunk],use_reentrant=False))
  logp=torch.cat(outputs);loss=-logp.mean();loss.backward();torch.cuda.synchronize()
  elapsed=time.perf_counter()-started;grad=exp.policy.decoder.project_node_embeddings.weight.grad.detach().clone()
  if base is None:base=grad.clone();values=logp.detach().clone()
  row=dict(chunk=chunk,seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),gradient_max_error=float((grad-base).abs().max()),reference_gradient_max=float(base.abs().max()),logprob_max_error=float((logp.detach()-values).abs().max()))
  print(json.dumps(row),flush=True);rows.append(row);del logp,loss,outputs,grad
 except torch.cuda.OutOfMemoryError:
  rows.append(dict(chunk=chunk,oom=True));print(rows[-1],flush=True);break
atomic_json(camp/'expert_chunk_benchmark.json',dict(rows=rows,same_sampled_decisions=True,same_neighbor_samples=True,checkpointing=True))
