#!/usr/bin/env python3
"""Native tensor-only host smoke; this does NOT certify an E2 adapter or run."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evrptw-rl-model',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    torch.manual_seed(3009);torch.set_num_threads(1)
    result={}
    module=load('e2_pomo_native_audit',ROOT/'benchmarks/native_sources/pomo/NEW_py_ver/CVRP/POMO/CVRPModel.py')
    model=module.CVRPModel(embedding_dim=128,sqrt_embedding_dim=128**.5,encoder_layer_num=6,qkv_dim=16,head_num=8,logit_clipping=10,ff_hidden_dim=512,eval_type='argmax')
    obs=SimpleNamespace(depot_xy=torch.rand(2,1,2),node_xy=torch.rand(2,100,2),node_demand=torch.rand(2,100))
    model.pre_forward(obs);mask=torch.zeros(2,3,101);mask[:,:,7]=-torch.inf
    probs=model.decoder(model.encoded_nodes[:,:3],torch.ones(2,3),mask)
    assert torch.isfinite(probs).all() and (probs[:,:,7]==0).all()
    (-probs[:,:,1].log().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result['pomo']=dict(native_forward_backward=True,customers=100,parameters=sum(p.numel() for p in model.parameters()),plugin_integration_tested=False,real_environment_tested=False)
    module=load('e2_evrptw_rl_audit',args.evrptw_rl_model)
    model=module.EVRPTWRLPolicy(embedding_dim=128);B,T,N=2,3,121
    obs=dict(cus_loc=torch.rand(B,100,2),depot_loc=torch.rand(B,1,2),rs_loc=torch.rand(B,20,2),time_window=torch.stack([torch.zeros(B,N),torch.ones(B,N)],-1),service_time=torch.zeros(B,N),charging_time_ratio=torch.zeros(B,N),remaining_demand=torch.rand(B,T,N),current_time=torch.zeros(B,T),remaining_battery=torch.ones(B,T),remaining_vehicle_ratio=torch.ones(B,T),last_node_idx=torch.zeros(B,T,dtype=torch.long),action_mask=torch.ones(B,T,N,dtype=torch.bool))
    obs['action_mask'][:,:,7]=False;travel=torch.rand(B,N,N)
    state=model.initial_state(B,T);fixed=model.encode_static(obs,travel)
    x,s1=model.logits(obs,travel,state,fixed=fixed);y,s2=model.logits(obs,travel,state,fixed=fixed)
    assert torch.equal(x,y) and torch.isneginf(x[:,:,7]).all()
    (-x.log_softmax(-1)[:,:,1].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result['evrptw_rl_reimplementation']=dict(native_forward_backward=True,repeat_same_state_identical=True,customers=100,stations=20,parameters=sum(p.numel() for p in model.parameters()),plugin_integration_tested=False,real_environment_tested=False,full_trajectory_ppo_replay_tested=False)
    from e1.native import native_status
    result['native_dependency_probe']={m:native_status(m) for m in ('radar','rrnco')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
