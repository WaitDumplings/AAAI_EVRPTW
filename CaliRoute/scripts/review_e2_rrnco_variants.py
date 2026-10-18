#!/usr/bin/env python3
"""Real VRPTW100 short tests for native RRNCO module integration."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from e1.native import atomic_json
from e1.native_worker import RrncoRunner,evaluate_runner
from e2.rrnco_native import make_runner,seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['rdi','agda'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    baseline=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2/runs/rrnco')
    work=a.output/a.variant;work.mkdir(parents=True,exist_ok=True)
    for name in ['native_source','native_data']:
        if not (work/name).exists():(work/name).symlink_to(baseline/name,target_is_directory=True)
    cfg=json.loads((baseline/'native_resolved.json').read_text())
    cfg.update(batch_size=2,n_traj=3,e2_variant=a.variant,agda_activation_checkpointing=True,
        agda_trajectory_chunk_size=3,eval_limit=2,eval_k=3,eval_batch_size=2)
    cfg['experiment_protocol'].update(smoke_only=True,hardware_preflight_only=True)
    seed_all(cfg);base=RrncoRunner(cfg,work,torch.device('cuda:0'))
    base_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    seed_all(cfg);off=make_runner(cfg,work);off.model.policy.e2_enabled=False
    seed_all(cfg);on=make_runner(cfg,work)
    assert all(torch.equal(x,y) for x,y in zip(base_rng,(torch.get_rng_state(),torch.cuda.get_rng_state())))
    def train(runner):
        grads={}
        def capture(*args):
            for k,v in runner.model.named_parameters():
                if v.grad is not None:grads[k]=v.grad.detach().clone()
        hook=runner.optimizer.register_step_pre_hook(capture)
        seed_all(cfg);result=runner.train_batch(2);hook.remove()
        return result,grads
    expected,bg=train(base);disabled,dg=train(off);actual,ag=train(on)
    for result in [disabled,actual]:
        for key in ['loss','best_training_cost_km','decision_steps','completed_rollouts']:
            assert abs(result[key]-expected[key])<1e-5,(key,result[key],expected[key])
    err=max(float((v-ag[k]).abs().max()) for k,v in bg.items())
    norm=sum(float(v.double().square().sum()) for v in bg.values())**.5
    relative=(sum(float((v-ag[k]).double().square().sum()) for k,v in bg.items())**.5)/max(norm,1e-12)
    off_relative=(sum(float((v-dg[k]).double().square().sum()) for k,v in bg.items())**.5)/max(norm,1e-12)
    print(dict(gradient_abs_max=err,gradient_relative_l2=relative,disabled_relative_l2=off_relative),flush=True)
    assert relative<1e-5 and off_relative<1e-5,(relative,off_relative)
    plugin_grad=sum(float(v.abs().sum()) for k,v in ag.items() if '.rdi.' in k or '.agda.' in k)
    assert plugin_grad>0
    updates=[actual,on.train_batch(2),on.train_batch(2)]
    result=evaluate_runner(on,cfg,work,split='val',checkpoint_id='smoke',
        output=work/'validation.jsonl',inference_seed=17003009)
    assert result['feasible_instances']==2 and result['cost_mismatch_candidates']==0
    torch.save(on.model.state_dict(),work/'smoke.pt')
    seed_all(cfg);restored=make_runner(cfg,work)
    restored.model.load_state_dict(torch.load(work/'smoke.pt',weights_only=True))
    again=evaluate_runner(restored,cfg,work,split='val',checkpoint_id='smoke',
        output=work/'reload_validation.jsonl',inference_seed=17003009)
    assert result['mean_distance_km']==again['mean_distance_km']
    report=dict(passed=True,variant=a.variant,batch=2,n_traj=3,baseline_first_batch=expected,
        enabled_first_batch=actual,disabled_first_batch=disabled,shared_gradient_max_error=err,
        shared_gradient_relative_l2=relative,disabled_gradient_relative_l2=off_relative,
        plugin_gradient_l1=plugin_grad,plugin_rng_preserved=True,optimizer_coverage=True,
        updates=updates,validation=result,reload_validation=again)
    atomic_json(work/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
