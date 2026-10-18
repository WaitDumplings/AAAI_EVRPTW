#!/usr/bin/env python3
"""Reproducible real-data E2 engineering checks; no formal training campaign."""
import argparse, copy, csv, hashlib, json, os, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from e1.native_worker import RadarRunner, RrncoRunner, load_data, instance_for, action_routes, evaluation_rng
from e1.validator import validate_routes


def seed(value=3009):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def parameter_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def workspace(args):
    campaign=ROOT/('results/e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1' if args.method=='radar' else 'results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2')
    original=campaign/'runs'/args.method
    work=args.output/args.method
    work.mkdir(parents=True,exist_ok=True)
    for name in ('native_source','native_data'):
        target=work/name
        if not target.exists():target.symlink_to(original/name,target_is_directory=True)
    cfg=json.loads((original/'native_resolved.json').read_text())
    cfg.update(batch_size=2,n_traj=3)
    return work,cfg,original


def radar(args,work,cfg):
    from e2.adapters.radar import install_radar_agda
    def make(enabled=None):
        seed()
        source=work/'native_source'
        sys.path[:0]=[str(source/'acvrp'),str(source)]
        import ACVRPTrainer as module
        native=module.Model
        if enabled is not None:
            module.Model=lambda **kw:install_radar_agda(native(**kw),enabled=enabled)
        try:runner=RadarRunner(cfg,work,torch.device('cuda:0'))
        finally:module.Model=native
        runner.initial_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        return runner
    base=make();base_initial=parameter_hash(base.model)
    off=make(False);on=make(True)
    assert all(torch.equal(a,b) for a,b in zip(base.initial_rng,on.initial_rng))
    def shared(model):
        return {k.replace('decoder.native.','decoder.'):v for k,v in model.state_dict().items() if '.agda.' not in k}
    assert all(torch.equal(v,shared(on.model)[k]) for k,v in base.model.state_dict().items())
    optids=[id(p) for g in on.optimizer.param_groups for p in g['params']]
    assert len(optids)==len(set(optids)) and set(optids)=={id(p) for p in on.model.parameters()}
    arrays,meta=load_data(work,'train')
    def decode(runner, backward=False):
        runner.model.zero_grad(set_to_none=True)
        records=[]
        handle=runner.model.decoder.register_forward_hook(lambda m,a,o:records.append(o))
        with evaluation_rng(17003009):
            actions,costs,completed=runner.decode(arrays,0,2,3)
        handle.remove()
        probs=torch.stack(records)
        if backward:
            chosen=torch.tensor(actions[:,:,2:],device=probs.device).permute(2,0,1)
            loss=-probs.gather(-1,chosen[...,None]).squeeze(-1).clamp_min(1e-30).log().mean()
            loss.backward()
        grads={k.replace('decoder.native.','decoder.'):v.grad.detach().clone() for k,v in runner.model.named_parameters() if v.grad is not None and '.agda.' not in k}
        checks=[validate_routes(instance_for(arrays,meta,b),action_routes(actions[b,s].tolist()),costs[b,s]) for b in range(2) for s in range(3)]
        assert all(x['valid'] for x in checks),checks
        return probs.detach(),actions,costs,grads,checks
    b=decode(base,True);d=decode(off,True);z=decode(on,True)
    error=lambda a,b:float((a-b).abs().max())
    gradmax=max(error(b[3][k],z[3][k]) for k in b[3])
    assert error(b[0],d[0])==0 and error(b[0],z[0])==0
    assert np.array_equal(b[1],z[1]) and np.array_equal(b[2],z[2])
    assert gradmax<1e-5,gradmax
    plugin_grad=sum(float(p.grad.abs().sum()) for p in on.model.decoder.agda.parameters() if p.grad is not None)
    assert plugin_grad>0
    losses=[]
    for _ in range(3):losses.append(on.train_batch(2))
    u=decode(on)
    assert error(z[0],u[0])>0 if z[0].shape==u[0].shape else True
    # Mask legality is enforced at every environment step below by a decoder hook.
    masked=[]
    def check_mask(m,a,kw,o):
        mask=kw.get('ninf_mask',a[2] if len(a)>2 else None)
        masked.append(float(o[~torch.isfinite(mask)].abs().max()) if (~torch.isfinite(mask)).any() else 0.)
    handle=on.model.decoder.register_forward_hook(check_mask,with_kwargs=True)
    u=decode(on);handle.remove();assert max(masked)==0
    checkpoint=work/'smoke_model.pt';torch.save(on.model.state_dict(),checkpoint)
    restored=make(True);restored.model.load_state_dict(torch.load(checkpoint,weights_only=True))
    restored_result=decode(restored)
    assert error(u[0],restored_result[0])==0
    return dict(status='verified',batch=2,trajectories=3,customers=100,native_training_updates=3,
        host_initial_sha256=base_initial,shared_initial_parameters_equal=True,optimizer_parameter_coverage_once=True,
        plugin_install_preserves_cpu_cuda_rng=True,
        disabled_prob_max_error=error(b[0],d[0]),zero_prob_max_error=error(b[0],z[0]),
        zero_fixed_action_logprob_max_error=error(b[0].clamp_min(1e-30).log(),z[0].clamp_min(1e-30).log()),
        zero_shared_gradient_max_error=gradmax,zero_cost_max_error=float(np.max(np.abs(b[2]-z[2]))),
        initial_plugin_gradient_l1=plugin_grad,plugin_parameter_count=sum(p.numel() for p in on.model.decoder.agda.parameters()),
        checkpoint_reload_prob_max_error=error(u[0],restored_result[0]),masked_probability_max=max(masked),
        updates=losses,route_checks=u[4],post_update_probability_changed=not torch.equal(z[0],u[0]),
        complete_route_test=True)


def rrnco(args,work,cfg):
    from tensordict import TensorDict
    from rl4co.utils.ops import batchify,unbatchify
    from e2.adapters.rrnco_step import StepCritic,run_actions,encode_static,step_ppo_loss
    from caliroute.plugins.slppo import normalized_route_advantages,solution_level_ppo_loss
    seed();runner=RrncoRunner(cfg,work,torch.device('cuda:0'));policy=runner.model.policy
    with torch.random.fork_rng(devices=[0]):critic=StepCritic().cuda()
    arrays,meta=load_data(work,'train')
    keys=('locs','demand_linehaul','distance_matrix','duration_matrix','time_windows','service_time')
    batch=TensorDict({k:torch.tensor(arrays[k][:2],device='cuda') for k in keys},batch_size=[2],device='cuda')
    # All experiments use train phase, instance normalization, FP32 by default.
    policy.train();critic.train()
    with torch.no_grad():
        old=run_actions(policy,runner.env,batch,critic)
        repeat=run_actions(policy,runner.env,batch,critic,actions=old['actions'],neighbors=old['neighbors'])
        unfixed=run_actions(policy,runner.env,batch,critic,actions=old['actions'])
    mask=old['policy_valid']
    delta=(repeat['logp']-old['logp'])[mask]
    frozen_error=float(delta.abs().max());unfixed_error=float((unfixed['logp']-old['logp'])[mask].abs().max())
    assert frozen_error<2e-5,frozen_error
    assert torch.equal(old['mask'],repeat['mask']) and torch.equal(old['policy_valid'],repeat['policy_valid'])
    assert not old['policy_valid'][0].any() and (old['logp'][0]==0).all()
    # Check native policy forced-start order and logits against the new interface.
    from e2.adapters.rrnco_step import fixed_neighbors
    with torch.no_grad(),evaluation_rng(45):
        native_td=runner.env.reset(batch.clone())
        with fixed_neighbors(policy,old['neighbors']):
            native=policy(native_td,runner.env,phase='train',num_starts=3,decode_type='multistart_greedy',return_sum_log_likelihood=False)
        wrapped=run_actions(policy,runner.env,batch,critic,neighbors=old['neighbors'],greedy=True)
    native_actions=native['actions'].T
    assert torch.equal(native_actions,wrapped['actions'])
    native_lp_error=float((native['log_likelihood'].T-wrapped['logp']).abs().max())
    assert native_lp_error<2e-5,native_lp_error
    # Independent physical checks use real road D/T, including implicit return.
    def validate(result,starts):
        costs=result['physical_cost'].detach().cpu().numpy();actions=result['actions'].T.cpu().tolist()
        checks=[validate_routes(instance_for(arrays,meta,i%2),action_routes(a,implicit_final_return=True),costs[i]) for i,a in enumerate(actions)]
        assert all(x['valid'] for x in checks),checks
        return checks
    checks=validate(old,3)
    assert (~old['alive']).any(), 'Need variable-length routes for padding coverage'
    assert not old['policy_valid'][~old['alive']].any()
    rejected={}
    for name,kw in [('truncation',dict(cap=1)),('masked_expert_action',dict(actions=old['actions'].clone(),neighbors=old['neighbors']))]:
        if name=='masked_expert_action':
            # Revisit a forced-start customer at the next decision.
            kw['actions'][1]=kw['actions'][0]
        try:
            with torch.no_grad():run_actions(policy,runner.env,batch,critic,**kw)
        except (RuntimeError,ValueError):rejected[name]=True
        else:raise AssertionError('Invalid sequence accepted: '+name)
    # Explicit BF16 experiment, not an assumption about the formal precision.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        amp=run_actions(policy,runner.env,batch,critic)
        amp_replay=run_actions(policy,runner.env,batch,critic,actions=amp['actions'],neighbors=amp['neighbors'])
    amp_error=float((amp['logp']-amp_replay['logp'])[amp['policy_valid']].abs().max())
    assert amp_error<2e-5,amp_error
    # Train-split external expert data only, verified before teacher forcing.
    expertpath=Path('/data/Maojie/AAAI/AAAI_Dataset/dataset/vrptw/train/Cus100/expert_solutions.csv')
    with expertpath.open() as f:rows=list(csv.DictReader(f))[:2]
    sequences=[];expert_cost=[]
    for i,row in enumerate(rows):
        assert row['instance_id']==meta['instances'][i]['instance_id']
        cost=float(row['objective_distance_km']);routes=json.loads(row['routes_json'])
        assert validate_routes(instance_for(arrays,meta,i),routes,cost)['valid']
        seq=json.loads(row['route_sequence_json'])[1:]
        while seq[-1]==0:seq.pop()
        sequences.append(seq);expert_cost.append(cost)
    expert_actions=torch.zeros(max(map(len,sequences)),2,dtype=torch.long,device='cuda')
    for i,seq in enumerate(sequences):expert_actions[:len(seq),i]=torch.tensor(seq,device='cuda')
    with torch.no_grad():expert_old=run_actions(policy,runner.env,batch,critic,starts=1,actions=expert_actions,start_actions=expert_actions[0])
    assert not expert_old['policy_valid'][0].any()
    validate(expert_old,1)
    policy0=copy.deepcopy(policy.state_dict());critic0=copy.deepcopy(critic.state_dict())
    arms={}
    for arm in ('ppo','slppo'):
        policy.load_state_dict(policy0);critic.load_state_dict(critic0)
        optimizer=torch.optim.Adam([dict(params=policy.parameters(),lr=args.actor_lr),
            dict(params=critic.parameters(),lr=args.critic_lr)])
        losses=[]
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            replay=run_actions(policy,runner.env,batch,critic,actions=old['actions'],neighbors=old['neighbors'])
            loss=step_ppo_loss(replay,old)
            if arm=='slppo':
                costs=old['physical_cost'].reshape(3,2).T
                ref=torch.tensor(expert_cost,device='cuda')
                group=normalized_route_advantages(costs,torch.ones_like(costs,dtype=torch.bool),ref).T.reshape(-1)
                solution=solution_level_ppo_loss(replay['logp'],old['logp'],old['policy_valid'],group)
                expert=run_actions(policy,runner.env,batch,critic,starts=1,actions=expert_actions,neighbors=expert_old['neighbors'],start_actions=expert_actions[0])
                # Positive expert advantage measured against same-instance online
                # group; route clipping reuses the existing portable E1 primitive.
                expert_adv=((costs.mean(-1)-ref)/costs.std(-1,unbiased=False).clamp_min(ref*.01)).clamp(0,3)
                expert_solution=solution_level_ppo_loss(expert['logp'],expert_old['logp'],expert_old['policy_valid'],expert_adv)
                loss=loss+.3*solution.loss+.1*expert_solution.loss
            loss.backward()
            grads=[p.grad for p in policy.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads)
            gradnorm=float(torch.nn.utils.clip_grad_norm_(list(policy.parameters())+list(critic.parameters()),1.))
            assert gradnorm>0 and np.isfinite(gradnorm)
            optimizer.step();losses.append(dict(loss=float(loss.detach()),gradient_norm=gradnorm))
        with torch.no_grad():after=run_actions(policy,runner.env,batch,critic,actions=old['actions'],neighbors=old['neighbors'])
        diff=(after['logp']-old['logp'])[mask];changed=float(diff.abs().max());assert changed>0
        arms[arm]=dict(status='verified',updates=losses,post_update_logprob_change=changed,
            post_update_approx_kl=float((diff.exp()-1-diff).mean()),
            warning='Engineering update only; formal runs need KL guard and LR/pass calibration')
    # Round-trip parameter serialization, rebuild all parameter-dependent caches.
    path=work/'smoke_model.pt';torch.save(dict(policy=policy.state_dict(),critic=critic.state_dict()),path)
    state=torch.load(path,weights_only=True);policy.load_state_dict(state['policy']);critic.load_state_dict(state['critic'])
    with torch.no_grad():reloaded=run_actions(policy,runner.env,batch,critic,actions=old['actions'],neighbors=old['neighbors'])
    assert torch.equal(reloaded['logp'],after['logp'])
    return dict(status='verified',batch=2,trajectories=3,customers=100,actor_lr=args.actor_lr,critic_lr=args.critic_lr,reward_scale_km=100.,fp32_old_replay_logprob_max_error=frozen_error,
        fp32_ratio_max_error=float((delta.exp()-1).abs().max()),fp32_approx_kl=float((delta.exp()-1-delta).mean()),
        unfixed_neighbors_logprob_max_error=unfixed_error,bf16_old_replay_logprob_max_error=amp_error,
        native_greedy_actions_identical=True,native_greedy_logprob_max_error=native_lp_error,
        masks_equal=True,start_major_order_verified=True,forced_start_excluded=True,
        variable_length_padding_tested=True,negative_cases_rejected=rejected,
        dropout_modules=[str(m) for m in policy.modules() if isinstance(m,torch.nn.Dropout)],
        batchnorm_modules=[str(m) for m in policy.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)],
        route_checks=checks,expert_train_only=True,expert_csv_sha256=hashlib.sha256(expertpath.read_bytes()).hexdigest(),
        expert_ids=[r['instance_id'] for r in rows],expert_teacher_forcing_verified=True,
        arms=arms,checkpoint_replay_equal=True,
        scope='PPO/GAE plus portable SL-PPO group/expert losses engineering smoke; NOT complete E1 SL-PPO curriculum or approved E2 protocol')


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['radar','rrnco'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actor-lr',type=float,default=1e-6);p.add_argument('--critic-lr',type=float,default=1e-4);args=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    work,cfg,original=workspace(args)
    result=(radar if args.method=='radar' else rrnco)(args,work,cfg)
    torch.cuda.synchronize();result.update(elapsed_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,original_run=str(original))
    (args.output/(args.method+'_smoke.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
