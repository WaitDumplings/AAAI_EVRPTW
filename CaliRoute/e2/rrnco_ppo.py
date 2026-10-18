"""E2-B common per-action PPO worker; native RRNCO actor, no RDI/AGDA."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict
from e1.native import atomic_json,file_hash,json_hash
from e1.native_worker import (RrncoRunner,load_data,instance_for,action_routes,
    evaluate_runner,save_checkpoint,restore_checkpoint,now,evaluation_rng)
from e1.validator import validate_routes
from e2.adapters.rrnco_step import StepCritic,run_actions,gae
from e2.e1_solution_bridge import load_solution_functions
from e2.rrnco_native import seed_all,verify_run,BASE_INITIAL_SHA
from caliroute.plugins.slppo import solution_level_ppo_loss,clipped_route_surrogate


COMMON=dict(online_rollout_iterations=1500,batch_size=64,n_traj=50,ppo_passes=4,
    num_minibatches=4,microbatch_instances=16,actor_lr=1e-6,critic_lr=1e-4,
    gamma=1.,gae_lambda=.95,clip=.2,value_coef=.5,entropy_coef=.01,
    weight_decay=0.,grad_clip=1.,target_kl=.02,reward_scale_km=100.,precision='FP32',
    lr_schedule='constant',eval_every_iterations=50,checkpoint_every_iterations=5,
    seed=3009,initialization='scratch',decode_cap=201,validation_count=1000,
    validation_batch=8,test_count=1000,test_batch=1,K=50,inference_seed=17003009,
    advantage_normalization='all valid policy decisions in the current global rollout',
    loss_reduction='mean valid decisions for policy/entropy; mean alive decisions for value',
    kl_guard='before each minibatch update; skip it and all remaining passes if mean exp(logratio)-1-logratio exceeds target',
    sampler='dedicated numpy default_rng(seed), uniform with replacement; fixed B64 each iteration',
    forced_start='native POMO starts1..50; excluded from policy/entropy, included in GAE/value',
    truncated='error, no silently successful truncated routes',
    optimizer_steps_max=24000,instance_exposures=96000,rollouts=4800000,
    native_road_inputs_retained=True,rdi=False,agda=False)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ExpertPool:
    def __init__(self,path,arrays,metadata):
        self.records={};mapping={x['instance_id']:i for i,x in enumerate(metadata['instances'])}
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                identity=row['instance_id']
                if identity not in mapping or identity in self.records:raise ValueError('Expert identity invalid')
                cost=float(row['objective_distance_km']);routes=json.loads(row['routes_json'])
                check=validate_routes(instance_for(arrays,metadata,mapping[identity]),routes,cost)
                if not check['valid']:raise ValueError((identity,check))
                seq=json.loads(row['route_sequence_json'])
                if seq[0]!=0:raise ValueError('Expert missing initial depot')
                seq=seq[1:]
                # Native done includes the depot's visited bit. A single-tour
                # expert needs its explicit final depot; otherwise completion
                # accidentally depends on another route supplying batch padding.
                if not seq or seq[-1]!=0:raise ValueError('Expert missing final depot')
                # Validate the very representation used by teacher forcing.
                check=validate_routes(instance_for(arrays,metadata,mapping[identity]),
                    action_routes(seq,implicit_final_return=True),cost)
                if not check['valid']:raise ValueError((identity,check))
                self.records[identity]=SimpleNamespace(length=len(seq),actions=seq,
                    observations=[],objective_distance_km=cost)
        if set(self.records)!=set(mapping):raise ValueError('The frozen pool must cover all train IDs')
    def reference_objective(self,key):return self.records[key].objective_distance_km
    def trajectory_for_instance(self,key):return self.records.get(key)


def subset_rollout(old,indices,batch,starts):
    flat=(torch.arange(starts,device=indices.device)[:,None]*batch+indices[None]).reshape(-1)
    result={k:v[:,flat] for k,v in old.items() if k in ('actions','logp','value','entropy','policy_valid','alive','terminal','reward','mask')}
    result['neighbors']=old['neighbors'][indices];result['physical_cost']=old['physical_cost'][flat]
    return result,flat


class PPOExperiment:
    def __init__(self,cfg,run,manifest):
        self.cfg=cfg;self.run=run;self.manifest=manifest;self.p=cfg['ppo_protocol'];seed_all(cfg)
        self.runner=RrncoRunner(cfg,run,torch.device('cuda:0'))
        if state_hash(self.runner.model)!=BASE_INITIAL_SHA:raise ValueError('Native host initial hash mismatch')
        self.actor_initial=state_hash(self.runner.model.policy)
        with torch.random.fork_rng(devices=[0]):self.runner.model.e2_critic=StepCritic().to('cuda')
        self.critic=self.runner.model.e2_critic;self.critic_initial=state_hash(self.critic)
        self.policy=self.runner.model.policy;self.env=self.runner.env
        self.runner.optimizer=torch.optim.Adam([
            dict(params=self.policy.parameters(),lr=self.p['actor_lr']),
            dict(params=self.critic.parameters(),lr=self.p['critic_lr'])],weight_decay=self.p['weight_decay'])
        self.runner.scheduler=torch.optim.lr_scheduler.LambdaLR(self.runner.optimizer,lambda step:1.)
        ids=[id(p) for g in self.runner.optimizer.param_groups for p in g['params']]
        if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in self.runner.model.parameters()}:
            raise ValueError('Actor/critic optimizer coverage mismatch')
        self.arrays,self.metadata=load_data(run,'train')
        self.online_rng=np.random.default_rng(3009);self.minibatch_rng=np.random.default_rng(13009)
        self.incumbent={};self.iteration=0;self.optimizer_steps=0;self.decisions=0
        self.expert_steps=0;self.expert_routes=0;self.best=None
        self.train_wall_s=0.;self.evaluation_wall_s=0.;self.total_active_wall_s=0.
        self.experts=None;self.expert_neighbors={};self.current_indices=None
        if cfg['e2b_method']=='slppo':
            if file_hash(run/'expert_pool.csv')!=cfg['expert_sha256']:raise ValueError('Changed train expert pool')
            self.experts=ExpertPool(run/'expert_pool.csv',self.arrays,self.metadata)
            self.e1,self.e1_evidence=load_solution_functions(self.teacher_callback)
            self.method_cfg=deepcopy(cfg['e1_method_config']);self.e1['_apply_solution_level_aliases'](self.method_cfg)
            self.e1['_validate_e1_solution_contract'](self.method_cfg)
        self.runner.sampler_state=self.method_state
        self.runner.restore_sampler=self.restore_method_state
        self.runner.architecture.update(e2b_method=cfg['e2b_method'],critic='StepCritic(current128+mean128+time+capacity)',
            actor_initial_sha256=self.actor_initial,critic_initial_sha256=self.critic_initial,
            native_host_initial_sha256=BASE_INITIAL_SHA,ppo_protocol=self.p)

    def method_state(self):
        return dict(online_rng=self.online_rng.bit_generator.state,minibatch_rng=self.minibatch_rng.bit_generator.state,
            incumbent=deepcopy(self.incumbent),iteration=self.iteration,optimizer_steps=self.optimizer_steps,
            decisions=self.decisions,expert_steps=self.expert_steps,expert_routes=self.expert_routes,best=self.best,
            train_wall_s=self.train_wall_s,evaluation_wall_s=self.evaluation_wall_s,total_active_wall_s=self.total_active_wall_s,
            train_sha256=self.manifest['datasets']['train']['npz_sha256'],expert_sha256=self.cfg.get('expert_sha256'))

    def restore_method_state(self,state):
        if state['train_sha256']!=self.manifest['datasets']['train']['npz_sha256'] or state['expert_sha256']!=self.cfg.get('expert_sha256'):
            raise ValueError('Method state train/expert identities differ')
        allowed={x['instance_id'] for x in self.metadata['instances']}
        if any(k not in allowed or not np.isfinite(v) or v<0 for k,v in state['incumbent'].items()):raise ValueError('Invalid cost-only incumbent')
        self.online_rng.bit_generator.state=state['online_rng'];self.minibatch_rng.bit_generator.state=state['minibatch_rng']
        for k in ['incumbent','iteration','optimizer_steps','decisions','expert_steps','expert_routes','best','train_wall_s','evaluation_wall_s','total_active_wall_s']:setattr(self,k,state[k])

    def batch(self,indices):
        keys=('locs','demand_linehaul','distance_matrix','duration_matrix','time_windows','service_time')
        return TensorDict({k:torch.as_tensor(self.arrays[k][indices],device='cuda') for k in keys},batch_size=[len(indices)],device='cuda')

    def rollout(self,batch,**kw):
        return run_actions(self.policy,self.env,batch,self.critic,starts=self.p['n_traj'],
            cap=self.p['decode_cap'],allow_terminal_padding=True,**kw)

    def teacher_callback(self,agent,candidates,device,chunk_size,**kwargs):
        # Whole native routes share one static encoding. Different instances may
        # have different lengths; padding is terminal and excluded from means.
        means=[]
        for start in range(0,len(candidates),4):
            group=candidates[start:start+4];indices=[self.current_indices[c.env_idx] for c in group]
            sequences=[self.experts.trajectory_for_instance(self.metadata['instances'][i]['instance_id']).actions for i in indices]
            actions=torch.zeros(max(map(len,sequences)),len(group),dtype=torch.long,device='cuda')
            for j,seq in enumerate(sequences):actions[:len(seq),j]=torch.tensor(seq,device='cuda')
            saved=[self.expert_neighbors.get(c.env_idx) for c in group]
            neighbors=torch.stack(saved) if all(x is not None for x in saved) else None
            out=run_actions(self.policy,self.env,self.batch(indices),self.critic,starts=1,
                actions=actions,start_actions=actions[0],neighbors=neighbors,cap=self.p['decode_cap'],allow_terminal_padding=True)
            for j,c in enumerate(group):self.expert_neighbors[c.env_idx]=out['neighbors'][j]
            valid=out['policy_valid'];means.append(out['logp'].sum(0)/valid.sum(0).clamp_min(1))
            if torch.is_grad_enabled():
                self.expert_steps+=int(valid.sum());self.expert_routes+=len(group)
            costs=np.asarray([self.experts.reference_objective(self.metadata['instances'][i]['instance_id']) for i in indices])
            if not np.allclose(out['physical_cost'].detach().cpu().numpy(),costs,rtol=2e-5,atol=.01):raise ValueError('Expert physical teacher-forcing cost mismatch')
        return torch.cat(means)

    def prepare_solution(self,old,indices):
        b=len(indices);s=self.p['n_traj'];t=old['actions'].size(0)
        reshape=lambda x:x.reshape(t,s,b).transpose(1,2)
        costs=old['physical_cost'].reshape(s,b).T.detach().cpu().numpy()
        envs=[SimpleNamespace(instance=SimpleNamespace(instance_id=self.metadata['instances'][i]['instance_id'])) for i in indices]
        batch=SimpleNamespace(actions=reshape(old['actions']),old_logprobs=reshape(old['logp']),
            dones=torch.ones((1,b,s),dtype=torch.bool,device='cuda'),
            final_infos=[dict(objective_distance_km=c,success=np.ones(s,bool),served_customers=np.full(s,100)) for c in costs])
        snapshot=dict(self.incumbent);self.current_indices=list(indices);self.expert_neighbors={}
        adv,feasible,info=self.e1['_solution_level_advantage_tensors'](batch,self.method_cfg,envs,self.experts,'cuda',policy_best_objectives=snapshot)
        candidates,extra=self.e1['_prepare_sl_expert_candidates'](None,batch,self.method_cfg,envs,self.experts,snapshot,'cuda')
        for env,cost in zip(envs,costs):
            key=env.instance.instance_id;self.incumbent[key]=min(self.incumbent.get(key,float('inf')),float(cost.min()))
        return adv.T.reshape(-1),candidates,dict(info,**extra)

    def iteration_step(self,*,audit=False):
        import time
        begun=time.perf_counter();p=self.p;b=p['batch_size'];s=p['n_traj']
        indices=self.online_rng.integers(0,len(self.metadata['instances']),size=b)
        batch=self.batch(indices);self.runner.model.train()
        with torch.no_grad():old=self.rollout(batch)
        self.decisions+=int(old['policy_valid'].sum())
        advantage,returns=gae(old,p['gamma'],p['gae_lambda'],p['reward_scale_km'])
        valid=old['policy_valid'];a=advantage[valid]
        advantage=(advantage-a.mean())/a.std(unbiased=False).clamp_min(1e-8)
        route_adv=None;candidates=[];method_info={}
        if self.experts is not None:route_adv,candidates,method_info=self.prepare_solution(old,indices)
        report={};kl_values=[];step_count=0;stop=False
        if audit:
            with torch.no_grad():replay=self.rollout(batch,actions=old['actions'],neighbors=old['neighbors'])
            error=(old['logp']-replay['logp'])[valid].abs()
            report['replay_error_quantiles']=torch.quantile(error,torch.tensor([0.,.5,.95,1.],device='cuda')).tolist()
            report['replay_ratio_max_error']=float(((replay['logp']-old['logp'])[valid].exp()-1).abs().max())
            if float(error.max())>2e-5:raise ValueError('Old/replay mismatch')
            # Physical validation of all sampled routes in short/full preflight.
            acts=old['actions'].reshape(-1,s,b).permute(2,1,0).cpu().numpy()
            costs=old['physical_cost'].reshape(s,b).T.cpu().numpy()
            for j,index in enumerate(indices):
                inst=instance_for(self.arrays,self.metadata,index)
                for k in range(s):
                    check=validate_routes(inst,action_routes(acts[j,k],implicit_final_return=True),float(costs[j,k]))
                    if not check['valid']:raise ValueError(check)
            del replay
        for epoch in range(p['ppo_passes']):
            for group in np.array_split(self.minibatch_rng.permutation(b),p['num_minibatches']):
                ix=torch.as_tensor(group,device='cuda');mini,flat=subset_rollout(old,ix,b,s)
                mb=batch[ix];new=self.rollout(mb,actions=mini['actions'],neighbors=mini['neighbors'])
                mask=mini['policy_valid'];delta=new['logp'][mask]-mini['logp'][mask]
                kl=float((delta.double().exp()-1-delta.double()).mean().detach());kl_values.append(kl)
                if not np.isfinite(kl) or kl>p['target_kl']:
                    stop=True;del new;break
                ratio=delta.exp();adv=advantage[:,flat][mask]
                policy=-torch.minimum(ratio*adv,ratio.clamp(1-p['clip'],1+p['clip'])*adv).mean()
                alive=mini['alive'];value=.5*(new['value'][alive]-returns[:,flat][alive]).square().mean()
                entropy=new['entropy'][mask].mean()
                loss=policy+p['value_coef']*value-p['entropy_coef']*entropy
                if route_adv is not None:
                    solution=solution_level_ppo_loss(new['logp'],mini['logp'],mask,route_adv[flat],clip_coef=.2)
                    loss=loss+self.method_cfg['offline']['sl_coef']*solution.loss
                self.runner.optimizer.zero_grad(set_to_none=True);loss.backward()
                selected=[c for c in candidates if c.env_idx in set(group.tolist())]
                if selected:
                    current=self.teacher_callback(None,selected,'cuda',128)
                    previous=torch.tensor([c.old_mean_logprob for c in selected],device='cuda')
                    advantages=torch.tensor([c.advantage for c in selected],device='cuda')
                    aux=clipped_route_surrogate(current-previous,advantages,torch.ones_like(advantages,dtype=torch.bool),clip_coef=.2).loss
                    if audit and 'expert_gradient_l2' not in report:
                        check=torch.autograd.grad(aux,self.policy.decoder.project_node_embeddings.weight,retain_graph=True)[0]
                        report['expert_gradient_l2']=float(check.norm())
                        if not torch.isfinite(check).all() or not bool(check.abs().sum()>0):raise ValueError('Missing expert gradient')
                    (self.method_cfg['offline']['sl_coef']*aux).backward()
                norm=torch.nn.utils.clip_grad_norm_(self.runner.model.parameters(),p['grad_clip'],error_if_nonfinite=True)
                self.runner.optimizer.step();self.optimizer_steps+=1;step_count+=1
                if selected and audit and 'expert_probability_change' not in report:
                    with torch.no_grad():changed=self.teacher_callback(None,selected,'cuda',128)
                    report['expert_probability_change']=float((changed-current.detach()).abs().max())
                    if report['expert_probability_change']==0:raise ValueError('Expert policy did not change')
                report.update(policy_loss=float(policy.detach()),value_loss=float(value.detach()),entropy=float(entropy.detach()),grad_norm=float(norm))
                del new,loss
            if stop:break
        self.iteration+=1;torch.cuda.synchronize();elapsed=time.perf_counter()-begun
        self.train_wall_s+=elapsed;self.total_active_wall_s+=elapsed
        report.update(iteration=self.iteration,online_rollout_iterations=self.iteration,instance_exposures=self.iteration*b,
            complete_rollouts=self.iteration*b*s,valid_decision_steps=self.decisions,optimizer_steps_actual=self.optimizer_steps,
            optimizer_steps_this_iteration=step_count,kl_guard_triggered=stop,kl_max=max(kl_values),
            expert_backward_steps=self.expert_steps,expert_backward_routes=self.expert_routes,
            training_wall_s=self.train_wall_s,iteration_wall_s=elapsed,mean_cost_km=float(old['physical_cost'].mean()),
            mean_best_cost_km=float(old['physical_cost'].reshape(s,b).min(0).values.mean()),
            instance_indices_sha256=json_hash(indices.tolist()),incumbent_count=len(self.incumbent),
            method_diagnostics=method_info,updated_at=now())
        return report

    def save(self,name):
        counters=dict(online_rollout_iterations=self.iteration,instance_exposures=self.iteration*self.p['batch_size'],
            optimizer_steps_actual=self.optimizer_steps,valid_decision_steps=self.decisions,
            expert_backward_steps=self.expert_steps,training_wall_time_s=self.train_wall_s,evaluation_wall_time_s=self.evaluation_wall_s)
        return save_checkpoint(self.runner,self.cfg,self.run,self.manifest,counters,self.best,name=name)

    def restore(self,path):return restore_checkpoint(self.runner,self.cfg,self.manifest,path)
