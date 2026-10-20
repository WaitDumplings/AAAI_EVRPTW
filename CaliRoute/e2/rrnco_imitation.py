"""E1 AWBC/BC loss bodies with native RRNCO expert decision states."""
import ast
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from e2.adapters.rrnco_step import encode_static,step_logits


def load_losses(statistics):
    path=Path(__file__).resolve().parents[1]/'offline2online/offline_data.py'
    source=path.read_text();tree=ast.parse(source)
    names={'compute_bc_loss','compute_awbc_loss'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    namespace=dict(np=np,torch=torch,_expert_policy_statistics=statistics)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace,dict(source=str(path),sha256=hashlib.sha256(source.encode()).hexdigest(),
        unchanged_functions=sorted(names),host_replaced_functions=['_expert_policy_statistics'],
        dapg='E1 ablation/dapg.py delegates directly to compute_bc_loss')


class NativeExpertSteps:
    """Uniform sampling over valid native expert decisions, with replacement.

    Cache only parameter-independent dynamic state. Every forward recomputes
    native encoding from static physical input; no cached model embeddings.
    Forced initial customer and post-done padding are excluded, as in B01/B04.
    """
    dynamic=('current_node','current_route_length','current_time',
             'used_capacity_linehaul','used_capacity_backhaul','visited','done','action_mask')

    def __init__(self,exp):
        self.exp=exp;self.rng=np.random.default_rng(3009+17000)
        blocks={k:[] for k in self.dynamic};instance_indices=[];actions=[];times=[]
        with torch.no_grad():
            for begin in range(0,len(exp.metadata['instances']),128):
                ids=list(range(begin,min(begin+128,len(exp.metadata['instances']))))
                records=[exp.experts.trajectory_for_instance(exp.metadata['instances'][i]['instance_id']) for i in ids]
                seq=torch.zeros(max(r.length for r in records),len(ids),dtype=torch.long,device='cuda')
                for j,r in enumerate(records):seq[:r.length,j]=torch.tensor(r.actions,device='cuda')
                td=exp.env.reset(exp.batch(ids).clone());cpu_ids=torch.tensor(ids)
                for t,action in enumerate(seq):
                    valid=~td['done'].reshape(-1)
                    if not td['action_mask'].gather(-1,action[:,None]).all():raise ValueError('Invalid expert state cache')
                    if t and valid.any():
                        mask=valid.cpu();instance_indices.append(cpu_ids[mask]);actions.append(action[valid].cpu())
                        times.append(torch.full((int(mask.sum()),),t,dtype=torch.long))
                        for k in self.dynamic:blocks[k].append(td[k][valid].cpu().clone())
                    td.set('action',action);td=exp.env.step(td)['next']
                if not td['done'].all():raise ValueError('Incomplete expert cache route')
        self.states={k:torch.cat(v) for k,v in blocks.items()}
        self.instances=torch.cat(instance_indices);self.actions=torch.cat(actions);self.times=torch.cat(times)
        costs=[exp.experts.reference_objective(m['instance_id']) for m in exp.metadata['instances']]
        self.objectives=torch.tensor(costs,dtype=torch.float32)[self.instances]
        self.last_indices=None
        self.losses,self.evidence=load_losses(self.statistics)
        self.evidence.update(valid_native_decisions=len(self.actions),expert_sampler_seed=20009,
            excluded='external first customer and post-terminal depot padding',
            reduction='unchanged E1 global sampled-step loss; batch512 with activation checkpointed chunks32')

    def sample_step_batch_with_objectives(self,batch_size):
        ix=self.rng.integers(0,len(self.actions),size=max(1,int(batch_size)))
        self.last_indices=ix.copy()
        return {'step_indices':ix},self.actions[ix,None],self.objectives[ix]

    def sample_step_batch(self,batch_size):
        obs,act,_=self.sample_step_batch_with_objectives(batch_size)
        return obs,act

    def forward(self,ix,neighbors=None):
        ix=ix.cpu();ids=self.instances[ix].numpy()
        td=self.exp.env.reset(self.exp.batch(ids).clone())
        cache,samples=encode_static(self.exp.policy,td,neighbors)
        for k in self.dynamic:td.set(k,self.states[k][ix].to('cuda'))
        lp,mask=step_logits(self.exp.policy,td,cache,1)
        action=self.actions[ix].to('cuda')
        if not mask.gather(-1,action[:,None]).all():raise ValueError('Expert action mask changed')
        selected=lp.gather(-1,action[:,None]).squeeze(-1)
        entropy=-(lp.exp()*torch.where(mask,lp,0.)).sum(-1)
        return selected,entropy,lp.argmax(-1),samples

    def statistics(self,agent,obs,action,microbatch_size=None):
        del agent
        ix=torch.as_tensor(obs['step_indices'],dtype=torch.long)
        if not torch.equal(self.actions[ix],action.detach().cpu().reshape(-1)):raise ValueError('Expert sampled actions differ')
        chunk=int(microbatch_size or 32);outputs=[]
        from torch.utils.checkpoint import checkpoint
        def forward(j):return self.forward(j)[:3]
        for start in range(0,len(ix),chunk):
            part=ix[start:start+chunk]
            outputs.append(checkpoint(forward,part,use_reentrant=False) if torch.is_grad_enabled() else forward(part))
        return tuple(torch.cat([o[k] for o in outputs]) for k in range(3))

    def loss(self,method,baseline=None):
        cfg=self.exp.method_cfg['offline']
        args=dict(batch_size=int(cfg['bc_batch_size']),device='cuda',microbatch_size=int(cfg['expert_microbatch_size']))
        if method=='awbc':args.update(eta=cfg['awbc_eta'],normalize=cfg['awbc_normalize'],baseline=cfg['awbc_baseline'],baseline_objective=baseline)
        fn=self.losses['compute_awbc_loss' if method=='awbc' else 'compute_bc_loss']
        loss,info=fn(None,self,**args)
        self.exp.expert_steps+=info['bc_steps']
        self.exp.expert_routes+=int(torch.unique(self.instances[self.last_indices]).numel())
        return loss,info

    def audit(self):
        # Compare complete teacher forcing to cached prefix-state probabilities,
        # and full/chunked encoder gradients with identical discrete neighbors.
        from e2.adapters.rrnco_step import run_actions
        from e1.native_worker import rng_state,restore_rng
        saved=rng_state()
        identity=0;record=self.exp.experts.trajectory_for_instance(self.exp.metadata['instances'][identity]['instance_id'])
        seq=torch.tensor(record.actions,device='cuda')[:,None]
        with torch.no_grad():
            out=run_actions(self.exp.policy,self.exp.env,self.exp.batch([identity]),self.exp.critic,starts=1,
                actions=seq,start_actions=seq[0],allow_terminal_padding=True)
            ids=torch.where(self.instances==identity)[0][:32]
            neighbors=out['neighbors'].expand(len(ids),-1,-1).contiguous()
            cached=self.forward(ids,neighbors)[0]
            expected=out['logp'][self.times[ids].to('cuda'),0]
            error=float((cached-expected).abs().max())
        if error>2e-5:raise ValueError(('Prefix/full route probability mismatch',error))
        param=self.exp.policy.decoder.project_node_embeddings.weight
        full=self.forward(ids,neighbors)[0];g1=torch.autograd.grad(full.mean(),param)[0]
        parts=[self.forward(ids[i:i+8],neighbors[i:i+8])[0] for i in range(0,len(ids),8)]
        g2=torch.autograd.grad(torch.cat(parts).mean(),param)[0]
        grad_error=float((g1-g2).abs().max());scale=max(float(g1.abs().max()),1.)
        if grad_error>2e-5*scale:raise ValueError(('Expert microbatch gradient mismatch',grad_error))
        restore_rng(saved)
        return dict(passed=True,cached_vs_full_route_logprob_max_error=error,
                    microbatch_gradient_max_error=grad_error,valid_steps=len(self.actions))
