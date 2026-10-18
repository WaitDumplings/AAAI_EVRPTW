"""Additive VRPTW adapters; native AAFM/NAB and REINFORCE are retained."""
import math
import torch
from torch.utils.checkpoint import checkpoint
from caliroute.plugins.rdi import RoadDistanceInjection
from caliroute.plugins.agda import AdaptiveGraphAttention


def install_rrnco_rdi(policy, *, enabled=True):
    device=next(policy.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        policy.rdi=RoadDistanceInjection(1).to(device)
    policy.e2_enabled=enabled
    def capture(module,args):
        td=args[0]
        # Only immutable static data; never keep a mutable rollout TensorDict.
        policy._e2_static=tuple(td[k] for k in ('distance_matrix','duration_matrix','time_windows','service_time'))
    policy.encoder.register_forward_pre_hook(capture)
    def inject(transposed):
        def hook(module,args,kwargs):
            if not policy.e2_enabled:return args,kwargs
            distance,duration,windows,service=policy._e2_static
            bias=policy.rdi(distance,travel_time=duration,time_windows=windows,service_time=service)[:,0]
            if transposed:bias=bias.transpose(-1,-2)
            kwargs=dict(kwargs,adapt_bias=kwargs['adapt_bias']+bias)
            return args,kwargs
        return hook
    for layer in policy.encoder.net.layers:
        layer.row_encoding_block.attn_free.register_forward_pre_hook(inject(False),with_kwargs=True)
        layer.col_encoding_block.attn_free.register_forward_pre_hook(inject(True),with_kwargs=True)
    return policy


def install_bias_processing():
    """Carry a decision's bias on its logits tensor, not global mutable state."""
    import rrnco.models.decoding as decoding
    if getattr(decoding.process_logits,'e2_bias_aware',False):return
    original=decoding.process_logits
    def process(logits,mask=None,temperature=1.,top_p=0.,top_k=0,tanh_clipping=0,mask_logits=True):
        bias=getattr(logits,'_e2_action_bias',None)
        if bias is not None:
            logits=(torch.tanh(logits)*tanh_clipping if tanh_clipping>0 else logits)+bias
            tanh_clipping=0
        return original(logits,mask,temperature,top_p,top_k,tanh_clipping,mask_logits)
    process.e2_bias_aware=True
    decoding.process_logits=process


def vrptw_features(td,nodes):
    """td batch [B,S]; road/time are native independently normalized fields."""
    b,s=td.batch_size;n=nodes.size(1)
    bi=torch.arange(b,device=nodes.device)[:,None];current=td['current_node']
    road=td['distance_matrix'][:,0];time=td['duration_matrix'][:,0]
    forward=road[bi,current];reverse=road.transpose(-1,-2)[bi,current]
    travel=time[bi,current];backward=time.transpose(-1,-2)[bi,current]
    expand=lambda x:x[:,None].expand(b,s,n)
    to_depot=expand(road[:,:,0]);from_depot=expand(road[:,0])
    return_time=expand(time[:,:,0]);depart_time=expand(time[:,0])
    cap=td['vehicle_capacity'];load=(cap-td['used_capacity_linehaul'])/cap
    demand=td['demand_linehaul']/cap
    visited=td['visited'];feasible=td['action_mask']
    depot=torch.zeros_like(demand);depot[...,0]=1
    remaining=(~visited).to(nodes.dtype)*(1-depot)
    clock=td['current_time'];arrival=clock+travel
    windows=td['time_windows'];service=td['service_time']
    start=torch.maximum(arrival,windows[...,0])
    wait=(windows[...,0]-arrival).clamp_min(0)
    slack=windows[...,1]-start
    return_slack=windows[...,0,1,None]-start-service-return_time
    candidate=torch.stack((forward,reverse,to_depot,from_depot,travel,backward,
        return_time,depart_time,demand,load.expand_as(demand),visited.to(nodes.dtype),
        feasible.to(nodes.dtype),depot,wait,slack,return_slack,service,
        windows[...,1]-windows[...,0],arrival),-1)
    system=torch.stack((load.squeeze(-1),(demand*remaining).sum(-1),
        remaining.sum(-1)/(n-1),clock.squeeze(-1),travel[...,0]),-1)
    mean_unvisited=torch.einsum('bsn,bnd->bsd',remaining,nodes)/remaining.sum(-1,keepdim=True).clamp_min(1)
    tokens=torch.stack((nodes[bi,current],nodes[:,0,None].expand(b,s,-1),
        nodes.mean(1)[:,None].expand(b,s,-1),mean_unvisited),-2)
    return tokens,candidate,system,forward,travel


def install_rrnco_agda(policy, *, enabled=True, chunk=50, activation_checkpointing=True):
    from rl4co.utils.ops import unbatchify
    decoder=policy.decoder
    if decoder.is_dynamic_embedding:raise ValueError('Only the audited static VRPTW host is supported')
    device=next(policy.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        decoder.agda=AdaptiveGraphAttention(decoder.embed_dim,enabled=True,
            enable_delta_k=False,enable_delta_v=False,enable_delta_action_key=True,
            enable_action_bias=True,optimize_dynamic_projections=True,use_agda_v2=False,
            candidate_feature_dim=19,system_feature_dim=5,num_tokens=4).to(device)
    policy.e2_enabled=enabled
    decoder.e2_activation_checkpointing=activation_checkpointing
    decoder.e2_chunk=chunk
    original=decoder.forward
    install_bias_processing()
    def pointer(query,key,value,logit_key,nodes,tokens,candidate,system,mask):
        native=decoder.pointer
        _,_,delta,bias=decoder.agda(nodes,tokens,candidate,system_features=system)
        heads=native._inner_mha(query,key,value,mask)
        glimpse=heads+query;glimpse=native.ffn(glimpse)+glimpse
        logits=torch.bmm(glimpse,logit_key.squeeze(-2).transpose(-2,-1)).squeeze(-2)
        # Native B,1,N result squeezes the query axis for a single start.
        if logits.ndim==2:logits=logits[:,None]
        logits=(logits+torch.einsum('bsd,bsnd->bsn',glimpse,delta))/math.sqrt(glimpse.size(-1))
        return logits,bias
    def forward(td,cached,num_starts=0):
        if not policy.e2_enabled:return original(td,cached,num_starts)
        starts=max(1,num_starts)
        grouped=unbatchify(td,starts) if starts>1 else td.unsqueeze(1)
        q=decoder._compute_q(cached,grouped)
        k,v,lk=decoder._compute_kvl(cached,grouped)
        tokens,features,system,distance,duration=vrptw_features(grouped,cached.node_embeddings)
        mask=grouped['action_mask'].clone()
        logits=[];biases=[]
        for begin in range(0,starts,decoder.e2_chunk):
            end=min(begin+decoder.e2_chunk,starts)
            args=(q[:,begin:end],k,v,lk,cached.node_embeddings,tokens[:,begin:end],
                features[:,begin:end],system[:,begin:end],mask[:,begin:end])
            if decoder.e2_activation_checkpointing and torch.is_grad_enabled():
                score,bias=checkpoint(pointer,*args,use_reentrant=False,preserve_rng_state=True)
            else:score,bias=pointer(*args)
            logits.append(score);biases.append(bias)
        logits=torch.cat(logits,1).float();bias=torch.cat(biases,1)
        # Preserve native alpha/beta, log-exp transform, then defer bias until
        # AFTER native tanh clipping in process_logits, before mask/temperature.
        inductive=(decoder.alpha*distance+decoder.beta*duration).float()
        logits=torch.log(torch.exp(logits-inductive)+1e-6)
        flatten=lambda x:x.transpose(0,1).reshape(-1,x.size(-1))
        logits=flatten(logits);logits._e2_action_bias=flatten(bias)
        return logits,flatten(mask)
    decoder.forward=forward
    return policy
