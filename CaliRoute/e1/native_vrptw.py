"""VRPTW task adapters for the native baselines; no CaliRoute model plugins.

All time quantities share one instance horizon unit, relative to depot opening.
Distances retain their separate physical-km objective. RADAR retains its encoder,
decoder and REINFORCE loss, with task-specific node/time inputs and feasibility.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

TIME_EPS = 1e-7


def checkpoint_rrnco_encoder(policy):
    """Recompute encoder-layer activations; retain native weights and batch loss."""
    from torch.utils.checkpoint import checkpoint
    for layer in policy.encoder.net.layers:
        original = layer.forward

        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = forward


def convert_vrptw_instances(instances, expected_customers):
    from e1.native import convert_cvrp_instances, json_hash
    instances=list(instances)
    arrays, ids=convert_cvrp_instances(instances,expected_customers=expected_customers)
    extra={k:[] for k in ('travel_time_s','tw_s','service_s','working_window_s','duration_matrix','time_windows','service_time')}
    for instance, identity in zip(instances,ids):
        n=instance.num_customers
        travel=np.asarray(instance.travel_time_matrix_s,dtype=np.float32)
        tw=np.asarray(instance.tw_s,dtype=np.float32)
        service=np.asarray(instance.service_time_s,dtype=np.float32)
        opening,closing=float(instance.working_start_s),float(instance.working_end_s)
        horizon=closing-opening
        if travel.shape!=(n+1,n+1) or tw.shape!=(n,2) or service.shape!=(n,) or not np.isfinite([opening,closing]).all() or horizon<=0:
            raise ValueError('VRPTW requires authoritative travel seconds, customer windows/service and a positive horizon')
        if not all(np.isfinite(x).all() for x in (travel,tw,service)) or (travel<0).any() or (service<0).any() or (tw[:,1]<tw[:,0]).any():
            raise ValueError('Invalid VRPTW time arrays')
        starts=np.maximum(opening+travel[0,1:],tw[:,0])
        if (starts>tw[:,1]+.01).any() or (starts+service+travel[1:,0]>closing+.01).any():
            raise ValueError('A customer cannot be served by a direct depot round trip; forced starts are unsafe')
        windows=np.concatenate([np.array([[opening,closing]]),tw])
        values=dict(travel_time_s=travel,tw_s=tw,service_s=service,working_window_s=np.array([opening,closing]),
                    duration_matrix=travel/horizon,time_windows=(windows-opening)/horizon,
                    service_time=np.concatenate([[0.],service])/horizon)
        for k,v in values.items():extra[k].append(np.asarray(v,dtype=np.float32))
        identity['semantic_sha256']=json_hash({'cvrp':identity['semantic_sha256'],
            'travel':travel.tolist(),'tw':tw.tolist(),'service':service.tolist(),'working':[opening,closing]})
    arrays.update({k:np.stack(v) for k,v in extra.items()})
    arrays['locs']=np.concatenate([arrays['depot'][:,None],arrays['locs']],axis=1)
    arrays['demand_linehaul']=arrays['demand']/arrays['capacity'][:,None]
    return arrays,ids


def rrnco_environment(base):
    class RoadVRPTWEnv(base):
        @staticmethod
        def get_action_mask(td):
            current=td['current_node']
            idx=torch.arange(current.shape[0],device=current.device)
            arrival=td['current_time']+td['duration_matrix'][idx,current]
            start=torch.maximum(arrival,td['time_windows'][...,0])
            finish=start+td['service_time']+td['duration_matrix'][...,0]
            feasible=(start<=td['time_windows'][...,1]+TIME_EPS) & (finish<=td['time_windows'][:,0,1,None]+TIME_EPS)
            feasible &= (td['demand_linehaul']+td['used_capacity_linehaul']<=td['vehicle_capacity']+1e-6) & ~td['visited']
            done=td.get('done',torch.zeros_like(current,dtype=torch.bool)).reshape_as(current)
            feasible[:,0]=(current!=0)|done
            if not feasible.any(-1).all(): raise RuntimeError('VRPTW empty action mask; cannot silently repair')
            return feasible
    return RoadVRPTWEnv


def radar_environment(base):
    class RoadVRPTWEnv(base):
        def _load_fixed_dataset(self,path):
            super()._load_fixed_dataset(path)
            with np.load(Path(path),allow_pickle=False) as data:
                self.tw_arrays={k:np.array(data[k]) for k in ('duration_matrix','time_windows','service_time')}
        def set_time_data(self,arrays,start=None,stop=None,indices=None):
            for key in ('duration_matrix','time_windows','service_time'):
                value=arrays[key][indices] if indices is not None else arrays[key][start:stop]
                setattr(self,key,torch.as_tensor(value,dtype=torch.float32,device=self.problems.device))
            self.reset_state.time_features=torch.cat([self.time_windows,self.service_time[...,None]],-1)
        def load_fixed_problems(self,batch_size):
            indices=self.fixed_order[self.fixed_cursor:self.fixed_cursor+batch_size].copy()
            super().load_fixed_problems(batch_size)
            self.set_time_data(self.tw_arrays,indices=indices)
        def reset(self):
            result=super().reset()
            self.current_time=torch.zeros_like(self.load)
            self._time_mask()
            return result
        def _time_mask(self):
            current=self.current_node if self.current_node is not None else torch.zeros_like(self.BATCH_IDX)
            arrival=self.current_time[...,None]+self.duration_matrix[self.BATCH_IDX,current]
            start=torch.maximum(arrival,self.time_windows[:,None,:,0])
            finish=start+self.service_time[:,None,:]+self.duration_matrix[:,None,:,0]
            feasible=(start<=self.time_windows[:,None,:,1]+TIME_EPS)&(finish<=self.time_windows[:,0,1,None,None]+TIME_EPS)
            self.ninf_mask=self.ninf_mask.masked_fill(~feasible,float('-inf'))
            self.ninf_mask[:,:,0]=torch.where((current!=0)|self.finished,0.,float('-inf'))
            if self.selected_count==0:self.ninf_mask[:,:,0]=0. # native initial depot action
            if not torch.isfinite(self.ninf_mask).any(-1).all():raise RuntimeError('VRPTW empty action mask')
            self.step_state.ninf_mask=self.ninf_mask
            self.step_state.current_time=self.current_time
        def step(self,action):
            if self.selected_count and not torch.isfinite(self.ninf_mask[self.BATCH_IDX,self.POMO_IDX,action]).all():
                raise ValueError('Native RADAR selected a time/capacity-infeasible action')
            previous=self.current_node if self.current_node is not None else torch.zeros_like(action)
            arrival=self.current_time+self.duration_matrix[self.BATCH_IDX,previous,action]
            clock=torch.maximum(arrival,self.time_windows[self.BATCH_IDX,action,0])+self.service_time[self.BATCH_IDX,action]
            self.current_time=torch.where(action==0,0.,clock)
            _,reward,done=super().step(action)
            self._time_mask()
            return self.step_state,reward,done
    return RoadVRPTWEnv


def install_radar_time_inputs(model):
    """Task-specific linear inputs before the original encoder and query projection."""
    width=model.embedding_dim
    model.vrptw_node_input=nn.Linear(3,width,bias=False)
    model.vrptw_clock_input=nn.Linear(1,width,bias=False)
    original_pre=model.pre_forward
    def pre(reset):
        model._vrptw_time_features=reset.time_features
        return original_pre(reset)
    model.pre_forward=pre
    def encoder_input(module,args):
        embedding,cost=args
        return embedding+model.vrptw_node_input(model._vrptw_time_features),cost
    model.encoder.register_forward_pre_hook(encoder_input)
    original_forward=model.forward
    def forward(state):
        model._vrptw_current_time=state.current_time
        return original_forward(state)
    model.forward=forward
    def query_input(module,args):
        query,load=args[:2]
        return (query+model.vrptw_clock_input(model._vrptw_current_time[...,None]),load)+args[2:]
    model.decoder.register_forward_pre_hook(query_input)
