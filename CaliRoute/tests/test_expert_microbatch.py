"""Full sampled batch loss/gradient equivalence, including AWBC global quantiles."""
import copy
import inspect
import numpy as np
import pytest
import torch
from offline2online.offline_data import compute_bc_loss, compute_awbc_loss, _load_terran_runtime
from ablation.dapg import compute_dapg_demo_loss
from e1.vrptw_campaign import configuration
from offline2online.models import Agent
from offline2online.model_integration import configure
from test_e1_evrptw import tiny


@pytest.mark.parametrize('method', ['bc', 'dapg', 'awbc'])
def test_actual_graph_expert_microbatch_gradient_equivalence(method,tmp_path):
    torch.manual_seed(17)
    cfg=configuration('awbc',tmp_path,tmp_path/'runs/awbc',40,tmp_path/'backup',50,1,8,32,problem_type='evrptw')
    cfg['model']['embedding_dim']=32
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    configure(agent,cfg)
    other=copy.deepcopy(agent)
    make_env,stack=_load_terran_runtime()
    env=make_env(instance=tiny(),n_traj=1,use_jit_mask=False,prefer_explicit_edge_matrices=True,observation_input_context=True,observation_distance_scale_km=40,observation_coordinate_mode='depot_fixed')
    obs=env.reset()[0];env.close()
    observations=[]
    for i in range(7):
        x=copy.deepcopy(obs)
        x['edge_distance']=x['edge_distance']*(1+i*.03)
        observations.append(x)
    class Buffer:
        calls=0
        def sample_step_batch(self,batch_size):
            self.calls+=1
            return stack(observations),torch.ones((7,1),dtype=torch.long)
        def sample_step_batch_with_objectives(self,batch_size):
            x,a=self.sample_step_batch(batch_size)
            return x,a,torch.tensor([1.,2.,3.,5.,8.,13.,21.])
    f={'bc':compute_bc_loss,'dapg':compute_dapg_demo_loss,'awbc':compute_awbc_loss}[method]
    b1,b2=Buffer(),Buffer()
    loss,info=f(agent,b1,7,'cpu');loss.backward()
    small,details=f(other,b2,7,'cpu',microbatch_size=3);small.backward()
    assert b1.calls==b2.calls==1
    torch.testing.assert_close(loss,small,rtol=1e-5,atol=1e-6)
    for k,v in info.items():assert details[k]==pytest.approx(v,rel=1e-5,abs=1e-6),k
    for (name,p),(_,q) in zip(agent.named_parameters(),other.named_parameters()):
        if p.grad is None:assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=2e-6,msg=name)
