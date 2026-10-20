import numpy as np
import torch
from e2.rrnco_imitation import load_losses

class Buffer:
    def sample_step_batch_with_objectives(self,batch_size):
        return {},torch.tensor([[0],[1],[0]]),torch.tensor([80.,99.,120.])
    def sample_step_batch(self,batch_size):
        x,y,_=self.sample_step_batch_with_objectives(batch_size);return x,y


def test_awbc_uses_e1_positive_cost_weights_and_global_denominator():
    logp=torch.tensor([-1.,-2.,-3.],requires_grad=True)
    f,meta=load_losses(lambda *args:(logp,torch.zeros(3),torch.tensor([0,1,0])))
    loss,info=f['compute_awbc_loss'](None,Buffer(),3,'cpu',eta=.1,normalize='eta_objective',baseline='mean_successful',baseline_objective=100.)
    weights=torch.tensor([1.,1/9.9,0.])
    torch.testing.assert_close(loss,-(weights*logp).sum()/weights.sum())
    loss.backward();torch.testing.assert_close(logp.grad,-weights/weights.sum())
    assert meta['unchanged_functions']==['compute_awbc_loss','compute_bc_loss']
    assert info['bc_steps']==3


def test_dapg_bc_is_unweighted_step_mean_and_awbc_zero_gate_has_zero_gradient():
    logp=torch.tensor([-1.,-2.,-3.],requires_grad=True)
    f,_=load_losses(lambda *args:(logp,torch.zeros(3),torch.tensor([0,1,0])))
    loss,_=f['compute_bc_loss'](None,Buffer(),3,'cpu');torch.testing.assert_close(loss,torch.tensor(2.))
    zero,_=f['compute_awbc_loss'](None,Buffer(),3,'cpu',normalize='eta_objective',baseline_objective=70.)
    zero.backward();torch.testing.assert_close(logp.grad,torch.zeros(3))
