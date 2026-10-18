import torch
from e2.adapters.rrnco_step import gae, step_ppo_loss


def test_gae_counts_forced_transition_return_and_excludes_padding():
    # First trajectory finishes one step earlier. With lambda=1, each return
    # must equal its remaining physical edge costs / the fixed reward scale.
    old=dict(value=torch.zeros(3,2),reward=torch.tensor([[-2.,-3.],[-4.,-5.],[0.,-7.]]),
        terminal=torch.tensor([[False,False],[True,False],[False,True]]),
        alive=torch.tensor([[True,True],[True,True],[False,True]]),
        policy_valid=torch.tensor([[False,False],[True,True],[False,True]]))
    adv,ret=gae(old,lam=1.,reward_scale=1.)
    torch.testing.assert_close(ret,torch.tensor([[-6.,-15.],[-4.,-12.],[0.,-7.]]))
    old['logp']=torch.zeros(3,2)
    logp=torch.zeros(3,2,requires_grad=True)
    replay=dict(logp=logp,value=torch.zeros(3,2,requires_grad=True),entropy=torch.zeros(3,2))
    step_ppo_loss(replay,old).backward()
    assert (logp.grad[~old['policy_valid']]==0).all()
    assert logp.grad[old['policy_valid']].abs().sum()>0
