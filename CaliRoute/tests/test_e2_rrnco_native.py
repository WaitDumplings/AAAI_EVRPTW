from copy import deepcopy
from pathlib import Path
import sys
import torch
from tensordict import TensorDict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/native_sources/rrnco'))
from e2.adapters.rrnco_native import install_bias_processing,install_rrnco_agda


def test_bias_after_clipping_before_mask_and_temperature():
    from rrnco.models import decoding
    install_bias_processing()
    logits=torch.tensor([[3.,-2.,7.]],requires_grad=True)
    bias=torch.tensor([[2.,-1.,8.]],requires_grad=True)
    logits._e2_action_bias=bias
    mask=torch.tensor([[True,True,False]])
    actual=decoding.process_logits(logits,mask,temperature=.7,tanh_clipping=10.)
    expected=torch.log_softmax(((10*torch.tanh(logits)+bias)/.7).masked_fill(~mask,-torch.inf),-1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    (-actual[0,1]).backward()
    assert bias.grad[0,1]!=0 and torch.isneginf(actual[0,2])
    plain=logits.detach().clone()
    torch.testing.assert_close(decoding.process_logits(plain,mask,tanh_clipping=10),
        torch.log_softmax((10*torch.tanh(plain)).masked_fill(~mask,-torch.inf),-1),rtol=0,atol=0)


def test_agda_chunk_checkpoint_preserves_logits_and_gradients():
    from rrnco.models.decoder import RRNetDecoder
    from rrnco.models.decoding import process_logits
    torch.manual_seed(5);torch.set_num_threads(1)
    host=torch.nn.Module();host.decoder=RRNetDecoder(embed_dim=32,num_heads=4,env_name='rcvrptw',use_graph_context=False)
    install_rrnco_agda(host,chunk=3,activation_checkpointing=False)
    # Nonzero learned residuals ensure replay checks more than the zero boundary.
    with torch.no_grad():host.decoder.agda.candidate_action_key_delta_proj.weight.normal_(std=.01)
    other=torch.nn.Module();other.decoder=RRNetDecoder(embed_dim=32,num_heads=4,env_name='rcvrptw',use_graph_context=False)
    install_rrnco_agda(other,chunk=1,activation_checkpointing=True);other.load_state_dict(host.state_dict())
    b,s,n=2,3,7
    z=lambda *shape:torch.zeros(*shape)
    distance=torch.rand(b*s,n,n);duration=torch.rand(b*s,n,n)
    td=TensorDict(dict(current_node=torch.zeros(b*s,dtype=torch.long),
        distance_matrix=distance,duration_matrix=duration,vehicle_capacity=torch.ones(b*s,1),
        used_capacity_linehaul=z(b*s,1),used_capacity_backhaul=z(b*s,1),
        demand_linehaul=torch.rand(b*s,n)*.1,visited=torch.zeros(b*s,n,dtype=torch.bool),
        action_mask=torch.ones(b*s,n,dtype=torch.bool),current_time=z(b*s,1),
        time_windows=torch.stack((z(b*s,n),torch.ones(b*s,n)),-1),service_time=z(b*s,n),
        current_route_length=z(b*s,1),distance_limit=torch.full((b*s,1),torch.inf),
        open_route=torch.zeros(b*s,1,dtype=torch.bool)),batch_size=[b*s])
    # Static instance data must agree across the start-major replicated rows.
    for k in ['distance_matrix','duration_matrix','demand_linehaul']:
        td[k]=td[k][:b].repeat(s,*([1]*(td[k].ndim-1)))
    td['action_mask'][:,4]=False
    nodes=(torch.randn(b,n,32),torch.randn(b,n,32));outputs=[]
    for model in [host,other]:
        local=td.clone();cache=model.decoder._precompute_cache(nodes)
        logits,mask=model.decoder(local,cache,s)
        lp=process_logits(logits,mask,tanh_clipping=10)
        # Environment mutation must not change saved decision state.
        local['current_time'].add_(1);local['used_capacity_linehaul'].add_(.5)
        (-lp[:,1].mean()).backward();outputs.append(lp)
    torch.testing.assert_close(*outputs,rtol=2e-5,atol=2e-6)
    for (name,p),(other_name,q) in zip(host.named_parameters(),other.named_parameters()):
        assert name==other_name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
