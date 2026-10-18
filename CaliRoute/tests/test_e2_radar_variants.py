from copy import deepcopy
from pathlib import Path
import torch
from e2.adapters.radar_rdi import CVRPRoadInjection,RadarRDIAttention
from e2.adapters.radar import RadarAGDADecoder


def test_rdi_preserves_native_sinkhorn_at_zero_and_learns_directed_bias(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/native_sources/radar/acvrp'))
    from ACVRPModel_LIB import MixedScore_MultiHeadAttention
    torch.manual_seed(123)
    kw=dict(head_num=4,qkv_dim=8,sqrt_qkv_dim=8**.5,ms_hidden_dim=16,ms_layer1_init=2**-.5,ms_layer2_init=.25)
    native=MixedScore_MultiHeadAttention(**kw)
    q,k,v=[torch.randn(2,4,11,8) for _ in range(3)]
    road=torch.rand(2,11,11);cost=(road-road.mean())/road.std()
    rdi=CVRPRoadInjection(4);adapter=RadarRDIAttention(native)
    adapter.road_bias=rdi(road)
    expected=native(q,k,v,cost);actual=adapter(q,k,v,cost)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    assert rdi.net[-1].weight.grad.abs().sum()>0
    assert (rdi.features(road)[...,[3,4,5,6,7,9]]==0).all()
    with torch.no_grad():rdi.net[-1].weight.add_(.05*rdi.net[-1].weight.grad)
    adapter.road_bias=rdi(road)
    assert not torch.equal(adapter(q,k,v,cost),expected)


def test_agda_checkpoint_and_trajectory_chunk_preserve_probabilities_and_gradients(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/native_sources/radar/acvrp'))
    from ACVRPModel import ACVRP_Decoder
    torch.manual_seed(12)
    native=ACVRP_Decoder(embedding_dim=32,head_num=4,qkv_dim=8,sqrt_qkv_dim=8**.5,sqrt_embedding_dim=32**.5,logit_clipping=10)
    base=RadarAGDADecoder(native,enabled=True)
    with torch.no_grad():base.agda.candidate_action_key_delta_proj.weight.normal_(std=.01)
    alternative=deepcopy(base);alternative.activation_checkpointing=True;alternative.trajectory_chunk_size=2
    nodes=torch.randn(2,11,32);query=torch.randn(2,5,32);load=torch.ones(2,5);mask=torch.zeros(2,5,11);mask[:,:,4]=-torch.inf
    kw=dict(node_embeddings=nodes,decision_tokens=torch.randn(2,5,4,32),candidate_features=torch.randn(2,5,11,30),system_features=torch.randn(2,5,15))
    outputs=[]
    for model in [base,alternative]:
        step_load=load.clone();step_mask=mask.clone()
        model.set_kv(nodes);out=model(query,step_load,step_mask,**kw)
        if model.activation_checkpointing:
            step_load.zero_();step_mask.fill_(-torch.inf)
        (-out[:,:,1].log().mean()).backward();outputs.append(out)
    torch.testing.assert_close(*outputs,rtol=2e-6,atol=1e-7)
    for (name,p),(other,q) in zip(base.named_parameters(),alternative.named_parameters()):
        assert name==other
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-7)
