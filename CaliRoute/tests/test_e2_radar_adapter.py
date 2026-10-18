import importlib.util
import sys
from pathlib import Path
from copy import deepcopy
import torch
import pytest
from e2.adapters.radar import RadarAGDADecoder


def native_decoder():
    directory=Path(__file__).resolve().parents[1]/'benchmarks/native_sources/radar/acvrp'
    spec=importlib.util.spec_from_file_location('ACVRPModel_LIB',directory/'ACVRPModel_LIB.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    previous=sys.modules.get('ACVRPModel_LIB');sys.modules['ACVRPModel_LIB']=lib
    try:
        spec=importlib.util.spec_from_file_location('e2_test_radar_native',directory/'ACVRPModel.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        if previous is None:sys.modules.pop('ACVRPModel_LIB',None)
        else:sys.modules['ACVRPModel_LIB']=previous
    return module.ACVRP_Decoder(embedding_dim=32,head_num=4,qkv_dim=8,
            sqrt_qkv_dim=8**.5,sqrt_embedding_dim=32**.5,logit_clipping=10)


def test_disabled_and_zero_initialized_adapter_preserve_native_probabilities_and_masks():
    torch.manual_seed(12);native=native_decoder();nodes=torch.randn(2,101,32)
    native.set_kv(nodes);query=nodes[:,:3];load=torch.rand(2,3);mask=torch.zeros(2,3,101);mask[:,:,7]=-torch.inf
    expected=native(query,load,mask)
    adapter=RadarAGDADecoder(native,enabled=False)
    torch.testing.assert_close(adapter(query,load,mask),expected,rtol=0,atol=0)
    adapter.enabled=True
    with pytest.raises(ValueError):adapter(query,load,mask)
    kw=dict(node_embeddings=nodes,decision_tokens=torch.randn(2,3,3,32),
            candidate_features=torch.randn(2,3,101,30),system_features=torch.randn(2,3,15))
    actual=adapter(query,load,mask,**kw)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    (-actual[:,:,1].log().mean()).backward()
    assert adapter.agda.candidate_action_key_delta_proj.weight.grad.abs().sum()>0
    assert adapter.agda.action_bias_proj[-1].weight.grad.abs().sum()>0
    with torch.no_grad():adapter.agda.candidate_action_key_delta_proj.weight.normal_(std=.1)
    updated=adapter(query,load,mask,**kw)
    assert not torch.allclose(updated,expected)
    assert (updated[:,:,7]==0).all() and torch.isfinite(updated).all()
    torch.testing.assert_close(updated.sum(-1),torch.ones(2,3))


def test_action_bias_is_added_after_native_clipping():
    torch.manual_seed(7)
    native=native_decoder();nodes=torch.randn(2,11,32);native.set_kv(nodes)
    query=nodes[:,:3];load=torch.ones(2,3);mask=torch.zeros(2,3,11);mask[:,:,4]=-torch.inf
    original=native(query,load,mask)
    bias=torch.linspace(-2,2,11).expand(2,3,11)
    class BiasOnly(torch.nn.Module):
        def forward(self,*args,**kwargs):
            return 0,0,torch.zeros(2,3,11,32),bias
    adapter=RadarAGDADecoder(native,enabled=True);adapter.agda=BiasOnly()
    actual=adapter(query,load,mask,node_embeddings=nodes,
        decision_tokens=torch.zeros(2,3,1,32),candidate_features=torch.zeros(2,3,11,30),
        system_features=torch.zeros(2,3,15))
    expected=torch.softmax(original.log()+bias,dim=-1)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
