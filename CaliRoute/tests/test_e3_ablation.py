from copy import deepcopy
import inspect
import torch
import pytest
from e3.configs import build
from offline2online.models import Agent
from offline2online.model_integration import configure, checkpoint_metadata, load_checkpoint_profile
from test_e1_distance_row import observation


def make(tmp_path, variant):
    cfg=build(variant,data_root=tmp_path,output_dir=tmp_path/variant,distance_unit_km=10.)
    cfg['model'].update(embedding_dim=32,n_encode_layers=1)
    torch.manual_seed(3009)
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    configure(agent,cfg)
    return cfg,agent


def test_a_only_changes_encoder_metric_and_preserves_road_decoder(tmp_path):
    pairs=[make(tmp_path,v) for v in ('A0_no_bias','AE_euclidean_bias','AR_road_bias')]
    obs=observation()
    for _,a in pairs:
        assert a.backbone.decoder.e1_distance_row is not None
        for key,value in pairs[0][1].state_dict().items():assert torch.equal(value,a.state_dict()[key]),key
    states=[a.backbone._build_state(obs) for _,a in pairs]
    biases=[a.backbone._build_attn_bias(s) for (_,a),s in zip(pairs,states)]
    assert torch.count_nonzero(biases[0])==0
    torch.testing.assert_close(biases[1],-pairs[1][1].backbone._build_distance_matrix(states[1].observations))
    torch.testing.assert_close(biases[2],-states[2].states['edge_distance'])
    assert not torch.allclose(biases[1],biases[2])
    for s in states:torch.testing.assert_close(s.states['edge_distance'],states[0].states['edge_distance'])
    features=[a.backbone.decoder.e1_distance_row.features(s,torch.zeros(1,4,32),2) for (_,a),s in zip(pairs,states)]
    for f in features:torch.testing.assert_close(f,features[0])
    # Rigid transforms preserve the geometry, road input is left unchanged.
    rotated=deepcopy(obs)
    for k in ('depot_loc','cus_loc','rs_loc'):
        rotated[k]=obs[k][...,::-1].copy()+13
    _,ae=pairs[1];state=ae.backbone._build_state(rotated)
    torch.testing.assert_close(ae.backbone._build_attn_bias(state),biases[1],atol=2e-6,rtol=1e-5)


def test_b_features_gradients_mask_and_checkpoint(tmp_path):
    b0=make(tmp_path,'B0_static_candidates')[1]
    cfg,b1=make(tmp_path,'B1_feature_matched_logit');_,b2=make(tmp_path,'B2_agda')
    core1=b1.backbone.decoder.dynamic_graph_kv_encoder
    assert b0.backbone.decoder.e1_distance_row is None
    for key,value in b0.state_dict().items():assert torch.equal(value,b1.state_dict()[key]),key
    obs=observation();out0=b0.backbone(obs)[0];out1=b1.backbone(obs)[0];out2=b2.backbone(obs)[0]
    torch.testing.assert_close(out0,out1);torch.testing.assert_close(out0,out2)
    captured={}
    def hook(name):
        def inner(module,args):captured[name]=args[0].detach().clone()
        return inner
    h1=core1.e3_logit_head.register_forward_pre_hook(hook('mlp'))
    core2=b2.backbone.decoder.dynamic_graph_kv_encoder
    h2=core2.candidate_delta_base.register_forward_pre_hook(hook('features'))
    b1.backbone(obs);b2.backbone(obs);h1.remove();h2.remove()
    torch.testing.assert_close(captured['mlp'][...,64:94],captured['features'])
    loss=-torch.distributions.Categorical(logits=out1).log_prob(torch.ones((1,2),dtype=torch.long)).mean();loss.backward()
    assert core1.e3_logit_head[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in b1.parameters() if p.grad is not None)
    optim=torch.optim.SGD(b1.parameters(),lr=.1);optim.step()
    updated=b1.backbone(obs)[0];assert not torch.allclose(out1[...,1:],updated[...,1:])
    assert torch.equal(torch.isneginf(out1),torch.isneginf(updated))
    ckpt={'config':cfg,**checkpoint_metadata(b1,cfg)}
    _,restored=make(tmp_path,'B1_feature_matched_logit');load_checkpoint_profile(restored,ckpt,resume=True)
    restored.load_state_dict(b1.state_dict());torch.testing.assert_close(restored.backbone(obs)[0],updated)
    with pytest.raises(ValueError,match='E3 architecture'):
        load_checkpoint_profile(b2,ckpt,resume=True)


def test_full_size_mlp_parameter_match(tmp_path):
    from e3.model import install
    cfg=build('B1_feature_matched_logit',data_root=tmp_path,output_dir=tmp_path/'run',distance_unit_km=10.)
    a=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters});install(a,cfg)
    m=a.backbone.decoder.dynamic_graph_kv_encoder.e3_parameter_match
    assert abs(m['b1_candidate_parameters']/m['b2_active_candidate_parameters']-1)<.01
