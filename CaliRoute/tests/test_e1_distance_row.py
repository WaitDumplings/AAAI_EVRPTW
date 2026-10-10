"""E1 keeps road access in the base without silently enabling RDI/AGDA."""
from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e1.configs import build_config
from offline2online.instance_adapter import adapt_instance_payload
from offline2online.models import Agent
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.decoder import E1BaseDistanceRow
from offline2online.model_integration import signature, configure, checkpoint_metadata, load_checkpoint_profile
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env


def configuration(tmp_path):
    cfg=build_config('ppo_base',data_root=tmp_path/'data',output_dir=tmp_path/'out',distance_unit_km=10.)
    cfg['model'].update(e1_base_distance_row=True,embedding_dim=32,n_encode_layers=1)
    return cfg


def model(cfg):
    return Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})


def observation():
    inst=adapt_instance_payload(dict(instance_id='fixture',problem_class='CVRP',
        depot=np.array([0.,0.]), customers=np.array([[10.,0.],[0.,10.],[20.,20.]]),
        distance_matrix_km=np.array([[0.,1.,5.,2.],[4.,0.,2.,3.],[6.,3.,0.,4.],[2.,5.,6.,0.]],dtype=np.float32),
        demands_cm3=np.array([1.,1.,1.],dtype=np.float32),vehicle={'cargo_capacity_cm3':2.}),
        problem_type='cvrp',strict_road_metric=True)
    env=make_terran_env(instance=inst,n_traj=2,use_fast_env=True,use_jit_mask=False,
        normalize_reward=True,observation_distance_scale_km=10.,observation_coordinate_mode='depot_fixed')
    obs,_=env.reset();env.close()
    return obs


def test_row_features_match_original_slots_clamps_direction_and_units():
    edge=torch.tensor([[[0.,.1,.5],[.4,0.,.2],[.6,.3,0.]]])
    state=SimpleNamespace(states={'edge_distance':edge},get_current_node=lambda:torch.tensor([[0,1]]))
    features=E1BaseDistanceRow.features(state,torch.zeros(1,3,16),2)
    expected=torch.tensor([[[[0.,0.,0.],[.1,.4,.5],[.5,.6,1.1]],
                            [[.4,0.,0.],[0.,.4,0.],[.2,.6,.4]]]])
    torch.testing.assert_close(features[...,12:15],expected)
    assert features[...,:12].count_nonzero()==features[...,15:].count_nonzero()==0
    edge[0,1,2]=9.;features=E1BaseDistanceRow.features(state,torch.zeros(1,3,16),2)
    assert features[0,1,2,12]==2. and features[0,1,2,14]==2.


def test_row_has_no_euclidean_or_resource_fallback_and_handles_masked_infinity():
    edge=torch.tensor([[[0.,float('inf')],[2.,0.]]])
    state=SimpleNamespace(states={'edge_distance':edge},get_current_node=lambda:torch.tensor([[0]]))
    row=E1BaseDistanceRow(16)
    key,bias=row(state,torch.zeros(1,2,16),1)
    assert torch.isfinite(key).all() and torch.isfinite(bias).all()
    state.states={}
    with pytest.raises(KeyError):row(state,torch.zeros(1,2,16),1)


def test_zero_head_initialization_does_not_shift_shared_weights_or_rng(tmp_path):
    cfg=configuration(tmp_path);old=deepcopy(cfg);old['model']['e1_base_distance_row']=False
    torch.manual_seed(7);prior=model(old);after_prior=torch.rand(5)
    torch.manual_seed(7);updated=model(cfg);after_updated=torch.rand(5)
    assert torch.equal(after_prior,after_updated)
    for name,value in prior.state_dict().items():assert torch.equal(value,updated.state_dict()[name]),name
    obs=observation()
    with torch.no_grad():before=prior.backbone(obs)[0];after=updated.backbone(obs)[0]
    torch.testing.assert_close(before,after,atol=2e-6,rtol=1e-5)


def test_disabled_default_exactly_matches_explicit_false(tmp_path):
    cfg=configuration(tmp_path);cfg['model'].pop('e1_base_distance_row')
    torch.manual_seed(8);default=model(cfg)
    cfg['model']['e1_base_distance_row']=False
    torch.manual_seed(8);explicit=model(cfg)
    assert default.backbone.decoder.e1_distance_row is None
    assert default.state_dict().keys()==explicit.state_dict().keys()
    for name,value in default.state_dict().items():assert torch.equal(value,explicit.state_dict()[name])
    with torch.no_grad():assert torch.equal(default.backbone(observation())[0],explicit.backbone(observation())[0])


def test_base_learns_distance_sensitivity_without_plugin_gradients(tmp_path):
    cfg=configuration(tmp_path);torch.manual_seed(9);agent=model(cfg);obs=observation()
    optimizer=torch.optim.SGD(agent.parameters(),lr=.2)
    logits=agent.backbone(obs)[0]
    loss=-torch.distributions.Categorical(logits=logits).log_prob(torch.ones((1,2),dtype=torch.long)).mean()
    loss.backward()
    row=agent.backbone.decoder.e1_distance_row
    assert row.action_bias_proj[-1].weight.grad.abs().sum()>0
    assert row.candidate_action_key_delta_proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in agent.backbone.decoder.dynamic_graph_kv_encoder.parameters())
    assert agent.backbone.rdi_adapter is None and agent.backbone.joint_graph_encoder is None
    assert agent.backbone.edge_relation_encoder is None and agent.backbone.physical_input_adapter is None
    assert agent.backbone.decoder.resource_decoder is None
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    changed=deepcopy(obs);changed['edge_distance']=changed['edge_distance'].copy();changed['edge_distance'][0,1]+=.2
    with torch.no_grad():baseline=agent.backbone(obs)[0];altered=agent.backbone(changed)[0]
    assert not torch.allclose(baseline[...,1:],altered[...,1:])
    resources=deepcopy(obs);resources['edge_time']=resources['edge_time']*100.
    with torch.no_grad():assert torch.equal(baseline,agent.backbone(resources)[0])
    # The historical non-graph encoder has an energy reachability mask. Keep
    # that unchanged; only this extracted branch promises resource independence.
    resources['edge_energy']=resources['edge_energy']+100.
    original_state=StateWrapper(obs,'cpu');resource_state=StateWrapper(resources,'cpu')
    like=torch.zeros(1,4,32)
    torch.testing.assert_close(row.features(original_state,like,2),row.features(resource_state,like,2))
    gradobs=deepcopy(obs);gradobs['edge_distance']=torch.tensor(gradobs['edge_distance'],requires_grad=True)
    logits=agent.backbone(gradobs)[0];(-torch.log_softmax(logits,-1)[...,1].mean()).backward()
    assert torch.isfinite(gradobs['edge_distance'].grad).all()
    assert gradobs['edge_distance'].grad.abs().sum()>0


def test_e1_profile_is_explicit_and_legacy_signature_unchanged(tmp_path):
    cfg=configuration(tmp_path);profile=signature(cfg)
    assert profile['e1_base_distance_row'] is True
    agent=model(cfg);configure(agent,cfg)
    checkpoint={'config':cfg,**checkpoint_metadata(agent,cfg)}
    assert not load_checkpoint_profile(agent,checkpoint,resume=True)['migrated']
    old=deepcopy(cfg);old['model']['e1_base_distance_row']=False
    old_profile=signature(old);assert 'e1_base_distance_row' not in old_profile
    del old['model']['e1_base_distance_row'];assert signature(old)==old_profile
    older=model(old);configure(older,old)
    with pytest.raises(ValueError,match='distance-row profile changed'):
        load_checkpoint_profile(older,checkpoint,resume=True)
    cfg['model']['e1_base_distance_row']='yes'
    with pytest.raises(ValueError,match='boolean'):signature(cfg)


def test_all_controlled_methods_share_base_distance_path(tmp_path):
    for method in ('ppo_base','ppo_rdi_agda','awbc','dapg','slppo'):
        cfg=build_config(method,data_root=tmp_path/'data',output_dir=tmp_path/method,
                         expert_pool=tmp_path/'pool.csv',distance_unit_km=10.)
        assert cfg['model']['e1_base_distance_row'] is True
