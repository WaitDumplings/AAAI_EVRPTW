"""The corrected base cannot observe road metrics, including indirect resources."""
from copy import deepcopy
import inspect
import numpy as np
import pytest
import torch
from e1.configs import build_config
from offline2online.models import Agent
from offline2online.model_integration import configure, signature, checkpoint_metadata, load_checkpoint_profile
from test_e1_distance_row import observation


def base(tmp_path):
    cfg=build_config('ppo_base',data_root=tmp_path,output_dir=tmp_path/'run',distance_unit_km=10.)
    cfg['model'].update(embedding_dim=32,n_encode_layers=1)
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    return cfg,agent


def test_actor_critic_cached_and_uncached_ignore_road_and_resources(tmp_path):
    cfg,agent=base(tmp_path);agent.eval();obs=observation();changed=deepcopy(obs)
    for key in ('edge_distance','edge_time','edge_energy','current_time','current_battery',
                'remaining_battery','battery_capacity','time_window','service_time'):
        if key in changed:changed[key]=np.full_like(changed[key],float('inf'))
    changed['node_input_context']=np.ones((4,8));changed['graph_input_context']=np.ones(12)
    changed['future_road_summary']=np.ones(3)*999
    with torch.no_grad():
        plain=agent.backbone(obs);other=agent.backbone(changed)
        assert torch.equal(plain[0],other[0]) and torch.equal(plain[1],other[1])
        assert torch.equal(agent.critic(plain),agent.critic(other))
        cached=agent.backbone.encode(obs)
        a=agent.backbone.decode(obs,cached);b=agent.backbone.decode(changed,cached)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    state=agent.backbone._build_state(changed)
    assert state.states['edge_distance'].count_nonzero()==0
    assert 'future_road_summary' not in state.states
    assert np.isfinite(obs['edge_distance']).all()  # input/env was not modified
    gradobs=deepcopy(obs);gradobs['edge_distance']=torch.tensor(obs['edge_distance'],requires_grad=True)
    agent.backbone(gradobs)[0].sum().backward()
    assert gradobs['edge_distance'].grad is None


def test_profile_preserved_on_resume_and_cannot_silently_enable_road(tmp_path):
    cfg,agent=base(tmp_path);configure(agent,cfg)
    ckpt={'config':cfg,**checkpoint_metadata(agent,cfg)}
    assert load_checkpoint_profile(agent,ckpt,resume=True)['migrated'] is False
    old=deepcopy(cfg);old['model']['e1_no_road_input']=False
    other=Agent(device='cpu',**{k:v for k,v in old['model'].items() if k in inspect.signature(Agent).parameters});configure(other,old)
    with pytest.raises(ValueError,match='no-road input contract'):
        load_checkpoint_profile(other,ckpt,resume=True)
    with pytest.raises(ValueError,match='no-road base'):
        Agent(device='cpu',e1_no_road_input=True,use_encoder_distance_bias=True)


def test_changed_physical_roads_change_reward_but_not_cvrp_policy(tmp_path):
    from test_e1_native import payload
    from offline2online.instance_adapter import adapt_instance_payload
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
    cfg,agent=base(tmp_path);agent.eval();envs=[]
    try:
        for scale in (1.,3.):
            inst=adapt_instance_payload(payload('same',scale,customers=3),problem_type='cvrp',strict_road_metric=True)
            envs.append(make_terran_env(instance=inst,n_traj=2,use_jit_mask=False,
                observation_distance_scale_km=10.,observation_coordinate_mode='depot_fixed',
                normalize_reward=True,reward_distance_scale_km=10.))
        obs=[e.reset()[0] for e in envs];totals=[0.,0.]
        for a in (1,2,3,0):
            np.testing.assert_array_equal(obs[0]['action_mask'],obs[1]['action_mask'])
            with torch.no_grad():
                assert torch.equal(agent.backbone(obs[0])[0],agent.backbone(obs[1])[0])
            for j,e in enumerate(envs):
                obs[j],reward,*_=e.step(np.array([a,a]));totals[j]+=float(reward.sum())
        assert totals[0]!=totals[1]
    finally:
        for e in envs:e.close()
