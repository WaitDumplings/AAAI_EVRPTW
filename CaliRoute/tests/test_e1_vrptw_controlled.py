from copy import deepcopy
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from e1.vrptw_controlled import configuration,predecessor_state
from offline2online.models import Agent
from offline2online.model_integration import configure
from test_e1_distance_row import observation


def test_vrptw_no_explicit_roads_preserves_time_inputs(tmp_path):
    cfg=configuration('ppo_base',tmp_path,tmp_path/'runs/base',10.,tmp_path/'backup')
    cfg['model'].update(embedding_dim=32,n_encode_layers=1)
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    configure(agent,cfg);agent.eval();obs=observation();changed=deepcopy(obs)
    for key in ('edge_distance','edge_time','edge_energy','current_battery'):
        if key in changed:changed[key]=np.full_like(changed[key],1234.)
    with torch.no_grad():
        assert torch.equal(agent.backbone(obs)[0],agent.backbone(changed)[0])
        encoded=agent.backbone.encode(obs)
        assert torch.equal(agent.backbone.decode(obs,encoded)[0],agent.backbone.decode(changed,encoded)[0])
        timed=deepcopy(obs);timed['current_time']=np.ones_like(timed['current_time'])*.4
        timed['time_window']=timed['time_window'].copy();timed['time_window'][...,1]=.7
        assert not torch.equal(agent.backbone(obs)[0],agent.backbone(timed)[0])
    state=agent.backbone._build_state(timed)
    assert state.states['edge_time'].count_nonzero()==0
    assert state.states['current_time'].max()>0
    assert state.states['time_window'].max()<1.


def test_protocol_matches_and_predecessor_requires_actual_completion(tmp_path):
    base=configuration('ppo_base',tmp_path,tmp_path/'runs/base',10.,tmp_path/'backup')
    graph=configuration('ppo_rdi_agda',tmp_path,tmp_path/'runs/graph',10.,tmp_path/'backup')
    for key in ('epochs','num_envs_per_gpu','n_traj','ppo_update_epochs','num_minibatches','learning_rate','gamma'):
        assert base['training'][key]==graph['training'][key]
    assert base['data']['problem_type']=='vrptw'
    assert graph['model']['use_joint_graph_encoder'] and graph['model']['use_dynamic_decision_encoder']
    assert base['model']['e1_no_road_task']=='vrptw' and not base['model']['e1_base_distance_row']
    assert predecessor_state(tmp_path)=='waiting'
    status=tmp_path/'scheduler_2080ti_a/status.json';status.parent.mkdir()
    status.write_text(json.dumps({'methods':{'ppo_base':{'state':'stopped_by_user'}}}))
    assert predecessor_state(tmp_path)=='blocked'
    status.write_text(json.dumps({'methods':{'ppo_base':{'state':'completed'}}}))
    assert predecessor_state(tmp_path)=='waiting'
    run=tmp_path/'runs/ppo_base';run.mkdir(parents=True)
    (run/'training_result.json').write_text(json.dumps({'state':'completed','last_checkpoint':str(tmp_path/'missing.pt')}))
    assert predecessor_state(tmp_path)=='blocked'
