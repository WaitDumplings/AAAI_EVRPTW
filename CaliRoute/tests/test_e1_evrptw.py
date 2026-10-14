from dataclasses import replace
from pathlib import Path
import csv
import importlib.util
import pickle
from itertools import islice
import numpy as np
import pytest
from e1.vrptw_campaign import configuration, audit
from e1.validator import validate_routes, teacher_force_routes, extract_routes
from e1.evrptw_contract import export_audit_dataset
from e1.lifecycle import _checkpoint_dir
from evrptw_core.schema import EVRPTWInstance


def tiny():
    d=np.ones((4,4),dtype=np.float32);np.fill_diagonal(d,0)
    return EVRPTWInstance.from_dict(dict(instance_id='electric',working_start_s=0,working_end_s=100,
        depot=[0,0],customers=[[1,0],[2,0]],charging_stations=[[1,1]],distance_matrix_km=d,
        demands_cm3=[1,1],package_counts=[1,1],service_time_s=[1,1],tw_s=[[0,100],[0,100]],
        cs_time_to_depot_s=[1],vehicle=dict(cargo_capacity_cm3=3,battery_capacity_kwh=2.1,
        full_charge_time_s=3,consumption_kwh_per_km=1),speed_profile=dict(effective_speed_kmh=3600)))


def test_energy_charge_and_time_checked_independently():
    i=tiny();routes=[[0,1,3,2,0]]
    assert validate_routes(i,routes,4)['valid']
    assert teacher_force_routes(i,routes,dict(prefer_explicit_edge_matrices=True,charging_mode='fixed_full'))['valid']
    assert 'battery_exceeded' in validate_routes(i,[[0,1,2,0]])['reasons']
    assert 'depot_closing_exceeded' in validate_routes(replace(i,working_end_s=8),routes)['reasons']
    e=np.array(i.distance_matrix_km);e[1,3]=2
    assert 'battery_exceeded' in validate_routes(replace(i,energy_matrix_kwh=e),routes)['reasons']
    t=np.array(i.distance_matrix_km);t[1,3]=100
    assert 'depot_closing_exceeded' in validate_routes(replace(i,travel_time_matrix_s=t),routes)['reasons']
    assert 'repeated_station_in_route' in validate_routes(i,[[0,1,3,2,3,0]])['reasons']
    assert not validate_routes(i,[[0,1,4,2,0]])['valid']
    bad=validate_routes(i,routes,5)
    assert bad['feasible'] and not bad['valid'] and bad['distance_matches'] is False


@pytest.mark.parametrize('scale,stations,cap',[(15,3,76),(50,10,244)])
def test_electric_config_and_checkpoint_paths(tmp_path,scale,stations,cap):
    cfg=configuration('slppo',tmp_path,tmp_path/'runs/slppo',40,tmp_path/'backup',scale,1,32,128,problem_type='evrptw')
    assert cfg['data']['problem_type']=='evrptw'
    assert cfg['data']['num_charging_stations']==stations
    assert cfg['training']['rollout_steps']==cfg['evaluation']['eval_max_steps']==cap
    assert cfg['training']['num_envs_per_gpu']==64
    assert cfg['training']['n_traj']==50 and cfg['training']['ppo_update_epochs']==4
    assert cfg['offline']['solution_reference_contract']=='e1_cost_incumbent_v1'
    assert '/evrptw/train/' in cfg['offline']['expert_dataset_path']
    assert f'CS_{stations}' in str(_checkpoint_dir({'code_root':str(tmp_path)},cfg))


@pytest.mark.parametrize('scale',[15,50])
def test_real_electric_expert_audit(tmp_path,scale):
    from offline2online.instance_adapter import iter_instance_payloads
    from e1.assets import _expert_candidates
    source=Path(__file__).parents[2]/'AAAI_Dataset'
    if not (source/f'dataset/evrptw/train/Cus{scale}/instances.pkl').exists():pytest.skip('No real data')
    root=tmp_path/'data'
    for split in ('train','val','test'):
        rel=Path('test_release' if split=='test' else 'dataset')/'evrptw'/split/f'Cus{scale}'
        rows=list(islice(iter_instance_payloads(source/rel),2));target=root/rel;target.mkdir(parents=True)
        with (target/'instances.pkl').open('wb') as h:pickle.dump({'instances':rows},h)
        if split=='train':ids=[r['instance_id'] for r in rows]
        if split=='test':(target/'expert_solutions.csv').write_text('MUST NEVER LOAD TEST SOLUTIONS')
    train=root/f'dataset/evrptw/train/Cus{scale}'
    candidates,_,_=_expert_candidates(source/f'dataset/evrptw/train/Cus{scale}')
    rows=[dict(candidates[i][0]['row']) for i in ids];rows[1]['objective_distance_km']='0'
    with (train/'expert_solutions.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report,inputs=audit(root,tmp_path/'audit',scale,True,expected_counts={'train':2,'val':2,'test':2},problem_type='evrptw')
    assert report['expert_pool']['accepted_count']==1
    assert report['training_data_ready'] and report['test_data_ready']
    assert not any('/test/' in r['path'] and Path(r['path']).name=='expert_solutions.csv' for r in inputs)


def test_ev_reports_do_not_replace_vrptw(tmp_path):
    spec=importlib.util.spec_from_file_location('batch',Path(__file__).parents[1]/'scripts/run_e1_vrptw_batch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old=m.ensure_reports(tmp_path);before=old.read_bytes()
    new=m.ensure_reports(tmp_path,'EVRPTW')
    assert new.name=='EVRPTW_test_overview.csv' and old.read_bytes()==before
    assert {r['problem'] for r in csv.DictReader(new.open())}=={'EVRPTW'}


def test_electric_base_hides_road_inputs_but_keeps_resources(tmp_path):
    from copy import deepcopy
    import inspect
    import torch
    from offline2online.models import Agent
    from offline2online.model_integration import configure, checkpoint_metadata, load_checkpoint_profile
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
    from offline2online.models.no_road_observation import evrptw_no_road_observation
    cfg=configuration('ppo_base',tmp_path,tmp_path/'runs/ppo_base',40,tmp_path/'backup',15,2,32,32,problem_type='evrptw')
    cfg['model'].update(embedding_dim=32,n_encode_layers=1)
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    configure(agent,cfg);agent.eval()
    env=make_terran_env(instance=tiny(),n_traj=2,use_jit_mask=False,prefer_explicit_edge_matrices=True)
    try:
        obs=env.reset()[0];changed=deepcopy(obs)
        for key in ('edge_distance','edge_time','edge_energy'):
            changed[key]=np.full_like(changed[key],99.)
        changed['node_input_context']=np.ones((4,8))*999
        changed['graph_input_context']=np.ones(12)*999
        changed['future_road_summary']=np.array([999.])
        a=evrptw_no_road_observation(obs);b=evrptw_no_road_observation(changed)
        for key in ('time_window','service_time','current_time','current_battery','remaining_battery','battery_capacity','full_charge_time','fixed_full_charge','rs_loc','action_mask'):
            np.testing.assert_array_equal(a[key],b[key])
        assert 'future_road_summary' not in b
        for k in ('edge_distance','edge_time','edge_energy'):assert not np.any(b[k])
        with torch.no_grad():
            x=agent.backbone(obs);y=agent.backbone(changed)
            assert torch.equal(x[0],y[0]) and torch.equal(x[1],y[1])
            assert torch.equal(agent.critic(x),agent.critic(y))
            cache=agent.backbone.encode(obs)
            x=agent.backbone.decode(obs,cache);y=agent.backbone.decode(changed,cache)
            assert torch.equal(x[0],y[0])
        prof={'config':cfg,**checkpoint_metadata(agent,cfg)}
        assert not load_checkpoint_profile(agent,prof,resume=True)['migrated']
        assert cfg['training']['num_envs_per_gpu']==32
        assert cfg['model']['e1_no_road_task']=='evrptw'
    finally:env.close()
