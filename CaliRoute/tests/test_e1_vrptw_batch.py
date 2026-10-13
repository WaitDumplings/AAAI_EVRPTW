from pathlib import Path
import importlib.util
import pytest
from e1.vrptw_campaign import configuration
from e1.configs import SUPPORTED_SCHEMAS, content_hash

@pytest.mark.parametrize('scale',[15,50,100])
@pytest.mark.parametrize('method',['ppo_base','ppo_rdi_agda','awbc','dapg','slppo'])
def test_vrptw_paths_constraints_and_equal_online_budget(tmp_path,scale,method):
    data=tmp_path/'data';run=tmp_path/'campaign/runs'/method
    cfg=configuration(method,data,run,40.,tmp_path/'backup',scale,1,8,32)
    assert cfg['experiment_protocol']['schema'] in SUPPORTED_SCHEMAS
    assert cfg['data']['problem_type']=='vrptw'
    assert str(data/f'dataset/vrptw/train/Cus{scale}')==cfg['data']['train_dataset_path']
    assert '/vrptw/val/' in cfg['evaluation']['eval_path']
    assert cfg['experiment_protocol']['total_instance_exposures']==96000
    assert cfg['experiment_protocol']['total_online_loops']==1500
    assert cfg['training']['ppo_update_epochs']==4
    assert cfg['training']['num_envs_per_gpu']==64
    assert cfg['training']['n_traj']==50 and cfg['evaluation']['eval_n_traj']==50
    assert cfg['experiment_protocol']['actor_sha256']==content_hash(cfg['model'])
    if method in ('awbc','dapg','slppo'):
        assert cfg['offline']['expert_dataset_path']==cfg['data']['train_dataset_path']
        assert cfg['offline']['strict_replay']
    if method=='ppo_base':
        assert cfg['model']['e1_no_road_input'] and cfg['model']['e1_no_road_task']=='vrptw'
        assert not cfg['model']['e1_base_distance_row']
    else:assert cfg['model']['use_joint_graph_encoder']


def test_two_cards_preserve_global_batch(tmp_path):
    cfg=configuration('slppo',tmp_path,tmp_path/'campaign/runs/slppo',40.,tmp_path/'backup',100,2,8,32)
    assert cfg['training']['num_envs_per_gpu']==32
    assert cfg['experiment_protocol']['global_batch']==64
    with pytest.raises(ValueError):configuration('slppo',tmp_path,tmp_path/'campaign/runs/slppo',40.,tmp_path/'backup',100,3,8,32)


def test_profiles_fit_card_counts_and_do_not_overlap():
    spec=importlib.util.spec_from_file_location('vrptw_batch',Path(__file__).parents[1]/'scripts/run_e1_vrptw_batch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for profile,budget in [('2080ti_a',2),('2080ti_b',4),('a6000',2),('2080ti_3',3)]:
        for _,placements in m.PROFILES[profile]:
            cards=[g for gs in placements.values() for g in gs]
            assert len(cards)==len(set(cards)) and min(cards)>=0 and max(cards)<budget
        if profile!='2080ti_a':
            cards=[g for _,placements in m.PROFILES[profile] for gs in placements.values() for g in gs]
            assert len(cards)==len(set(cards))
    assert [n for n,_ in m.PROFILES['2080ti_a']]==[15,50]
    assert m.PROFILES['2080ti_3']==[(100,{'slppo':[0,1]})]


def test_real_vrptw_expert_audit_rejects_wrong_cost_without_reading_test_solutions(tmp_path):
    import csv,json,pickle
    from itertools import islice
    from e1.vrptw_campaign import audit
    from e1.assets import _expert_candidates
    from offline2online.instance_adapter import iter_instance_payloads
    source=Path(__file__).parents[2]/'AAAI_Dataset'
    if not (source/'dataset/vrptw/train/Cus15/instances.pkl').exists():pytest.skip('Local real dataset unavailable')
    root=tmp_path/'data';train=[]
    for split in ('train','val','test'):
        relative=Path('test_release' if split=='test' else 'dataset')/'vrptw'/split/'Cus15'
        rows=list(islice(iter_instance_payloads(source/relative),2))
        target=root/relative;target.mkdir(parents=True)
        with (target/'instances.pkl').open('wb') as h:pickle.dump({'instances':rows},h)
        if split=='train':train=rows
        if split=='test':(target/'expert_solutions.csv').write_text('THIS MUST NOT BE PARSED')
    candidates,_,_=_expert_candidates(source/'dataset/vrptw/train/Cus15')
    experts=[dict(candidates[r['instance_id']][0]['row']) for r in train]
    experts[1]['objective_distance_km']='0.0'
    with (root/'dataset/vrptw/train/Cus15/expert_solutions.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(experts[0]));w.writeheader();w.writerows(experts)
    report,inputs=audit(root,tmp_path/'audit',15,True,expected_counts={'train':2,'val':2,'test':2})
    assert report['training_data_ready'] and report['test_data_ready']
    assert report['expert_pool']['accepted_count']==1
    assert report['expert_pool']['teacher_forcing_required_for_acceptance']
    assert not any('test/Cus15/expert_solutions' in r['path'] for r in inputs)


def test_local_cus50_waits_for_training_and_tests_but_not_cus100(tmp_path):
    import json
    spec=importlib.util.spec_from_file_location('vrptw_queue',Path(__file__).parents[1]/'scripts/run_e1_vrptw_batch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.predecessor_state(tmp_path)=='waiting_for_cus15'
    scheduler=tmp_path/'scheduler_2080ti_a/status.json';scheduler.parent.mkdir()
    tests=tmp_path/'test_watcher/status.json';tests.parent.mkdir()
    def state(a,b,ta,tb):
        scheduler.write_text(json.dumps({'methods':{'ppo_base':{'state':a},'ppo_rdi_agda':{'state':b}}}))
        tests.write_text(json.dumps({'jobs':{'vrptw15_ppo_base':{'state':ta},'vrptw15_ppo_rdi_agda':{'state':tb}}}))
        return m.predecessor_state(tmp_path)
    assert state('completed','training','completed','waiting_for_training')=='waiting_for_cus15'
    assert state('completed','completed','completed','testing')=='waiting_for_cus15'
    assert state('completed','completed','completed','blocked')=='blocked_by_cus15'
    assert state('completed','completed','completed','completed')=='ready'
    # No Cus100 folder/status is needed or consulted.
