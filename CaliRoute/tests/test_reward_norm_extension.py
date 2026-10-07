"""Protect complete-state continuation and committed single-GPU history."""
import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reward_norm_extension import validate_extension_checkpoint, import_single_gpu_history
from run_reward_norm_comparison import build_arm
from offline2online.reward_normalization import signature


def fixture_checkpoint(tmp_path, arm='baseline'):
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/experiments/reward_norm_vrptw100.yaml').read_text())
    cfg = build_arm(base, arm=arm, output=tmp_path / arm, run_name=arm,
        init_checkpoint=tmp_path / 'init.pt', data_root=tmp_path / 'data', seed=3009,
        units=dict(reward_distance_scale_km=43.6, observation_distance_scale_km=43.6))
    rank = dict(rank=0, rng=dict(python=(1,), numpy=(1,), torch=torch.tensor([1]), cuda=torch.tensor([1])),
        sampler=dict(supported=True, **{'class':'AdaptedFixedDatasetInstancePool'}, rng={'state':1},
            attributes=dict(order=[0,1], cursor=1, sample_count=5120)), expert_rng={'state':1},
        policy_route_pool={'routes':[]}, policy_best_objectives={}, scaler={'scale':1024.},
        optimizer_steps=1600, amp_skipped_steps=0, sample_count_offset=0)
    checkpoint = dict(epoch=80, seed=3009, config=copy.deepcopy(cfg),
        model_state_dict={'weight':torch.ones(1)}, optimizer_state_dict={'state':{0:{}},'param_groups':[{'lr':1e-5}]},
        training_resume_state=dict(world_size=1, ranks=[rank], sampler_state_complete=True,
            completed_epoch=80, next_training_epoch=81, evaluation_pending=False))
    if arm in ('normalization','combined'):
        sig = signature(cfg)
        def t(value):
            return torch.tensor(value,dtype=torch.float64)
        shared = dict(beta=t(sig['beta']), sample_count=t(10000), update_count=torch.tensor(80))
        checkpoint['reward_normalization_state'] = dict(signature=sig,
            actor=dict(shared,minimum=t(sig['actor_min_scale']),second_moment=t(2.)),
            critic=dict(shared,minimum=t(sig['critic_min_std']),mean=t(-3.),variance=t(2.)),
            normalized_head=dict(weight=torch.ones(1,4),bias=torch.zeros(1)))
    return checkpoint,cfg


@pytest.mark.parametrize('arm',['baseline','reward','normalization','combined'])
def test_full_checkpoint_accepts_total_300_and_preserves_input(tmp_path,arm):
    checkpoint,cfg=fixture_checkpoint(tmp_path,arm)
    before=copy.deepcopy(cfg)
    assert validate_extension_checkpoint(checkpoint,cfg,300,3009)==80
    assert cfg==before


def test_runtime_solution_aliases_are_allowed(tmp_path):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    checkpoint['config']['advantage']['sl_candidate_incumbent_eta']=cfg['advantage']['sl_candidate_gate_eta']
    assert validate_extension_checkpoint(checkpoint,cfg,300,3009)==80
    checkpoint['config']['advantage']['sl_candidate_incumbent_eta']=999
    with pytest.raises(ValueError,match='configuration'):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('target',[80,60,301,300.5,True])
def test_invalid_new_horizon_rejected(tmp_path,target):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        validate_extension_checkpoint(checkpoint,cfg,target,3009)


@pytest.mark.parametrize('section,key,value',[
    ('training','gamma',1.),('training','ppo_update_epochs',6),('training','n_traj',25),
    ('training','epochs',40),('data','train_sample_mode','random'),('env','reward_distance_scale_km',1.),
    ('model','embedding_dim',128),('evaluation','eval_seed',4),('offline','sl_coef',.1),
    ('advantage','sl_candidate_gap_scale_coef',2.),('critic','advantage_mode','internal'),
    ('pbrs','use_customer_pbrs',True),('training','unknown_algorithm',True),
])
def test_changed_algorithm_rejected(tmp_path,section,key,value):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    checkpoint['config'][section][key]=value
    with pytest.raises(ValueError,match='configuration'):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('key,value',[('lr_schedule','warmup_cosine'),('lr_warmup_epochs',10),('entropy_final_coef',.001)])
def test_horizon_dependent_schedule_rejected(tmp_path,key,value):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    checkpoint['config']['training'][key]=cfg['training'][key]=value
    with pytest.raises(ValueError,match='requires'):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('key,value',[
    ('world_size',2),('sampler_state_complete',False),('evaluation_pending',True),
    ('completed_epoch',79),('next_training_epoch',80),('ranks',[]),
])
def test_incomplete_resume_state_rejected(tmp_path,key,value):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    checkpoint['training_resume_state'][key]=value
    with pytest.raises(ValueError):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('missing',['optimizer_state_dict','model_state_dict'])
def test_weights_only_is_not_resume(tmp_path,missing):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    del checkpoint[missing]
    with pytest.raises(ValueError,match='model and optimizer'):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('component',['rng','sampler','scaler','policy_route_pool','expert_rng'])
def test_all_training_state_is_required(tmp_path,component):
    checkpoint,cfg=fixture_checkpoint(tmp_path)
    rank=checkpoint['training_resume_state']['ranks'][0]
    if component=='rng': rank['rng']['numpy']=None
    elif component=='sampler': rank['sampler']['attributes'].pop('cursor')
    else: rank[component]=None
    with pytest.raises(ValueError):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


@pytest.mark.parametrize('fault',['missing','signature','head','rms','nan','count','minimum'])
def test_normalization_must_resume_without_reset(tmp_path,fault):
    checkpoint,cfg=fixture_checkpoint(tmp_path,'combined')
    norm=checkpoint['reward_normalization_state']
    if fault=='missing': del checkpoint['reward_normalization_state']
    elif fault=='signature': norm['signature']['gamma']=.99
    elif fault=='head': norm['normalized_head'].pop('bias')
    elif fault=='rms': norm['actor'].pop('second_moment')
    elif fault=='nan': norm['critic']['variance'].fill_(float('nan'))
    elif fault=='count': norm['actor']['update_count'].fill_(1)
    elif fault=='minimum': norm['critic']['minimum'].fill_(2.)
    with pytest.raises(ValueError,match='normalization|Normalization'):
        validate_extension_checkpoint(checkpoint,cfg,300,3009)


def fixture_history(tmp_path):
    old={k:str(tmp_path/'old'/k) for k in ('log_dir','checkpoint_dir','output_dir')}
    new={k:str(tmp_path/'new'/k) for k in old}
    for path in old.values(): Path(path).mkdir(parents=True)
    Path(new['output_dir']).mkdir(parents=True)
    logs=Path(old['log_dir']); output=Path(old['output_dir']); checkpoints=Path(old['checkpoint_dir'])
    (logs/'train_log.csv').write_text('epoch,loss\n'+''.join(f'{e},0\n' for e in range(1,42)))
    (logs/'eval_log.csv').write_text('epoch,eval_status,eval_num_instances\n0,ok,1000\n20,ok,1000\n40,ok,1000\n60,ok,1000\n')
    (output/'evaluations').mkdir()
    for e in (0,20,40,60):
        (output/'evaluations'/f'epoch_{e:04d}.jsonl').write_text(json.dumps({'instance_id':1,'epoch':e})+'\n')
    (output/'monitoring').mkdir()
    (output/'monitoring/monitor_rank_0.jsonl').write_text(''.join(json.dumps({'epoch':e,'rank':0,'reward_normalization':{'updates':e}})+'\n' for e in range(1,42)))
    (checkpoints/'best_checkpoint.json').write_text(json.dumps({'epoch':20,'eval_avg_objective_distance_km':250.}))
    (checkpoints/'checkpoint_best.pt').write_bytes(b'preserve exact bytes')
    return old,new


def test_imports_single_rank_complete_history_and_historical_best(tmp_path):
    old,new=fixture_history(tmp_path)
    result=import_single_gpu_history(old,new,40)
    with (Path(new['log_dir'])/'train_log.csv').open() as f:
        assert [int(r['epoch']) for r in csv.DictReader(f)]==list(range(1,41))
    with (Path(new['log_dir'])/'eval_log.csv').open() as f:
        assert [int(r['epoch']) for r in csv.DictReader(f)]==[0,20,40]
    assert len((Path(new['output_dir'])/'monitoring/monitor_rank_0.jsonl').read_text().splitlines())==40
    assert not (Path(new['output_dir'])/'evaluations/epoch_0060.jsonl').exists()
    assert (Path(new['checkpoint_dir'])/'checkpoint_best.pt').read_bytes()==b'preserve exact bytes'
    assert result[str(Path(new['log_dir'])/'train_log.csv')]['rows']==40
    assert result[str(Path(new['checkpoint_dir'])/'checkpoint_best.pt')]['source_sha256']==result[str(Path(new['checkpoint_dir'])/'checkpoint_best.pt')]['copied_sha256']
    assert not (Path(new['log_dir'])/'rank_1').exists()
    with pytest.raises(ValueError,match='already exist'):
        import_single_gpu_history(old,new,40)


@pytest.mark.parametrize('fault',['missing_train','duplicate_train','missing_eval','partial_eval','failed_eval','missing_routes','newer_best','missing_best','duplicate_monitor','existing_routes'])
def test_import_rejects_incomplete_or_overwritten_history_before_copying(tmp_path,fault):
    old,new=fixture_history(tmp_path)
    logs=Path(old['log_dir']); output=Path(old['output_dir']); checkpoints=Path(old['checkpoint_dir'])
    if fault=='missing_train':
        p=logs/'train_log.csv';p.write_text(p.read_text().replace('17,0\n',''))
    elif fault=='duplicate_train':
        p=logs/'train_log.csv';p.write_text(p.read_text().replace('17,0\n','17,0\n17,0\n'))
    elif fault=='missing_eval':
        p=logs/'eval_log.csv';p.write_text(p.read_text().replace('20,ok,1000\n',''))
    elif fault=='partial_eval':
        p=logs/'eval_log.csv';p.write_text(p.read_text().replace('40,ok,1000','40,ok,32'))
    elif fault=='failed_eval':
        p=logs/'eval_log.csv';p.write_text(p.read_text().replace('40,ok,1000','40,failed,1000'))
    elif fault=='missing_routes': (output/'evaluations/epoch_0040.jsonl').unlink()
    elif fault=='newer_best': (checkpoints/'best_checkpoint.json').write_text('{"epoch":60}')
    elif fault=='missing_best': (checkpoints/'checkpoint_best.pt').unlink()
    elif fault=='duplicate_monitor':
        p=output/'monitoring/monitor_rank_0.jsonl';p.write_text(p.read_text()+'{"epoch":1,"rank":0}\n')
    else:
        p=Path(new['output_dir'])/'evaluations/epoch_0000.jsonl';p.parent.mkdir();p.write_text('keep me')
    with pytest.raises(ValueError):
        import_single_gpu_history(old,new,40)
    assert not Path(new['log_dir']).exists()
    assert not Path(new['checkpoint_dir']).exists()
