"""E1 controlled budget/actor contracts and independent candidate selection."""
import copy
from pathlib import Path
import numpy as np
import pytest
from e1.configs import METHODS,CONTROLLED,SERVER_PLANS,build_config,fit_train_distance_unit
from e1.evaluation import select_candidates,summarize_rows
from offline2online.instance_adapter import adapt_instance_payload


def config(method,tmp_path,**kw):
    return build_config(method,data_root=tmp_path/'AAAI_Dataset',output_dir=tmp_path/method,
        expert_pool=tmp_path/'audited.csv',distance_unit_km=40.,world_size=2 if method in CONTROLLED else 1,
        backup_root=tmp_path/'independent_backups',**kw)


def test_controlled_models_and_budgets_match(tmp_path):
    configs={method:config(method,tmp_path) for method in METHODS}
    for method in CONTROLLED:
        c=configs[method];t=c['training'];e=c['evaluation'];p=c['experiment_protocol']
        assert t['gamma']==.99 and t['ppo_update_epochs']==4 and t['num_minibatches']==4
        assert t['num_envs_per_gpu']==32 and t['n_traj']==50 and t['gradient_accumulation_steps']==1
        assert t['mixed_precision'] is False and t['reward_norm_mode']=='legacy'
        assert e['eval_interval']==50 and e['eval_n_traj']==50 and e['eval_limit'] is None
        assert p['total_instance_exposures']==96000 and p['total_online_trajectory_attempts']==4800000
        assert c['model']['e1_base_distance_row'] and c['model']['use_graph_token']
        assert not c['offline']['policy_replay_enabled'] and not c['offline']['use_priority_sampler']
        assert not any('init_checkpoint'==k for k in c['offline'])
        assert 'test' not in Path(c['data']['train_dataset_path']).parts and 'val' in Path(e['eval_path']).parts
    assert all(configs[m]['model']==configs['ppo_rdi_agda']['model'] for m in ('awbc','dapg','slppo'))
    assert configs['dapg']['training']['epochs']==1503
    assert configs['dapg']['evaluation']['eval_epoch_offset']==3
    assert configs['awbc']['offline']['awbc_coef']==.1
    assert configs['slppo']['offline']['solution_reference_contract']=='e1_cost_incumbent_v1'
    assert configs['slppo']['advantage']['sl_expert_logprob_chunk_size']==configs['slppo']['offline']['sl_expert_logprob_chunk_size']
    for m in ('ppo_base','ppo_rdi_agda'):
        assert 'expert_solution_path' not in configs[m]['offline']


def test_base_removes_plugins_but_shared_row_preserved(tmp_path):
    m=config('ppo_base',tmp_path)['model']
    assert m['e1_base_distance_row'] and m['embedding_dim']==256 and m['n_encode_layers']==2
    for flag in ('use_encoder_distance_bias','use_joint_graph_encoder','use_rdi_v2',
                 'use_physical_input_context','use_dynamic_decision_encoder','use_agda_v2','use_resource_decoder'):
        assert not m[flag]


def test_server_plan_no_sharing_or_duplicate_methods():
    assigned=[]
    for jobs in SERVER_PLANS.values():
        cards=[]
        for method,gpus in jobs:
            assigned.append(method);cards+=gpus
            assert len(gpus)==(2 if method in CONTROLLED else 1)
        assert len(cards)==len(set(cards))
    assert set(assigned)==set(METHODS) and len(assigned)==7
    assert SERVER_PLANS['2080ti_3']==[]


def test_native_not_silently_dual_or_ppo(tmp_path):
    for method in ('rrnco','radar'):
        c=config(method,tmp_path)
        assert 'training' not in c and 'offline' not in c and c['instance_exposures']==96000
        with pytest.raises(ValueError,match='one GPU'):
            build_config(method,data_root=tmp_path,output_dir=tmp_path/method,distance_unit_km=40.,world_size=2)


def test_fit_rejects_validation_or_test_path(tmp_path):
    with pytest.raises(ValueError,match='train split'):fit_train_distance_unit(tmp_path/'cvrp/val/Cus100')


def instance():
    return adapt_instance_payload(dict(instance_id='x',depot=np.zeros(2),customers=np.ones((2,2)),
        distance_matrix_km=np.array([[0,2,8],[7,0,3],[4,6,0]],dtype=np.float32),
        demands_cm3=np.ones(2),vehicle={'cargo_capacity_cm3':2}),problem_type='cvrp',strict_road_metric=True)


def select(candidates):
    return select_candidates(instance(),candidates,method='ppo_base',run_id='run',training_seed=3009,
        inference_seed=17003009,checkpoint_id='hash',requested_k=50)


def test_select_validates_each_candidate_and_includes_return():
    row=select([dict(routes=[[0,1,0]],completed=True,truncated=False,reported_cost=1),
                dict(routes=[[0,1,2,0]],completed=True,truncated=False,reported_cost=9),
                dict(routes=[[0,2,1,0]],completed=True,truncated=False,reported_cost=21)])
    assert row['recomputed_cost_km']==9 and row['selected_candidate']==1 and row['actual_K']==3
    assert row['num_routes']==1 and row['action_count']==3 and row['feasible']


def test_failures_retained_null_no_penalty_or_retry():
    row=select([dict(routes=[[0,1,2]],completed=False,truncated=True,reported_cost=5)])
    assert not row['feasible'] and row['truncated'] and row['recomputed_cost_km'] is None
    assert row['reported_cost'] is None and row['actual_K']==1 and row['selected_route'] is None
    summary=summarize_rows([row]);assert summary['failure_count']==1 and summary['mean_distance_km'] is None


def test_cost_mismatch_remains_auditable_and_no_unrecorded_candidates():
    row=select([dict(routes=[[0,1,2,0]],completed=True,truncated=False,reported_cost=1)])
    assert row['reported_cost']==1 and row['recomputed_cost_km']==9
    assert row['selected_cost_matches_reported'] is False and row['cost_mismatch_candidate_count']==1
    with pytest.raises(ValueError):select([dict(routes=[])]*51)
