from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import torch
from caliroute.methods import method_preset
from e2.e1_solution_bridge import load_solution_functions


def test_cost_only_snapshot_uses_best_old_cost_and_does_not_mutate_memory():
    funcs,_=load_solution_functions(None)
    pool=NS(reference_objective=lambda key:80.)
    memory={'train_1':70.}
    snapshot=funcs['_E1CostReferenceSnapshot'](pool,memory,['train_1','train_1'])
    memory['train_1']=60.
    assert snapshot.reference_objective('train_1')==70.
    assert not hasattr(snapshot,'actions')


def test_unchanged_e1_group_statistics_and_expert_gate():
    teacher=lambda agent,candidates,device,chunk_size,**kw:torch.zeros(len(candidates))
    f,meta=load_solution_functions(teacher)
    preset=method_preset('slppo')
    cfg=dict(offline=preset.offline_config(),advantage=preset.advantage_config(),
        data=dict(num_customers=100),training=dict(reward_norm_mode='legacy'))
    cfg['offline'].update(solution_reference_contract='e1_cost_incumbent_v1')
    f['_apply_solution_level_aliases'](cfg);f['_validate_e1_solution_contract'](cfg)
    pool=NS(reference_objective=lambda key:80.,trajectory_for_instance=lambda key:NS(
        length=3,actions=[1,2,3],observations=[],objective_distance_km=80.))
    envs=[NS(instance=NS(instance_id='train_1'))]
    batch=NS(actions=torch.ones(1,1,2,dtype=torch.long),old_logprobs=torch.zeros(1,1,2),
        dones=torch.ones(1,1,2,dtype=torch.bool),final_infos=[dict(objective_distance_km=[100.,90.],
        success=[True,True],served_customers=[100,100])])
    group=deepcopy(cfg);group['advantage'].update(use_reference_advantage=False,use_expert_solution_level=False)
    adv,valid,_=f['_solution_level_advantage_tensors'](batch,group,envs,pool,'cpu',policy_best_objectives={'train_1':70.})
    scores=np.array([-100.,-90.,-70.]);expected=(scores[:2]-scores.mean())/scores.std()
    torch.testing.assert_close(adv,torch.tensor(expected,dtype=torch.float32)[None])
    candidates,_=f['_prepare_sl_expert_candidates'](None,batch,cfg,envs,pool,{},'cpu')
    assert len(candidates)==1 and candidates[0].advantage>0
    blocked,_=f['_prepare_sl_expert_candidates'](None,batch,cfg,envs,pool,{'train_1':60.},'cpu')
    assert blocked==[]
    assert '_solution_level_advantage_tensors' in meta['unchanged_functions']
