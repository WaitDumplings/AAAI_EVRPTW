"""Scale budgets, first-action validity and cross-scale queue isolation."""
import json
from types import SimpleNamespace
import pytest
import torch
from e1.configs import METHODS, CONTROLLED, build_config, server_plans, schema_for_customers
from e1.runner import dependency_state
from e1.native_worker import install_radar_cyclic_starts

@pytest.mark.parametrize('n',[15,50,100])
def test_scale_paths_and_decode_budget(n,tmp_path):
    for method in METHODS:
        cfg=build_config(method,data_root=tmp_path,output_dir=tmp_path/method,expert_pool=tmp_path/'expert.csv',
                         customers=n,distance_unit_km=40.,world_size=1)
        p=cfg['experiment_protocol']
        assert p['schema']==schema_for_customers(n) and p['customer_count']==n
        assert p['decode_cap']==2*n+1 and p['requested_K']==50 and p['total_instance_exposures']==96000
        if method in CONTROLLED:
            assert cfg['training']['num_envs_per_gpu']==64 and cfg['training']['n_traj']==50
            assert cfg['training']['rollout_steps']==cfg['evaluation']['eval_max_steps']==2*n+1
            assert cfg['data']['train_dataset_path'].endswith(f'cvrp/train/Cus{n}')
            assert cfg['training']['ppo_update_epochs']==4 and cfg['training']['learning_rate']==1e-4
            assert cfg['evaluation']['eval_path'].endswith(f'cvrp/val/Cus{n}')
        else:
            assert cfg['num_customers']==n and cfg['max_steps']==2*n+1
            assert cfg['allow_repeated_starts']==(n<50)


def test_batch2_gpu_assignment_does_not_overlap_except_queued_a6000():
    all_plans={n:server_plans(n) for n in (15,50)}
    for n,plan in all_plans.items():
        assert sorted(m for jobs in plan.values() for m,g in jobs)==sorted(METHODS)
        assert all(len(g)==1 for jobs in plan.values() for m,g in jobs)
    for host,capacity in [('2080ti_a',4),('2080ti_b',4),('2080ti_3',3)]:
        gpus=[g[0] for plan in all_plans.values() for m,g in plan[host]]
        assert len(gpus)==len(set(gpus)) and max(gpus)<capacity
    assert not all_plans[15]['2080ti_c'] and not all_plans[50]['2080ti_c']


def test_cyclic_radar_forced_actions_and_probabilities():
    class Model:
        encoded_node=torch.arange(2*16*3).reshape(2,16,3).float()
        def __init__(self):self.decoder=SimpleNamespace(set_q1=lambda x:setattr(self,'first',x))
        def forward(self,state):return 'native'
    model=Model();install_radar_cyclic_starts(model,15)
    state=SimpleNamespace(BATCH_IDX=torch.zeros(2,50),selected_count=1)
    selected,prob=model.forward(state)
    assert selected.shape==prob.shape==(2,50)
    assert selected[0].tolist()==[i%15+1 for i in range(50)]
    assert torch.all(prob==1) and model.first.shape==(2,50,3)
    assert torch.equal(model.first,model.encoded_node.gather(1,selected[...,None].expand(-1,-1,3)))
    state.selected_count=2;assert model.forward(state)=='native'
    state.selected_count=1;state.BATCH_IDX=torch.zeros(2,15);assert model.forward(state)=='native'


def test_dependency_blocks_until_all_predecessors_succeed(tmp_path):
    here=tmp_path/'new';old=tmp_path/'old';here.mkdir();old.mkdir()
    assert dependency_state(here)=='ready'
    (here/'dependency.json').write_text(json.dumps(dict(scheduler=str(old),methods=['rrnco','radar'])))
    assert dependency_state(here)=='waiting'
    def write(a,b):
        (old/'status.json').write_text(json.dumps(dict(methods={'rrnco':{'state':a},'radar':{'state':b}})))
    write('completed','training');assert dependency_state(here)=='waiting'
    write('completed','failed');assert dependency_state(here)=='failed'
    write('completed','completed');assert dependency_state(here)=='ready'
