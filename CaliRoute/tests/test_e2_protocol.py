from copy import deepcopy
import pytest
from e2.protocol import COMMON, require_matched_protocol, paired_results


def row(i,cost,feasible=True):
    return dict(instance_id=str(i),training_seed=3009,requested_K=50,actual_K=50,
                feasible=feasible,recomputed_cost_km=cost,completed=feasible,truncated=False)


def test_instance_wise_improvement_and_feasibility():
    # Mean percentage improvement is 0%, while ratio of means gives -6.67%.
    a=[row(1,100),row(2,500),row(3,None,False)]
    b=[row(1,90),row(2,550),row(3,100)]
    out=paired_results(a,b,expected_count=3,training_seed=3009)
    assert out['improvement_pct']==0
    assert out['common_count']==2
    assert out['base_feasible_count']==2 and out['variant_feasible_count']==3
    assert out['base_common_mean_km']==300


def test_missing_duplicates_and_candidate_budget_rejected():
    a=[row(1,100),row(2,500)]
    for b in ([row(1,90),row(1,100)], [row(1,90),row(3,100)],
              [row(1,90),dict(row(2,100),actual_K=49)]):
        with pytest.raises(ValueError):paired_results(a,b,expected_count=2,training_seed=3009)


def test_empty_common_subset_is_missing_not_zero():
    out=paired_results([row(1,None,False)],[row(1,100)],expected_count=1,training_seed=3009)
    assert out['common_count']==0 and out['improvement_pct'] is None


def test_native_baseline_reuse_requires_matching_budget_and_objective():
    a=dict.fromkeys(COMMON,'same')
    a.update(initialization='scratch',self_replay=False,incumbent_storage='cost_only',
             native_trainer_sha256='same',modules=[],training_objective='pomo_reinforce')
    b=deepcopy(a);b['modules']=['RDI']
    assert require_matched_protocol(a,b,'A')
    b['training_budget']='extra_epochs'
    with pytest.raises(ValueError,match='training_budget'):require_matched_protocol(a,b,'A')
    b=deepcopy(a);b['modules']=['RDI'];b['training_objective']='step_ppo'
    with pytest.raises(ValueError,match='native training objective'):require_matched_protocol(a,b,'A')


def test_panel_b_cannot_use_native_reinforce_or_architecture_changes():
    a=dict.fromkeys(COMMON,'same')
    a.update(initialization='scratch',self_replay=False,incumbent_storage='cost_only',modules=[],
             actor_architecture_sha256='same',critic_protocol='same',ppo_protocol='same',
             training_objective='step_ppo',expert_data_sha256=None)
    b=deepcopy(a);b.update(training_objective='step_ppo_slppo',expert_data_sha256='train_experts')
    assert require_matched_protocol(a,b,'B')
    a['training_objective']='pomo_reinforce'
    with pytest.raises(ValueError,match='PPO baseline'):require_matched_protocol(a,b,'B')
    a['training_objective']='step_ppo';b['modules']=['AGDA']
    with pytest.raises(ValueError,match='actor'):require_matched_protocol(a,b,'B')
