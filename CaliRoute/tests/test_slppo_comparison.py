from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest

from caliroute.optimization import apply_optimization_profile

script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_slppo_comparison.py'
spec = importlib.util.spec_from_file_location('comparison_launcher', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_comparison_uses_same_initialization_data_budget_and_evaluation(tmp_path):
    args = Namespace(problem='cvrp', customers=50, data_root=tmp_path / 'dataset',
                     init_checkpoint=tmp_path / 'ppo.pt', epochs=100, num_envs=64, n_traj=50,
                     learning_rate=5e-5, eval_interval=20, eval_batch_size=128, seed=3009,
                     eval_limit=None, expert_limit=None)
    configs = comparison.build_configs(args, tmp_path / 'experiment')
    left, right = configs['baseline'], configs['optimized']
    for section in ('data', 'env', 'critic', 'pbrs', 'advantage'):
        assert left[section] == right[section]
    assert left['offline']['init_checkpoint_path'] == right['offline']['init_checkpoint_path']
    for key in ('epochs', 'learning_rate', 'post_init_seed', 'n_traj', 'rollout_steps', 'ppo_update_epochs'):
        assert left['training'][key] == right['training'][key]
    for key in ('eval_seed', 'eval_path', 'eval_n_traj', 'eval_max_steps', 'eval_batch_size'):
        assert left['evaluation'][key] == right['evaluation'][key]
    assert '/train/' in right['offline']['expert_dataset_path']
    assert '/val/' in right['evaluation']['eval_path']
    assert right['model']['use_residual_edge_bias'] is True
    assert right['model']['use_post_charge_adapter'] is False
    assert left['offline']['policy_replay_enabled'] is False
    assert left['model']['use_static_rollout_cache'] is False


def test_profile_leaves_original_config_untouched_and_gates_charge_features():
    config = {'data': {'problem_type': 'evrptw'}}
    result = apply_optimization_profile(config, 'optimized')
    assert result['model']['use_post_charge_adapter']
    assert config == {'data': {'problem_type': 'evrptw'}}


def test_pairing_excludes_infeasible_short_partial_routes_and_uses_ids():
    baseline = [{'instance_id': 'a', 'feasible': True, 'objective_distance_km': 100},
                {'instance_id': 'b', 'feasible': True, 'objective_distance_km': 1000}]
    optimized = [{'instance_id': 'b', 'feasible': False, 'objective_distance_km': 1},
                 {'instance_id': 'a', 'feasible': True, 'objective_distance_km': 90}]
    result = comparison.paired_metrics(baseline, optimized)
    assert result['jointly_feasible_instances'] == 1
    assert result['mean_paired_improvement_pct'] == 10
    assert result['optimized_feasible_rate'] == .5
    assert result['baseline_feasible_rate'] == 1
    assert result['optimized_wins'] == 1
    with pytest.raises(ValueError, match='instance IDs differ'):
        comparison.paired_metrics(baseline, optimized[:1])
    with pytest.raises(ValueError, match='duplicate'):
        comparison.paired_metrics(baseline, optimized + [optimized[0]])


def test_optimized_ppo_does_not_enable_sl_only_route_replay():
    config = apply_optimization_profile({'offline': {'method': 'ppo'}}, 'optimized')
    assert config['model']['use_residual_edge_bias']
    assert not config['offline']['policy_replay_enabled']
