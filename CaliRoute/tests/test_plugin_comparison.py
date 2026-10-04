from argparse import Namespace
import importlib.util
import math
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/run_plugin_comparison.py'
spec = importlib.util.spec_from_file_location('plugin_comparison', path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_long_two_gpu_protocol_is_equal_across_models(tmp_path):
    args = Namespace(problem='cvrp', customers=50, data_root=tmp_path / 'dataset', init_checkpoint=tmp_path / 'ppo.pt',
                     epochs=1000, num_envs=64, n_traj=50, learning_rate=5e-5 * math.sqrt(2),
                     eval_interval=20, eval_batch_size=128, seed=3009, eval_limit=None, expert_limit=None,
                     num_minibatches=4, checkpoint_interval=50, lr_warmup_epochs=20, lr_min=1e-5, monitor_interval=20)
    cfg = launcher.build_long_configs(args, tmp_path / 'experiment')
    a, b = cfg['baseline'], cfg['optimized']
    assert a['data'] == b['data']
    assert a['offline']['init_checkpoint_path'] == b['offline']['init_checkpoint_path']
    for key in ('num_envs_per_gpu', 'n_traj', 'num_minibatches', 'ppo_update_epochs', 'learning_rate',
                'lr_schedule', 'lr_warmup_epochs', 'lr_min', 'entropy_initial_coef', 'entropy_final_coef'):
        assert a['training'][key] == b['training'][key]
    assert not a['model']['use_rdi_v2']
    assert b['model']['use_rdi_v2'] and b['model']['use_agda_v2']
    assert not b['model']['use_residual_edge_bias']
    assert b['advantage']['sl_advantage_scale_mode'] == 'relative'
    assert b['offline']['policy_replay_warmup_epochs'] == 25
    assert b['offline']['policy_replay_weight'] == .1


@pytest.mark.parametrize('left,right', [('0,1', '1,2'), ('0', '1,2'), ('0,x', '1,2')])
def test_gpu_assignments_cannot_overlap_or_silently_run_one_rank(left, right):
    with pytest.raises(ValueError):
        launcher.parse_gpu_pairs(left, right)
