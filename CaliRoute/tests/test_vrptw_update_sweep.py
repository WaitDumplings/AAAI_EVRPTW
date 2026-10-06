"""Protect the update-pass sweep's controlled comparison and completion budget."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finetune_cus100_config import build_config
from run_vrptw_update_sweep import build_arm, successful_training
from offline2online.training_schedule import schedule_for_epoch


def test_only_update_passes_vary_and_global_batch_is_preserved(tmp_path):
    base = build_config(problem='vrptw', phase='long_candidate', run_name='source',
        output_dir=tmp_path / 'source', data_root=tmp_path / 'data',
        init_checkpoint=tmp_path / 'ppo.pt', ppo_step_chunk_size=36)
    original = copy.deepcopy(base)
    normalized = []
    for passes in (3, 4, 5, 6):
        out = tmp_path / str(passes)
        cfg = build_arm(base, updates=passes, output=out, run_name=str(passes),
            init_checkpoint=tmp_path / 'shared.pt', chunk_size=16)
        t = cfg['training']
        assert t['epochs'] == 500
        assert t['learning_rate'] == 3e-5
        assert t['amp_init_scale'] == 1024.
        assert t['num_envs_per_gpu'] * t['n_traj'] == 3200
        assert t['num_envs_per_gpu'] / t['num_minibatches'] == 16
        assert t['ppo_update_epochs'] == passes
        assert cfg['offline']['sl_coef'] == .35
        assert cfg['model'] == original['model']
        schedule_for_epoch(cfg, 1)
        schedule_for_epoch(cfg, 500)
        for section, key in [('training', 'ppo_update_epochs'), ('training', 'monitor_output_dir'),
                             ('evaluation', 'eval_output_dir')]:
            cfg[section].pop(key)
        cfg.pop('run_name')
        cfg.pop('experiment_protocol')
        normalized.append(cfg)
    assert all(c == normalized[0] for c in normalized)
    assert base == original


def test_never_count_partial_or_duplicate_epoch_history_as_completed(tmp_path):
    spec = dict(log_dir=str(tmp_path), checkpoint_dir=str(tmp_path))
    detail = dict(latest_validation=dict(epoch='500', eval_status='ok', eval_num_instances='1000'))
    (tmp_path / 'checkpoint_final.pt').touch()
    log = tmp_path / 'train_log.csv'
    def history(epochs):
        log.write_text('epoch\n' + ''.join(f'{n}\n' for n in epochs))
    history(range(1, 501))
    assert successful_training(spec, detail)
    history(range(1, 500))
    assert not successful_training(spec, detail)
    history([*range(1, 500), 499])
    assert not successful_training(spec, detail)
    history(range(1, 501))
    detail['latest_validation']['epoch'] = '480'
    assert not successful_training(spec, detail)
