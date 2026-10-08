#!/usr/bin/env python3
"""External VRPTW100 configuration for the unmodified f388343 SL-PPO runtime.

The literals below are transcribed from that commit's public CLI, configuration
builder and SL-PPO preset. This module does not import or execute either the
historical or current training runtime. Archive the complete original source
separately and run its ``python -m offline2online.train --config ... --seed ...``.

``data_root`` denotes AAAI_Dataset, matching the comparison launchers. The old
public CLI instead expected AAAI_Dataset/dataset; absolute split paths below
avoid that difference. All weights, optimizer state and policy memory start
fresh; expert training routes are retained because they are part of SL-PPO.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ORIGINAL_COMMIT = 'f388343dbb1d54bbd3f76dd29ca95208070d31a8'
ORIGINAL_SOURCE_BLOBS = {
    'CaliRoute/caliroute/cli.py': '51b1741f7af07614fa86929a1bb2e8f82bd5664d',
    'CaliRoute/caliroute/config.py': '73c394cb81f0ee17f7db3c054b3e45033906aa97',
    'CaliRoute/caliroute/methods.py': '9dd5f9e4b6b4170cdb44be8d657fbc9f2841a396',
    'CaliRoute/offline2online/trainer.py': '4c708847155aefa9c24bc905857b2e81fb65049b',
    'CaliRoute/offline2online/instance_adapter.py': '5c82295e4843406c2112f60b530b9fee676915b2',
    'CaliRoute/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py': 'acab9a47c69910cd1615c16b195cd122485a086d',
}

# Source: caliroute/config.py:182-260, cli.py:42-81 and methods.py:46-106
# at ORIGINAL_COMMIT. Paths and run_name are supplied by the external protocol.
ORIGINAL_PUBLIC_SLPPO = {
    'dataset_name': 'Geo-VRPTW-v1',
    'data': {
        'problem_type': 'vrptw',
        'num_customers': 100,
        'num_charging_stations': 0,
        'train_sample_mode': 'shuffle_cycle',
        'async_instance_prefetch': False,
    },
    'env': {
        'use_fast_env': True,
        'use_jit_mask': True,
        'normalize_reward': True,
        'reward_distance_scale_mode': 'dataset_single_customer_repair_median',
        'charging_mode': 'fixed_full',
        'info_level': 'light',
    },
    'model': {
        'embedding_dim': 256,
        'tanh_clipping': 15.0,
        'n_encode_layers': 2,
        'use_graph_token': True,
        'use_dynamic_decision_encoder': True,
        'dynamic_decision_heads': 4,
        'dynamic_decision_delta_k': False,
        'dynamic_decision_delta_v': False,
        'dynamic_decision_delta_action_key': True,
        'dynamic_decision_action_bias': True,
        'distance_injection': 'encoder',
        'use_encoder_distance_bias': True,
    },
    'critic': {'use_decomposed_critic': False, 'advantage_mode': 'total'},
    'training': {
        'online_training': False,
        'epochs': 1500,
        'num_envs_per_gpu': 128,
        'n_traj': 50,
        'rollout_steps': 120,
        'ppo_step_chunk_size': 32,
        'ppo_update_epochs': 4,
        'num_minibatches': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_coef': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'learning_rate': 1e-4,
        'weight_decay': 0.0,
        'max_grad_norm': 1.0,
        'checkpoint_interval': 50,
        'debug': True,
        'debug_log_every': 1,
        'mixed_precision': True,
    },
    'pbrs': {
        'use_customer_pbrs': False,
        'use_repair_distance_pbrs': False,
        'use_feasible_ratio_pbrs': False,
        'use_terminal_heuristic': False,
    },
    'evaluation': {
        'eval_interval': 20,
        'eval_n_traj': 50,
        'eval_decode_mode': 'sample',
        'eval_max_steps': 120,
        'eval_batch_size': 1000,
        'eval_info_level': 'light',
        'eval_save_routes': False,
    },
    'offline': {
        'method': 'sl_ppo',
        'init_checkpoint_strict': False,
        'strict_replay': False,
        'sl_coef': 0.50,
        'sl_clip_coef': 0.20,
        'only_success_route_loss': True,
        'sl_expert_candidate_weight': 0.60,
        'sl_expert_logprob_chunk_size': 4096,
        'use_priority_sampler': True,
        'priority_selection_mode': 'weighted',
        'priority_mix_rho': 0.50,
        'priority_alpha': 0.70,
    },
    'advantage': {
        'use_group_advantage': True,
        'group_adv_coef': 1.0,
        'group_adv_clip': 3.0,
        'group_adv_std_floor': 5.0,
        'group_infeasible_penalty': 10.0,
        'sl_include_reference_in_group_stats': True,
        'sl_use_memory_incumbent': True,
        'use_reference_advantage': True,
        'reference_adv_coef': 0.50,
        'reference_adv_rho': 1.0,
        'reference_adv_clip': 3.0,
        'reference_success_only': True,
        'reference_advantage_mode': 'absolute',
        'use_reference_soft_gate': False,
        'use_reference_memory_gate': False,
        'use_expert_solution_level': True,
        'sl_expert_candidate_weight': 0.60,
        'sl_candidate_clip': 2.0,
        'sl_candidate_std_floor': 5.0,
        'sl_candidate_gap_baseline': 'mean',
        'sl_candidate_gap_scale_coef': 1.0,
        'sl_candidate_gap_floor_ratio': 0.01,
        'sl_candidate_quality_gate_eta': 0.05,
        'sl_candidate_margin': 0.005,
        'sl_candidate_gate_eta': 0.05,
        'sl_candidate_use_current_incumbent_gate': True,
        'sl_candidate_use_memory_incumbent_gate': True,
        'sl_use_expert_candidate': True,
    },
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _override_summary(cfg):
    """Every altered historical parameter, including sampling, stays visible."""
    reasons = {
        'training.epochs': 'shared from-scratch training budget',
        'training.num_envs_per_gpu': 'shared single-GPU rollout batch',
        'training.ppo_step_chunk_size': 'GPU memory allocation; same minibatch update count',
        'training.rollout_steps': 'shared Cus100 horizon covering up to 100 single-customer routes',
        'training.learning_rate': 'explicit common learning-rate override',
        'evaluation.eval_interval': 'shared validation schedule',
        'evaluation.eval_max_steps': 'shared Cus100 complete-solution horizon',
        'evaluation.eval_batch_size': 'shared evaluation memory budget',
        'offline.use_priority_sampler': 'uniform shuffle_cycle in every arm; original weighted-priority sampler disabled explicitly',
    }
    out = []
    for dotted, reason in reasons.items():
        section, key = dotted.split('.')
        before, after = ORIGINAL_PUBLIC_SLPPO[section][key], cfg[section][key]
        if before != after:
            out.append({'parameter': dotted, 'original': before, 'used': after, 'reason': reason})
    return out


def build_original_config(*, output, run_name, data_root, seed, epochs,
                          chunk_size, eval_interval, learning_rate=1e-4):
    """Return a pure legacy config plus provenance; never load weights or train.

    ``output`` only identifies the comparison arm directory in metadata. The
    unchanged old trainer writes under its own source-root/results; the launcher
    must link that directory or track it explicitly. No filesystem is modified.
    """
    _positive_int(epochs, 'epochs')
    _positive_int(chunk_size, 'chunk_size')
    _positive_int(eval_interval, 'eval_interval')
    if chunk_size > 201:
        raise ValueError('chunk_size must be at most 201')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not isinstance(run_name, str) or not run_name.strip() or run_name in {'.', '..'} or '/' in run_name or '\\' in run_name:
        raise ValueError('run_name must be a nonempty directory name')

    root = Path(data_root).expanduser().resolve()
    train = root / 'dataset/vrptw/train/Cus100'
    val = root / 'dataset/vrptw/val/Cus100'
    cfg = deepcopy(ORIGINAL_PUBLIC_SLPPO)
    cfg['run_name'] = run_name
    cfg['data']['train_dataset_path'] = str(train)
    cfg['training'].update(epochs=epochs, num_envs_per_gpu=64, rollout_steps=201,
                           ppo_step_chunk_size=chunk_size, learning_rate=float(learning_rate))
    cfg['offline'].update(use_priority_sampler=False, expert_dataset_path=str(train),
                           expert_solution_path=str(train / 'expert_solutions.csv'))
    cfg['evaluation'].update(eval_interval=eval_interval, eval_max_steps=201,
                              eval_batch_size=32, eval_path=str(val),
                              gurobi_summary_path=str(val / 'gurobi_summary.csv'))
    cfg['experiment_protocol'] = {
        'phase': 'original_slppo_from_scratch',
        'arm': 'legacy',
        'seed': seed,
        'epochs': epochs,
        'eval_interval': eval_interval,
        'world_size': 1,
        'output_directory': str(Path(output).expanduser().resolve()),
        'global_instances_per_rollout': 64,
        'global_trajectories_per_rollout': 3200,
        'global_instances_per_optimizer_step': 16,
        'ppo_update_epochs': 4,
        'attempted_optimizer_steps_per_epoch': 16,
        'expected_train_instances': 5000,
        'validation_instances': 1000,
        'dataset_counts_verified_by_builder': False,
        'selection_split': 'val',
        'test_enabled': False,
        'initialization': 'random policy weights, fresh optimizer and policy memory; no init/resume/reference model checkpoint',
        'expert_data': 'training expert routes retained by the original SL-PPO method; from scratch does not mean expert-free',
        'sampling': 'uniform shuffle_cycle in every arm; explicit override of original weighted-priority sampling',
        'original_source': {
            'commit': ORIGINAL_COMMIT,
            'file_blobs': deepcopy(ORIGINAL_SOURCE_BLOBS),
            'configuration_sources': [
                'CaliRoute/caliroute/config.py:15-26,148-260',
                'CaliRoute/caliroute/cli.py:26-81',
                'CaliRoute/caliroute/methods.py:46-106',
            ],
            'runtime_sources': [
                'CaliRoute/offline2online/trainer.py:1903-1953,2097-2156,3818-3904,5432',
                'CaliRoute/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py:48-59,106-131,298-332,508-545,569-573',
                'CaliRoute/offline2online/instance_adapter.py:528-570',
            ],
            'execution_requirement': 'unmodified complete git archive of commit; never import current model/trainer/env as the legacy arm',
        },
        'protocol_overrides': _override_summary(cfg),
        'original_runtime_semantics': {
            'optimizer': {'name': 'AdamW', 'eps': 1e-5, 'weight_decay': 0.0, 'schedule': 'constant'},
            'ppo_loss_reduction': 'per-step valid-action mean, followed by time-step mean',
            'bootstrap_at_rollout_cutoff': False,
            'kl_early_stop': False,
            'input_coordinates': 'per-instance per-axis min-max',
            'input_distance': 'physical road D / the training-set reward D0',
            'input_time': 'time / instance working horizon',
            'input_energy': 'energy / instance battery capacity',
            'reward': '-physical road distance / D0; invalid/no-action penalty -10; success bonus 0',
            'reward_D0': 'median of pooled training customers depot round-trip distances; fitted by original trainer',
            'pbrs': 'all four public SL-PPO switches off, unchanged from original config.py:241-245',
        },
        'evaluation_caveats': [
            'Original trainer has no epoch-zero evaluation; only scheduled/final epochs are native.',
            'Original evaluation seeds sample actions by seed + epoch * 1000000 + batch offset.',
            'Original evaluator trusts environment success, without the newer independent route validator.',
            'Original eval_save_routes does not provide the modern evaluation output artifacts.',
            'Report feasibility with distance because native means include successful instances only.',
        ],
        'comparison_scope': 'old source and training semantics with declared common budget/sampling overrides; not a historical warm-start reproduction or single-factor ablation',
    }
    return cfg


def write_original_config(cfg, config_path):
    """Write the external YAML and provenance sidecar; return both paths."""
    import yaml

    path = Path(config_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)
    path.write_text(rendered, encoding='utf-8')
    manifest = deepcopy(cfg['experiment_protocol'])
    manifest['config_path'] = str(path)
    manifest['config_sha256'] = hashlib.sha256(rendered.encode('utf-8')).hexdigest()
    sidecar = path.with_suffix('.provenance.json')
    sidecar.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path, sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='External YAML filename; also writes .provenance.json')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='AAAI_Dataset directory, not its dataset child')
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--chunk-size', type=int, default=8)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    cfg = build_original_config(output=args.output.parent, run_name=args.run_name,
        data_root=args.data_root, seed=args.seed, epochs=args.epochs,
        chunk_size=args.chunk_size, eval_interval=args.eval_interval,
        learning_rate=args.learning_rate)
    for path in write_original_config(cfg, args.output):
        print(path)


if __name__ == '__main__':
    main()
