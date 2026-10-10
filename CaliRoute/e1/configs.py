"""E1 experiment definitions. Hardware changes execution, never global budgets."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from caliroute.methods import method_preset

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ppo_base', 'ppo_rdi_agda', 'awbc', 'dapg', 'slppo', 'rrnco', 'radar')
CONTROLLED = METHODS[:5]
EXPERT_METHODS = ('awbc', 'dapg', 'slppo')
SCHEMA = 'aaai_e1_cvrp100_v1'
# Five controlled experiments use ten 2080 Ti cards; no hardware timing comparison
# is inferred between the native A6000 runs and controlled 2080 Ti runs.
SERVER_PLANS = {
    '2080ti_a': [('ppo_base', [0, 1]), ('ppo_rdi_agda', [2, 3])],
    '2080ti_b': [('awbc', [0, 1]), ('dapg', [2, 3])],
    '2080ti_c': [('slppo', [0, 1])],
    'a6000': [('rrnco', [0]), ('radar', [1])],
    '2080ti_3': [],
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def fit_train_distance_unit(train_path):
    """Historical dataset_single_customer_repair_median, fit on train only.

    This is a physical edge unit, not total route length or customer count.
    Freeze the result for all validation/test inputs, rewards and methods.
    """
    from offline2online.instance_adapter import iter_adapted_instances
    train_path = Path(train_path).resolve()
    if train_path.parts[-3:] != ('cvrp', 'train', 'Cus100'):
        raise ValueError('E1 units may be fitted only on the declared CVRP100 train split')
    repairs = []
    for instance in iter_adapted_instances(train_path, problem_type='cvrp', strict_road_metric=True):
        d = np.asarray(instance.distance_matrix_km, dtype=np.float64)
        r = d[0, 1:] + d[1:, 0]
        repairs.extend(r[np.isfinite(r)].tolist())
    unit = float(np.median(repairs)) if repairs else float('nan')
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Train distance unit must be positive and finite')
    return dict(km=unit, mode='single_customer_repair_median', fit_split='train',
                source_path=str(train_path), source_sha256=digest(train_path/'instances.pkl'), count=len(repairs))

def _model(base=False):
    reference = yaml.safe_load((ROOT / 'configs/recipes/aaai_graph_v1.yaml').read_text())['config']['model']
    model = copy.deepcopy(reference)
    core = yaml.safe_load((ROOT / 'configs/recipes/aaai_graph_core_v1.yaml').read_text())['overrides']['model']
    model.update(core)
    model['e1_base_distance_row'] = True
    if base:
        for flag in ('use_joint_graph_encoder', 'use_edge_relation_encoder', 'use_edge_state_updates',
                     'use_edge_value_messages', 'use_encoder_distance_bias', 'use_rdi_v2',
                     'use_residual_edge_bias', 'use_physical_input_context', 'use_typed_static_fusion',
                     'use_dynamic_decision_encoder', 'dynamic_decision_delta_k', 'dynamic_decision_delta_v',
                     'dynamic_decision_delta_action_key', 'dynamic_decision_action_bias', 'use_agda_v2',
                     'agda_physical_candidate_features', 'agda_smooth_distance_features',
                     'use_resource_decoder', 'use_post_charge_adapter'):
            model[flag] = False
        model['distance_injection'] = 'none'
    return model

def build_config(method, *, data_root, output_dir, expert_pool=None, distance_unit_km,
                 seed=3009, world_size=2, global_batch=64, epochs=1500,
                 ppo_chunk=8, expert_chunk=32, eval_batch=8, backup_root=None):
    if method not in METHODS:
        raise ValueError(f'Unknown E1 method: {method}')
    for name, value in dict(world_size=world_size, global_batch=global_batch, epochs=epochs,
                            ppo_chunk=ppo_chunk, expert_chunk=expert_chunk, eval_batch=eval_batch).items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if world_size not in (1, 2) or global_batch % (world_size * 4):
        raise ValueError('Use one/two ranks and a global batch divisible by world_size * 4 minibatches')
    if method not in CONTROLLED and world_size != 1:
        raise ValueError('E1 native runners currently require one GPU; never emulate DDP by duplicate independent training')
    if not np.isfinite(distance_unit_km) or distance_unit_km <= 0:
        raise ValueError('distance_unit_km must be fitted/frozen on train and positive')
    data = Path(data_root).resolve(); output = Path(output_dir).resolve()
    train = data / 'dataset/cvrp/train/Cus100'; val = data / 'dataset/cvrp/val/Cus100'
    protocol = dict(schema=SCHEMA, method=method, task='cvrp100', training_seed=int(seed),
        initialization='scratch', customer_count=100, requested_K=50, inference_seed=17000000 + int(seed),
        total_instance_exposures=epochs * global_batch, global_batch=global_batch, world_size=world_size,
        evaluation_interval_exposures=50 * global_batch, validation_instances=1000,
        selection_rule='highest validation feasibility; then lowest mean feasible independently validated km',
        controlled_epoch_meaning='one sampled global instance batch followed by PPO/loss updates, NOT a dataset pass',
        directed_road_metric=True, include_depot_returns=True, fixed_vehicle_cost=0,
        decode_cap=201, test_used_for_training_or_selection=False, training_seed_count=1,
        hardware_timing_comparable_only_on_same_hardware=True,
        checkpoint_backup_required=True, self_trajectory_replay=False, independent_search=False)
    if method not in CONTROLLED:
        return dict(schema=SCHEMA, method=method, training_seed=seed, run_id=output.name,
            train_path=str(train), val_path=str(val), instance_exposures=epochs*global_batch,
            batch_size=global_batch, n_traj=50, eval_k=50, eval_interval_exposures=50*global_batch,
            eval_batch_size=eval_batch, eval_seed=17000000+seed, max_steps=201,
            checkpoint_backup_dir=str(Path(backup_root).resolve() / output.name) if backup_root else None,
            experiment_protocol=protocol)
    preset = method_preset('ppo' if method.startswith('ppo_') else method)
    offline = preset.offline_config()
    offline.update(policy_replay_enabled=False, branch_exploration_enabled=False, exploration_enabled=False,
                   use_priority_sampler=False, share_expert_static_observations=True)
    if method in EXPERT_METHODS:
        if expert_pool is None:
            raise ValueError(f'{method} requires the common audited train expert pool')
        offline.update(expert_dataset_path=str(train), expert_solution_path=str(Path(expert_pool).resolve()), strict_replay=True)
    if method == 'slppo':
        offline.update(solution_reference_contract='e1_cost_incumbent_v1', sl_expert_logprob_chunk_size=expert_chunk)
    model = _model(base=method == 'ppo_base')
    protocol.update(ppo_passes=4, num_minibatches=4, gradient_accumulation_steps=1,
                    configured_optimizer_attempts_per_loop=16, n_traj=50,
                    total_online_trajectory_attempts=epochs*global_batch*50,
                    actor_sha256=content_hash(model), method_preset_sha256=content_hash(dict(
                        offline=preset.offline_config(), advantage=preset.advantage_config())),
                    base_preserves=['query', 'vehicle_state', 'node_attributes', 'mask', 'decoder_distance_row'],
                    method_specific_expert_work='Additional work; explicitly counted, not claimed equal compute')
    warmup = int(offline.get('bc_warmup_epochs', 0))
    protocol.update(total_online_loops=epochs, total_outer_loops=epochs+warmup, bc_only_warmup_loops=warmup)
    advantage = preset.advantage_config()
    if method == 'slppo': advantage['sl_expert_logprob_chunk_size'] = expert_chunk
    training = dict(online_training=False, epochs=epochs+warmup, num_envs_per_gpu=global_batch//world_size,
        n_traj=50, rollout_steps=201, require_complete_feasible_rollouts=True,
        ppo_step_chunk_size=ppo_chunk, ppo_update_epochs=4, num_minibatches=4,
        gradient_accumulation_steps=1, gamma=.99, gae_lambda=.95, clip_coef=.2, vf_coef=.5,
        ent_coef=.01, learning_rate=1e-4, weight_decay=0., max_grad_norm=1., target_kl=None,
        lr_schedule='constant', lr_min=1e-4, mixed_precision=False, reward_norm_mode='legacy',
        ppo_loss_reduction='legacy_step_mean', bootstrap_truncation=False,
        checkpoint_interval=50, latest_checkpoint_interval=5,
        track_experiment_budget=True, debug=True, debug_log_every=1, profile_timing=True,
        monitor_interval=10, post_update_kl_interval=50, monitor_gradient_components=False,
        monitor_output_dir=str(output/'monitoring'), cache_expert_route_encoding=True,
        share_ppo_sl_forward=True, distributed_timeout_minutes=120)
    if backup_root:
        training['checkpoint_backup_dir'] = str(Path(backup_root).resolve())
    return dict(run_name=output.name, dataset_name='Geo-CVRP-v1',
        data=dict(problem_type='cvrp', num_customers=100, num_charging_stations=0,
            train_dataset_path=str(train), train_sample_mode='shuffle_cycle',
            async_instance_prefetch=False, strict_road_metric=True),
        env=dict(use_fast_env=True, use_jit_mask=True, normalize_reward=True, reward_contract='legacy',
            reward_mode='distance', reward_distance_scale_mode='single_customer_repair_median',
            reward_distance_scale_km=float(distance_unit_km), observation_distance_scale_km=float(distance_unit_km),
            observation_coordinate_mode='depot_fixed', observation_input_context=model['use_physical_input_context'],
            prefer_explicit_edge_matrices=True, charging_mode='fixed_full', info_level='light', max_steps_factor=4),
        model=model, critic=dict(use_decomposed_critic=False, advantage_mode='total'), training=training,
        offline=offline, advantage=advantage,
        pbrs=dict(use_customer_pbrs=False, use_repair_distance_pbrs=False,
                  use_feasible_ratio_pbrs=False, use_terminal_heuristic=False),
        evaluation=dict(eval_before_training=False, eval_interval=50, eval_epoch_offset=warmup, eval_path=str(val),
            eval_n_traj=50, eval_decode_mode='sample', eval_max_steps=201, eval_limit=None,
            eval_batch_size=eval_batch, eval_info_level='light', eval_save_routes=True,
            eval_seed=17000000+seed, eval_output_dir=str(output/'validation'), gurobi_summary_path=None),
        experiment_protocol=protocol)
