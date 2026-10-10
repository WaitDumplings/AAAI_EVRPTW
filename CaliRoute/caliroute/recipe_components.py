"""Describe resolved recipe branches without importing or changing the trainer.

This is configuration metadata, not a performance claim or runtime inspection.
The input must be the effective configuration after applying recipe overrides.
Resource activity, expert availability, scheduled replay and successful AMP
updates still require observations or logs; configured branches are not proof
that a particular minibatch exercised them.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


SCHEMA = "caliroute_recipe_components_v1"
_SL_METHODS = {
    "sl_ppo", "sl-ppo", "solution_level_ppo", "solution-level-ppo",
    "solution_ppo", "solution-ppo", "sl_candidate", "sl_candidate_ppo",
    "sl_candidate-ppo",
}


def _section(cfg, name):
    value = cfg.get(name) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _alias(section, actual, public, default):
    # Match trainer._apply_solution_level_aliases: the actual key wins.
    return section.get(actual, section.get(public, default))


def describe_components(cfg):
    """Return fresh JSON-compatible metadata for the current trainer's recipe.

    Does not validate a runnable config, load data, inspect hardware, seed RNGs,
    or mutate ``cfg``. Disabled parent branches suppress their child metadata.
    Source of truth: trainer.train, RewardNormalization.route_advantages,
    graph_attention_model_wrapper and Decoder.get_action_log_probs.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    model, train, offline, adv, env, data, protocol = (
        _section(cfg, name) for name in
        ("model", "training", "offline", "advantage", "env", "data", "experiment_protocol")
    )
    joint = bool(model.get("use_joint_graph_encoder", False))
    relation = bool(model.get("use_edge_relation_encoder", False))
    edges = joint or relation
    resources = bool(model.get("use_resource_decoder", False))
    dde = bool(model.get("use_dynamic_decision_encoder", False))
    physical_candidates = dde and bool(model.get("agda_physical_candidate_features", False))
    context = bool(model.get("use_physical_input_context", False))
    typed = bool(model.get("use_typed_static_fusion", False))
    distance_mode = str(model.get("distance_injection", "encoder")).strip().lower().replace("-", "_")
    distance_bias = bool(model.get("use_encoder_distance_bias", distance_mode in {"encoder", "encoder_bias", "road_encoder"}))
    edge_dim = int(model.get("joint_graph_edge_dim", 32) if joint else model.get("edge_relation_dim", 16)) if edges else None
    method = str(offline.get("method", "ppo")).strip().lower()
    sl = method in _SL_METHODS
    physical_norm = str(train.get("reward_norm_mode", "legacy")) == "physical_shared_popart"
    physical_sl = sl and physical_norm
    sl_coef = float(offline.get("sl_coef", offline.get("route_loss_coef", .10)))
    expert_switch = adv.get("use_expert_solution_level", offline.get("use_expert_solution_level"))
    if expert_switch is None:
        expert_switch = float(adv.get("sl_expert_candidate_weight", offline.get(
            "sl_expert_candidate_weight", offline.get("expert_sl_weight", 0.)))) > 0
    expert_requested = sl and bool(expert_switch) and bool(_alias(
        adv, "sl_candidate_use_expert_candidate", "sl_use_expert_candidate", True))
    # The actual expert preparation uses advantage.*, not the offline fallback
    # consulted by the enable predicate. Keep this subtle distinction explicit.
    expert_weight = float(adv.get("sl_expert_candidate_weight", 2.0))
    replay = sl and bool(offline.get("policy_replay_enabled", False))
    replay_weight = float(offline.get("policy_replay_weight", .2))
    replay_fraction = float(offline.get("policy_replay_fraction", .25))
    replay_candidates = int(offline.get("policy_replay_max_candidates", 16))
    search_requested = sl and bool(offline.get(
        "branch_exploration_enabled", offline.get("exploration_enabled", False)))
    passes = int(train.get("ppo_update_epochs", 3))
    batch = int(train.get("num_envs_per_gpu", 128))
    minibatches = min(max(1, int(train.get("num_minibatches", 4))), batch)
    accumulation = max(1, int(train.get("gradient_accumulation_steps", 1)))
    world = protocol.get("world_size", protocol.get("world_size_per_arm"))
    world = int(world) if world is not None else None
    target = train.get("target_kl")

    static_channels = ["base_node_geometry_demand_time_and_node_type"]
    if context:
        static_channels.append("physical_node_and_graph_context")
    if typed:
        static_channels.append("typed_geometry_time_load_road_and_type_fusion")
    if distance_bias:
        static_channels.append("linear_road_distance_attention_prior")
    if model.get("use_rdi_v2", False):
        static_channels.append("rdi_v2_directed_physical_residual_bias")
    if model.get("use_residual_edge_bias", False):
        static_channels.append("legacy_residual_edge_bias")
    if edges:
        static_channels.append("forward_reverse_DTE_validity_types_and_resource_context_relations")

    result = {
        "schema": SCHEMA,
        "scope": "resolved_configuration_only; runtime_activity_and_quality_require_logs",
        "routing_inputs": {
            "problem_type": str(data.get("problem_type", "unspecified")),
            "coordinate_mode": str(env.get("observation_coordinate_mode", "legacy_minmax")),
            "distance_unit_km": env.get("observation_distance_scale_km"),
            "strict_road_metric": bool(data.get("strict_road_metric", False)),
            "prefer_explicit_edge_matrices": bool(env.get("prefer_explicit_edge_matrices", False)),
            "network_requires_road_matrices": ["edge_distance", "edge_time", "edge_energy"],
            "context_emitted": bool(env.get("observation_input_context", False)),
            "context_consumed": context,
            "context_widths": {"node": 12, "graph": 10} if context else None,
            "resource_activity_source": "per_instance_graph_input_context_flags; not_inferred_from_task_name",
            "physical_feasibility_owner": "environment; learned_edges_do_not_modify_hard_masks",
        },
        "static_graph": {
            "encoder": "joint_node_edge" if joint else "current_graph_attention",
            "embedding_dim": int(model.get("embedding_dim", 256)),
            "layers": int(model.get("n_encode_layers", 2)),
            "active_input_channels": static_channels,
            "typed_static_fusion": typed,
            "rdi_v2_residual_bias": bool(model.get("use_rdi_v2", False)),
            "linear_distance_prior": distance_bias,
            "directed_edge_relations": edges,
            "effective_edge_dim": edge_dim,
            "node_incoming_outgoing_edge_fusion": joint,
            "post_softmax_edge_gates": joint,
            "edge_value_messages": joint or (relation and bool(model.get("use_edge_value_messages", False))),
            "edge_state_updates": joint or (relation and bool(model.get("use_edge_state_updates", False))),
            "decoder_reads_current_node_edge_row": edges,
            "edge_storage": "per_instance_B_N_N_R; independent_of_trajectory_count" if edges else None,
        },
        "dynamic_decision": {
            "dde": dde,
            "agda_v2_candidate_gate": dde and bool(model.get("use_agda_v2", False)),
            "dde_residuals": {
                name: dde and bool(model.get("dynamic_decision_" + name, model.get("dynamic_" + name, model.get(name, True))))
                for name in ("delta_k", "delta_v", "delta_action_key", "action_bias")
            },
            "physical_candidate_features": physical_candidates,
            "smooth_distance_features": dde and bool(model.get("agda_smooth_distance_features", False)),
            "resource_adapter_instantiated": resources or edges,
            "resource_candidate_fusion": resources,
            "dual_resource_readout": resources and model.get("decoder_observation_mode", "feasible") == "dual",
            "learned_edge_readout": edges,
            "transitions_shared_between_dde_and_resource_adapter": physical_candidates and resources,
            "post_charge_adapter": bool(model.get("use_post_charge_adapter", False)),
        },
        "training": {
            "method": method,
            "gamma": float(train.get("gamma", .99)),
            "reward_contract": str(env.get("reward_contract", "legacy")),
            "reward_unit_km": env.get("reward_distance_scale_km"),
            "normalization": "physical_shared_popart" if physical_norm else "legacy",
            "ppo_actor_advantage": "GAE_shared_actor_RMS" if physical_norm else "legacy_trainer_advantage_path",
            "critic_target": "scalar_PopArt" if physical_norm else "legacy_critic_path",
            "ppo_loss_reduction": str(train.get("ppo_loss_reduction", "legacy_step_mean")),
            "ppo_passes_cap": passes,
            "ppo_pass_policy": "fixed_passes" if target is None else "post_pass_fresh_KL_early_stop",
            "target_kl": float(target) if target is not None else None,
            "monitor_target_kl_controls_updates": False,
            "minibatches_per_pass": minibatches,
            "gradient_accumulation_steps": accumulation,
            "optimizer_attempts_per_epoch_cap": passes * math.ceil(minibatches / accumulation),
            "optimizer_attempt_scope": "per_rank_synchronized_steps; AMP_skips_or_KL_stop_can_reduce_successes",
            "batch_per_rank": batch,
            "world_size_from_protocol": world,
            "global_instances_per_rollout": batch * world if world is not None else None,
        },
        "slppo": {
            "enabled": sl,
            "coefficient": sl_coef if sl else 0.,
            "online_loss_configured": sl and sl_coef > 0,
            "online_route_advantage": ("leave_one_out_feasible_physical_cost_shared_RMS" if physical_sl else
                "legacy_solution_level_advantage_tensors" if sl else "disabled"),
            "online_baseline": "other_feasible_trajectories_of_same_instance; singleton_gets_zero" if physical_sl else None,
            "online_scale": "fixed_reward_unit_then_same_rollout_actor_RMS_snapshot_as_PPO" if physical_sl else None,
            "legacy_group_reference_controls_bypassed_for_online_advantage": physical_sl,
            "expert_and_replay_in_online_group": False if physical_sl else None,
            "only_success_route_loss": bool(offline.get("only_success_route_loss", True)) if sl else None,
            "expert": {
                "candidate_branch_requested": expert_requested,
                "loss_configured": expert_requested and expert_weight > 0 and sl_coef > 0,
                "candidate_weight": expert_weight if expert_requested else 0.,
                "current_incumbent_gate": expert_requested and bool(_alias(
                    adv, "sl_candidate_use_current_incumbent", "sl_candidate_use_current_incumbent_gate", True)),
                "memory_incumbent_gate": expert_requested and bool(_alias(
                    adv, "sl_candidate_use_memory_incumbent", "sl_candidate_use_memory_incumbent_gate", True)),
                "runtime_requirements": "available_expert_buffer_and_routes; feasibility_and_positive_quality_gate; optional_incumbent_gates",
                "normalization": "separate_dimensionless_candidate_objective; excluded_from_actor_RMS" if physical_sl else "legacy_candidate_objective",
            },
            "replay": {
                "archive_enabled": replay,
                "loss_configured": replay and replay_weight > 0 and replay_fraction > 0 and replay_candidates > 0 and sl_coef > 0,
                "configured_weight": replay_weight if replay else 0.,
                "warmup_epochs": int(offline.get("policy_replay_warmup_epochs", 0)),
                "ramp_epochs": int(offline.get("policy_replay_ramp_epochs", 0)),
                "selection": str(offline.get("policy_replay_selection", "legacy")),
                "capacity_per_instance": int(offline.get("policy_replay_capacity", 3)) if replay else 0,
                "exploration_capacity_per_instance": int(offline.get("policy_replay_exploration_capacity", 0)) if replay else 0,
                "normalization": "separate_dimensionless_candidate_objective; excluded_from_actor_RMS" if physical_sl else "legacy_candidate_objective",
                "runtime_requirements": "scheduled_weight_and_available_verified_candidates; archive_can_collect_when_weight_is_zero",
            },
            "search": {
                "requested": search_requested,
                "enabled_with_archive": search_requested and replay,
                "missing_required_archive": search_requested and not replay,
                "interval_epochs": int(offline.get("exploration_interval", 5)),
                "max_instances_per_rank": int(offline.get("exploration_instances", 8)) if search_requested else 0,
                "trajectories_per_instance": int(offline.get("exploration_trajectories", 8)) if search_requested else 0,
                "result_use": "verified_archive_routes_for_future_epochs; excluded_from_current_PPO_rollout",
                "flag_precedence": "branch_exploration_enabled_over_exploration_enabled",
            },
        },
        "evidence": {
            "controlled_recipe_comparisons": [] if protocol.get("preset") == "core" else ["graph_vs_current_encoder_bundle"],
            "parent_reference_comparisons": ["graph_vs_current_encoder_bundle"] if protocol.get("preset") == "core" else [],
            "configured_recipe_quality_evidence": protocol.get("configured_recipe_quality_evidence", "consult_run_provenance"),
            "controlled_comparison_scope": "encoder_and_matching_latent_edge_decoder_interface_under_matched_training_protocol",
            "not_independently_established": ["each_joint_graph_gate_value_update", "rdi_residual", "agda_gate", "P0_P1_added_to_joint_graph"],
            "correctness_and_efficiency_evidence_is_not_quality_ablation": True,
        },
        "notes": [
            "use_rdi_v2=False disables only residual attention bias; other road-information channels remain.",
            "use_agda_v2=False disables only candidate residual modulation; DDE and resource/edge adapters may remain.",
            "Graph/current is an encoder bundle comparison, including edge width and decoder edge projections; not an isolated edge-gate ablation.",
            "Resource activity and realized expert/replay/search contributions cannot be established from config alone.",
            "P0/P1 historical branches are not silently enabled by this description or merged into the current joint graph.",
        ],
    }
    return result
