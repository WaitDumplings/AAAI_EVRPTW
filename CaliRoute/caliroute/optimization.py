"""Explicit, independently switchable optimization settings for ablations."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_optimization_profile(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {"baseline", "optimized"}:
        raise ValueError("optimization profile must be baseline or optimized")
    cfg = deepcopy(cfg)
    enabled = profile == "optimized"
    cfg.setdefault("model", {}).update({
        "use_residual_edge_bias": enabled,
        "residual_edge_hidden_dim": 32,
        "use_post_charge_adapter": enabled and cfg.get("data", {}).get("problem_type") == "evrptw",
        "post_charge_adapter_hidden_dim": 32,
        "optimize_dynamic_projections": enabled,
        "cache_static_observations": enabled,
        "use_static_rollout_cache": enabled,
    })
    cfg.setdefault("training", {}).update({
        "share_ppo_sl_forward": enabled,
        "cache_expert_route_encoding": enabled,
    })
    method = str(cfg.get("offline", {}).get("method", "ppo")).lower().replace("-", "_")
    solution_level = method in {"slppo", "sl_ppo", "solution_level_ppo", "solution_ppo"}
    cfg.setdefault("offline", {}).update({
        "policy_replay_enabled": enabled and solution_level,
        "policy_replay_capacity": 3,
        "policy_replay_fraction": 0.25,
        "policy_replay_max_candidates": 16,
        "policy_replay_max_new_routes": 32,
        "policy_replay_max_relative_gap": 0.05,
        "policy_replay_min_edge_distance": 0.10,
        "policy_replay_weight": 0.20,
    })
    cfg["optimization_profile"] = profile
    return cfg
