from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from .methods import canonical_method, method_preset


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DATA_ROOT = REPO_ROOT.parent / "AAAI_Dataset" / "dataset"
DEFAULT_EPOCHS = 1500
DEFAULT_NUM_ENVS = 128
DEFAULT_N_TRAJ = 50
DEFAULT_PPO_STEP_CHUNK = 32
DEFAULT_NUM_MINIBATCHES = 4
DEFAULT_EVAL_INTERVAL = 20
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_ROLLOUT_STEPS = {
    "cvrp": 110,
    "vrptw": 120,
    "evrptw": 160,
}


def normalize_problem(problem: str) -> str:
    key = str(problem).strip().lower()
    aliases = {
        "cvrp": "cvrp",
        "vrptw": "vrptw",
        "cvrptw": "vrptw",
        "evrptw": "evrptw",
        "evrp-tw": "evrptw",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("problem must be one of: cvrp, vrptw/cvrptw, evrptw") from exc


def default_charging_stations(problem: str, customers: int) -> int:
    if normalize_problem(problem) != "evrptw":
        return 0
    return max(1, int(customers) // 5)


def dataset_name(problem: str) -> str:
    label = normalize_problem(problem).upper()
    return f"Geo-{label}-v1"


def public_split_path(data_root: str | Path, problem: str, split: str, customers: int) -> Path:
    return Path(data_root) / normalize_problem(problem) / split / f"Cus{int(customers)}"


def _as_path_or_none(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _existing_csv(directory: Path | None, names: tuple[str, ...]) -> Path | None:
    if directory is None:
        return None
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _on(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def qkv_delta_flags(mode: str) -> tuple[bool, bool]:
    key = str(mode).strip().lower()
    if key in {"none", "off", "no"}:
        return False, False
    if key == "k":
        return True, False
    if key == "v":
        return False, True
    if key in {"kv", "qkv", "all"}:
        return True, True
    raise ValueError("qkv delta mode must be one of: none, k, v, kv")


def _clean_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean_none(v) for v in obj]
    return obj


def _method_offline_config(args: Namespace, train_data: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = method_preset(args.offline_method)
    offline_cfg = preset.offline_config()
    advantage_cfg = preset.advantage_config()

    expert_data = _as_path_or_none(args.expert_data) or train_data
    expert_solution = _as_path_or_none(args.expert_solution)
    if expert_solution is None and preset.requires_expert:
        expert_solution = _existing_csv(train_data, ("expert_solutions.csv", "gurobi_summary.csv"))
    if preset.requires_expert:
        if expert_solution is None:
            raise ValueError(
                f"{preset.name} requires expert routes. Pass --expert-solution or place "
                "expert_solutions.csv/gurobi_summary.csv under --train-data."
            )
        offline_cfg["expert_solution_path"] = str(expert_solution)
        offline_cfg["expert_dataset_path"] = str(expert_data)

    if args.init_checkpoint:
        offline_cfg["init_checkpoint_path"] = str(Path(args.init_checkpoint))

    if canonical_method(args.offline_method) == "slppo":
        pool = str(args.pool).strip().lower()
        if pool == "off":
            offline_cfg["use_priority_sampler"] = False
        else:
            offline_cfg["use_priority_sampler"] = True
            offline_cfg["priority_selection_mode"] = "best" if pool == "best" else "weighted"
        if args.sl_coef is not None:
            offline_cfg["sl_coef"] = float(args.sl_coef)
        if args.sl_expert_candidate_weight is not None:
            offline_cfg["sl_expert_candidate_weight"] = float(args.sl_expert_candidate_weight)

    if args.bc_coef is not None:
        offline_cfg["bc_coef"] = float(args.bc_coef)
    if args.bc_batch_size is not None:
        offline_cfg["bc_batch_size"] = int(args.bc_batch_size)
    if args.bc_updates_per_epoch is not None:
        offline_cfg["bc_updates_per_epoch"] = int(args.bc_updates_per_epoch)
    if args.awbc_coef is not None:
        offline_cfg["awbc_coef"] = float(args.awbc_coef)

    return offline_cfg, advantage_cfg


def build_training_config(args: Namespace) -> dict[str, Any]:
    problem = normalize_problem(args.problem)
    method = canonical_method(args.offline_method)
    customers = int(args.customers)
    charging_stations = (
        int(args.charging_stations)
        if args.charging_stations is not None
        else default_charging_stations(problem, customers)
    )
    if problem != "evrptw":
        charging_stations = 0

    data_root = Path(args.data_root)
    train_data = _as_path_or_none(args.train_data) or public_split_path(data_root, problem, "train", customers)
    val_data = _as_path_or_none(args.val_data) or public_split_path(data_root, problem, "val", customers)
    rollout_steps = int(args.rollout_steps or DEFAULT_ROLLOUT_STEPS[problem])
    eval_max_steps = int(args.eval_max_steps or rollout_steps)
    ppo_update_epochs = int(args.ppo_update_epochs or method_preset(method).ppo_update_epochs)
    delta_k, delta_v = qkv_delta_flags(args.qkv_delta)
    distance_injection = str(args.distance_injection).strip().lower()
    if distance_injection not in {"encoder", "none"}:
        raise ValueError("distance injection currently supports encoder or none; embedding is reserved for the next ablation hook")

    eval_summary = _as_path_or_none(args.gurobi_summary)
    if eval_summary is None:
        eval_summary = _existing_csv(val_data, ("gurobi_summary.csv", "expert_solutions.csv"))

    offline_cfg, advantage_cfg = _method_offline_config(args, train_data)
    run_name = args.run_name or (
        f"CALIROUTE_{problem.upper()}_CUS{customers}_CS{charging_stations}_"
        f"{method.upper()}_SEED{int(args.seed)}_E{int(args.epochs)}_"
        f"N{int(args.num_envs)}_R{rollout_steps}"
    )

    cfg: dict[str, Any] = {
        "run_name": run_name,
        "dataset_name": dataset_name(problem),
        "data": {
            "problem_type": problem,
            "num_customers": customers,
            "num_charging_stations": charging_stations,
            "train_dataset_path": str(train_data),
            "train_sample_mode": "shuffle_cycle",
            "async_instance_prefetch": bool(args.async_instance_prefetch),
        },
        "env": {
            "use_fast_env": True,
            "use_jit_mask": True,
            "normalize_reward": True,
            "reward_distance_scale_mode": "dataset_single_customer_repair_median",
            "charging_mode": "fixed_full",
            "info_level": "light",
        },
        "model": {
            "embedding_dim": int(args.embedding_dim),
            "tanh_clipping": 15.0,
            "n_encode_layers": int(args.encoder_layers),
            "use_graph_token": True,
            "use_dynamic_decision_encoder": _on(args.dde),
            "dynamic_decision_heads": int(args.dde_heads),
            "dynamic_decision_delta_k": delta_k,
            "dynamic_decision_delta_v": delta_v,
            "dynamic_decision_delta_action_key": _on(args.action_key),
            "dynamic_decision_action_bias": _on(args.action_bias),
            "distance_injection": distance_injection,
            "use_encoder_distance_bias": distance_injection == "encoder",
        },
        "critic": {
            "use_decomposed_critic": False,
            "advantage_mode": "total",
        },
        "training": {
            "online_training": False,
            "epochs": int(args.epochs),
            "num_envs_per_gpu": int(args.num_envs),
            "n_traj": int(args.n_traj),
            "rollout_steps": rollout_steps,
            "ppo_step_chunk_size": int(args.ppo_step_chunk_size),
            "ppo_update_epochs": ppo_update_epochs,
            "num_minibatches": int(args.num_minibatches),
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "learning_rate": float(args.learning_rate),
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "checkpoint_interval": int(args.checkpoint_interval),
            "debug": bool(args.debug),
            "debug_log_every": int(args.debug_log_every),
            "mixed_precision": bool(args.mixed_precision),
        },
        "pbrs": {
            "use_customer_pbrs": False,
            "use_repair_distance_pbrs": False,
            "use_feasible_ratio_pbrs": False,
            "use_terminal_heuristic": False,
        },
        "evaluation": {
            "eval_interval": int(args.eval_interval),
            "eval_path": str(val_data),
            "eval_n_traj": int(args.eval_n_traj),
            "eval_decode_mode": "sample",
            "eval_max_steps": eval_max_steps,
            "eval_limit": args.eval_limit,
            "eval_batch_size": int(args.eval_batch_size),
            "eval_info_level": "light",
            "eval_save_routes": bool(args.eval_save_routes),
            "gurobi_summary_path": str(eval_summary) if eval_summary is not None else None,
        },
        "offline": offline_cfg,
        "advantage": advantage_cfg,
    }
    if args.resume_checkpoint:
        cfg["training"]["resume_checkpoint_path"] = str(Path(args.resume_checkpoint))
    if args.resume_start_epoch is not None:
        cfg["training"]["resume_start_epoch"] = int(args.resume_start_epoch)
    return _clean_none(cfg)


def config_as_yaml(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)
