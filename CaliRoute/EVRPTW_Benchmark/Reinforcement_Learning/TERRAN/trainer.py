from __future__ import annotations

import csv
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import iter_instances

from .async_instances import AsyncInstancePool
from .data_pool import FixedDatasetInstancePool, OnlineInstancePool
from .env_factory import make_terran_env
from .models import Agent
from .pbrs import PotentialRewardConfig
from .rollout import collect_rollout, compute_returns, rollout_eval_batch


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        local = Path(__file__).resolve().parent / "configs" / cfg_path
        cfg_path = local if local.exists() else cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return (value * mask_f).sum() / denom


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def _eval_instance_batches(
    eval_path: Path,
    num_customers: int,
    num_charging_stations: int,
    batch_size: int,
    limit: int | None = None,
    num_batches_limit: int | None = None,
):
    max_count = None if limit is None else int(limit)
    if num_batches_limit is not None:
        by_batches = max(1, int(batch_size)) * int(num_batches_limit)
        max_count = by_batches if max_count is None else min(max_count, by_batches)
    batch = []
    seen = 0
    for instance in iter_instances(eval_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        if max_count is not None and seen >= max_count:
            break
        batch.append(instance)
        seen += 1
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _pbrs_enabled(cfg: dict[str, Any]) -> bool:
    pbrs = cfg.get("pbrs", {}) or {}
    return bool(
        pbrs.get("use_customer_pbrs", False)
        or pbrs.get("use_repair_distance_pbrs", False)
        or pbrs.get("use_feasible_ratio_pbrs", False)
        or pbrs.get("use_terminal_heuristic", False)
    )


def pbrs_scale_for_epoch(cfg: dict[str, Any], epoch: int, total_epochs: int) -> float:
    if not _pbrs_enabled(cfg):
        return 0.0
    pbrs = cfg.get("pbrs", {}) or {}
    annealing = pbrs.get("annealing", {}) or {}
    if not bool(annealing.get("enabled", False)):
        return float(annealing.get("start_scale", 1.0))
    start_scale = float(annealing.get("start_scale", 1.0))
    end_scale = float(annealing.get("end_scale", 0.2))
    start_epoch = max(1, int(annealing.get("start_epoch", 1)))
    end_epoch = max(start_epoch, int(annealing.get("end_epoch", total_epochs)))
    schedule = str(annealing.get("schedule", "cosine")).lower()
    if epoch <= start_epoch:
        return max(start_scale, 0.0)
    if epoch >= end_epoch:
        return max(end_scale, 0.0)
    progress = (float(epoch) - float(start_epoch)) / max(float(end_epoch - start_epoch), 1.0)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        weight = progress
    elif schedule == "exponential":
        if start_scale <= 0 or end_scale <= 0:
            weight = progress
            return max(start_scale + (end_scale - start_scale) * weight, 0.0)
        return max(start_scale * ((end_scale / start_scale) ** progress), 0.0)
    elif schedule == "constant":
        return max(start_scale, 0.0)
    else:
        weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    return max(start_scale + (end_scale - start_scale) * weight, 0.0)


def set_pbrs_reward_scale(envs: Sequence[Any], scale: float) -> None:
    for env in envs:
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            setter = getattr(current, "set_reward_scale", None)
            if callable(setter):
                setter(scale)
                break
            current = getattr(current, "env", None)


def build_pbrs_config(cfg: dict[str, Any]) -> PotentialRewardConfig | None:
    pbrs = cfg.get("pbrs", {})
    config = PotentialRewardConfig(
        use_customer_pbrs=bool(pbrs.get("use_customer_pbrs", False)),
        use_repair_distance_pbrs=bool(pbrs.get("use_repair_distance_pbrs", False)),
        use_feasible_ratio_pbrs=bool(pbrs.get("use_feasible_ratio_pbrs", False)),
        use_terminal_heuristic=bool(pbrs.get("use_terminal_heuristic", False)),
        customer_pbrs_mode=str(pbrs.get("customer_pbrs_mode", "progress")),
        gamma=float(cfg.get("training", {}).get("gamma", 0.99)),
        alpha=float(pbrs.get("alpha", 2.0)),
        beta=float(pbrs.get("beta", 0.5)),
        customer_pbrs_coef=float(pbrs.get("customer_pbrs_coef", 1.0)),
        customer_progress_budget=float(pbrs.get("customer_progress_budget", 0.5)),
        customer_progress_mix=float(pbrs.get("customer_progress_mix", 0.5)),
        repair_progress_coef=float(pbrs.get("repair_progress_coef", 0.5)),
        feasible_ratio_coef=float(pbrs.get("feasible_ratio_coef", 0.0)),
        pbrs_clip=pbrs.get("pbrs_clip", None),
        success_bonus=float(pbrs.get("success_bonus", 0.1)),
        failure_penalty=float(pbrs.get("failure_penalty", 0.5)),
    )
    if not (
        config.use_customer_pbrs
        or config.use_repair_distance_pbrs
        or config.use_feasible_ratio_pbrs
        or config.use_terminal_heuristic
    ):
        return None
    return config


def _configure_dataset_reward_scale(cfg: dict[str, Any], pool: Any) -> None:
    env_cfg = cfg.setdefault("env", {})
    mode = str(env_cfg.get("reward_distance_scale_mode", "single_customer_repair_median"))
    if not mode.startswith("dataset_"):
        return
    base_mode = mode[len("dataset_") :]
    scale_fn = getattr(pool, "reward_distance_scale_km", None)
    if not callable(scale_fn):
        raise ValueError(
            "reward_distance_scale_mode uses dataset_ prefix, but the training pool "
            "does not provide dataset-level reward scale statistics."
        )
    scale = float(scale_fn(base_mode))
    env_cfg["reward_distance_scale_mode"] = base_mode
    env_cfg["reward_distance_scale_km"] = scale
    cfg.setdefault("normalization", {})["reward_distance_scale_km"] = scale
    cfg["normalization"]["reward_distance_scale_mode"] = mode
    cfg["normalization"]["reward_distance_scale_base_mode"] = base_mode
    cfg["normalization"]["reward_distance_scale_source"] = getattr(pool, "region_pool_status", "dataset")


def make_envs(cfg: dict[str, Any], seed: int):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    num_envs = int(train_cfg.get("num_envs_per_gpu", 128))
    train_dataset_path = (
        data_cfg.get("train_dataset_path")
        or data_cfg.get("instance_dataset_path")
        or data_cfg.get("fixed_train_path")
    )
    if train_dataset_path not in (None, ""):
        pool = FixedDatasetInstancePool(
            dataset_path=train_dataset_path,
            num_customers=int(data_cfg.get("num_customers", 15)),
            num_charging_stations=int(data_cfg.get("num_charging_stations", 3)),
            seed=seed,
            sample_mode=str(data_cfg.get("train_sample_mode", "shuffle_cycle")),
        )
    else:
        common_pool_kwargs = dict(
            config_path=data_cfg.get("generator_config", "configs/amazon_hierarchy.yaml"),
            num_regions=int(data_cfg.get("mother_board_pool_size", 32)),
            mother_num_customers=int(data_cfg.get("mother_num_customers", 5000)),
            mother_num_charging_stations=int(data_cfg.get("mother_num_charging_stations", 120)),
            num_customers=int(data_cfg.get("num_customers", 15)),
            num_charging_stations=int(data_cfg.get("num_charging_stations", 3)),
            region_reuse_limit=int(data_cfg.get("region_reuse_limit", 200)),
            seed=seed,
            max_attempts_per_instance=data_cfg.get("max_attempts_per_instance"),
            territory_pool_path=data_cfg.get("territory_pool_path"),
            region_pool_path=data_cfg.get("region_pool_path"),
            region_pool_shuffle=bool(data_cfg.get("territory_pool_shuffle", data_cfg.get("region_pool_shuffle", True))),
            region_pool_replacement_policy=str(data_cfg.get("region_pool_replacement_policy", "cycle")),
        )
        if bool(data_cfg.get("async_instance_prefetch", False)):
            workers = int(data_cfg.get("async_instance_workers", min(8, max(1, num_envs))))
            queue_batches = int(data_cfg.get("async_instance_queue_batches", 2))
            regions_per_worker = data_cfg.get("async_regions_per_worker", None)
            pool = AsyncInstancePool(
                **common_pool_kwargs,
                num_workers=workers,
                queue_size=max(workers * 2, num_envs * max(1, queue_batches)),
                regions_per_worker=None if regions_per_worker is None else int(regions_per_worker),
                multiprocessing_context=str(data_cfg.get("async_multiprocessing_context", "spawn")),
                get_timeout_s=float(data_cfg.get("async_get_timeout_s", 300.0)),
            )
            pool.start()
        else:
            pool = OnlineInstancePool(**common_pool_kwargs)
    _configure_dataset_reward_scale(cfg, pool)
    pbrs_config = build_pbrs_config(cfg)
    env_cfg = dict(cfg.get("env", {}) or {})
    if bool(env_cfg.get("use_fast_env", True)):
        env_cfg.setdefault("info_level", "light")
    envs = [
        make_terran_env(
            instance_sampler=pool.sample,
            n_traj=int(train_cfg.get("n_traj", 50)),
            pbrs_config=pbrs_config,
            **env_cfg,
        )
        for _ in range(num_envs)
    ]
    return envs, pool

def evaluate_fixed_dataset(agent: Agent, cfg: dict[str, Any], seed: int, epoch: int, device: str | torch.device) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    num_customers = int(data_cfg.get("num_customers", 15))
    num_cs = int(data_cfg.get("num_charging_stations", 3))
    eval_path = _resolve_repo_path(eval_cfg.get("eval_path"))
    n_traj = int(eval_cfg.get("eval_n_traj", 8))
    decode_mode = str(eval_cfg.get("eval_decode_mode", "sample"))
    max_steps = int(eval_cfg.get("eval_max_steps", 128))
    limit = eval_cfg.get("eval_limit", None)
    batch_size = max(1, int(eval_cfg.get("eval_batch_size", 1)))
    num_batches_limit = eval_cfg.get("eval_num_batches", None)
    eval_save_routes = bool(eval_cfg.get("eval_save_routes", False))
    eval_info_level = str(eval_cfg.get("eval_info_level", "light"))
    if eval_path is None or not eval_path.exists():
        return {
            "eval_num_instances": 0,
            "eval_n_traj": n_traj,
            "eval_batch_size": batch_size,
            "eval_num_batches": 0,
            "eval_decode_mode": decode_mode,
            "eval_info_level": eval_info_level,
            "eval_save_routes": eval_save_routes,
            "eval_feasible_rate": np.nan,
            "eval_avg_objective_distance_km": np.nan,
            "eval_avg_vehicle_count": np.nan,
            "eval_avg_runtime_s": np.nan,
            "eval_status": f"missing_eval_path:{eval_path}",
        }
    was_training = agent.training
    agent.eval()
    rows: list[dict[str, Any]] = []
    num_batches = 0
    seen_before_batch = 0
    for instances in _eval_instance_batches(eval_path, num_customers, num_cs, batch_size, limit, num_batches_limit):
        eval_env_cfg = dict(cfg.get("env", {}) or {})
        if bool(eval_env_cfg.get("use_fast_env", True)):
            eval_env_cfg["info_level"] = "full" if eval_save_routes else eval_info_level
        envs = [make_terran_env(instance=instance, n_traj=n_traj, **eval_env_cfg) for instance in instances]
        batch_rows = rollout_eval_batch(
            agent,
            envs,
            decode_mode=decode_mode,
            max_steps=max_steps,
            device=device,
            seed=seed + epoch * 1_000_000 + seen_before_batch,
            include_routes=eval_save_routes,
        )
        for instance, row in zip(instances, batch_rows):
            row["instance_id"] = instance.instance_id
        rows.extend(batch_rows)
        num_batches += 1
        seen_before_batch += len(instances)
    if not rows:
        if was_training:
            agent.train()
        return {
            "eval_num_instances": 0,
            "eval_n_traj": n_traj,
            "eval_batch_size": batch_size,
            "eval_num_batches": 0,
            "eval_decode_mode": decode_mode,
            "eval_info_level": eval_info_level,
            "eval_save_routes": eval_save_routes,
            "eval_feasible_rate": np.nan,
            "eval_avg_objective_distance_km": np.nan,
            "eval_avg_vehicle_count": np.nan,
            "eval_avg_runtime_s": np.nan,
            "eval_status": f"no_instances:{eval_path}",
        }
    if was_training:
        agent.train()

    feasible_rows = [row for row in rows if row["feasible"]]
    return {
        "eval_num_instances": len(rows),
        "eval_n_traj": n_traj,
        "eval_batch_size": batch_size,
        "eval_num_batches": num_batches,
        "eval_decode_mode": decode_mode,
        "eval_info_level": eval_info_level,
        "eval_save_routes": eval_save_routes,
        "eval_feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "eval_avg_objective_distance_km": float(np.mean([row["objective_distance_km"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_vehicle_count": float(np.mean([row["vehicle_count"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_runtime_s": float(np.mean([row["runtime_s"] for row in rows])),
        "eval_status": "ok",
    }


def summarize_train_infos(final_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not final_infos:
        return {
            "train_feasible_rate": np.nan,
            "train_avg_best_objective_distance_km": np.nan,
            "train_avg_vehicle_count": np.nan,
            "train_avg_served_customers": np.nan,
        }
    feasible_flags = []
    best_objectives = []
    vehicle_counts = []
    served_counts = []
    for info in final_infos:
        success = np.asarray(info.get("success", []), dtype=bool)
        objective = np.asarray(info.get("objective_distance_km", []), dtype=np.float64)
        vehicle = np.asarray(info.get("vehicle_count", []), dtype=np.float64)
        served = np.asarray(info.get("served_customers", []), dtype=np.float64)
        if objective.size == 0:
            continue
        feasible_flags.extend(success.tolist())
        served_counts.extend(served.tolist())
        if np.any(success):
            candidates = np.where(success)[0]
            selected = int(candidates[np.argmin(objective[candidates])])
            best_objectives.append(float(objective[selected]))
            vehicle_counts.append(float(vehicle[selected]) if vehicle.size else np.nan)
    return {
        "train_feasible_rate": float(np.mean(feasible_flags)) if feasible_flags else np.nan,
        "train_avg_best_objective_distance_km": float(np.mean(best_objectives)) if best_objectives else np.nan,
        "train_avg_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else np.nan,
        "train_avg_served_customers": float(np.mean(served_counts)) if served_counts else np.nan,
    }


def _format_float(value: Any, precision: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{precision}f}"


def _debug_log(debug_enabled: bool, debug_file, message: str) -> None:
    if not debug_enabled:
        return
    print(message, flush=True)
    if debug_file is not None:
        debug_file.write(message + "\n")
        debug_file.flush()


def _slice_obs_by_env(obs: dict[str, Any], env_indices: Sequence[int] | np.ndarray) -> dict[str, Any]:
    indices = np.asarray(env_indices, dtype=np.int64)
    max_index = int(indices.max()) if indices.size else -1
    out: dict[str, Any] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] > max_index:
            out[key] = arr[indices]
        else:
            out[key] = value
    return out


def evaluate_policy_loss(
    agent,
    batch,
    returns,
    advantages,
    cfg,
    device,
    env_indices: Sequence[int] | np.ndarray | None = None,
    step_start: int = 0,
    step_end: int | None = None,
):
    del device
    clip_coef = float(cfg["training"].get("clip_coef", 0.2))
    vf_coef = float(cfg["training"].get("vf_coef", 0.5))
    ent_coef = float(cfg["training"].get("ent_coef", 0.01))
    if env_indices is None:
        env_indices = np.arange(batch.actions.size(1), dtype=np.int64)
    else:
        env_indices = np.asarray(env_indices, dtype=np.int64)

    if step_end is None:
        step_end = len(batch.observations)
    step_start = max(0, int(step_start))
    step_end = min(len(batch.observations), int(step_end))
    if step_start >= step_end:
        raise ValueError(f"empty PPO step range: [{step_start}, {step_end})")

    # Static node embeddings are identical across rollout steps. Encode once for
    # the selected env minibatch, then reuse cached K/V/logit projections while
    # each step supplies its own dynamic state. For large Cus1000-style graphs,
    # callers can invoke this function on time chunks to avoid retaining all
    # decoder graphs until a single backward pass.
    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))

    policy_losses = []
    value_losses = []
    entropy_losses = []
    for step in range(step_start, step_end):
        obs = batch.observations[step]
        obs_mb = _slice_obs_by_env(obs, env_indices)
        actions = batch.actions[step, env_indices].long()
        old_logprob = batch.old_logprobs[step, env_indices]
        _, new_logprob, entropy, value, _ = agent.get_action_and_value_cached(
            obs_mb,
            action=actions,
            state=cached_state,
        )
        value = value.squeeze(-1)
        ratio = torch.exp(new_logprob - old_logprob)
        adv = advantages[step, env_indices]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
        valid = batch.valid[step, env_indices]
        policy_losses.append(-masked_mean(torch.minimum(unclipped, clipped), valid))
        value_losses.append(masked_mean(F.mse_loss(value, returns[step, env_indices], reduction="none"), valid))
        entropy_losses.append(masked_mean(entropy, valid))
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach()


def save_checkpoint(path: Path, agent: Agent, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epoch: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "seed": int(seed),
            "config": cfg,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_from_config(cfg: dict[str, Any], seed: int, device: str | None = None, overrides: dict[str, Any] | None = None) -> Path:
    cfg = deep_update(cfg, overrides or {})
    set_seed(seed)
    train_cfg = cfg["training"]
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    run_name = str(cfg.get("run_name", "TERRAN"))
    num_customers = int(cfg["data"].get("num_customers", 15))
    num_cs = int(cfg["data"].get("num_charging_stations", 3))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 3)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", False)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        eps=1e-5,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    initial_env_start = time.perf_counter()
    envs, pool = make_envs(cfg, seed)
    initial_env_pool_time_s = time.perf_counter() - initial_env_start
    gamma = float(train_cfg.get("gamma", 0.99))
    epochs = int(train_cfg.get("epochs", 1000))
    rollout_steps = int(train_cfg.get("rollout_steps", 64))
    ppo_epochs = int(train_cfg.get("ppo_update_epochs", 4))
    num_minibatches = max(1, int(train_cfg.get("num_minibatches", 1)))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 50))
    eval_interval = int(eval_cfg.get("eval_interval", 0) or 0)
    debug_enabled = bool(train_cfg.get("debug", False))
    debug_log_every = max(1, int(train_cfg.get("debug_log_every", 1)))
    profile_timing = bool(train_cfg.get("profile_timing", False))
    ppo_step_chunk_size = int(train_cfg.get("ppo_step_chunk_size", 0) or 0)

    out_root = REPO_ROOT / "EVRPTW_Benchmark/Reinforcement_Learning/TERRAN"
    ckpt_dir = out_root / "checkpoints" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
    log_dir = out_root / "logs" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    eval_log_path = log_dir / "eval_log.csv"
    debug_log_path = log_dir / "debug_log.txt"

    train_fields = [
        "epoch",
        "reward_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "samples_seen",
        "num_envs",
        "n_traj",
        "rollout_steps",
        "num_minibatches",
        "gradient_accumulation_steps",
        "effective_instances_per_optimizer_step",
        "pbrs_scale",
        "initial_env_pool_time_s",
        "rollout_reset_time_s",
        "rollout_stack_obs_time_s",
        "rollout_model_action_time_s",
        "rollout_env_step_time_s",
        "rollout_interaction_time_s",
        "rollout_total_time_s",
        "ppo_update_time_s",
        "eval_wall_time_s",
        "epoch_wall_time_s",
        "train_feasible_rate",
        "train_avg_best_objective_distance_km",
        "train_avg_vehicle_count",
        "train_avg_served_customers",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]
    eval_fields = [
        "epoch",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]

    with log_path.open("w", newline="", encoding="utf-8") as f, eval_log_path.open("w", newline="", encoding="utf-8") as ef, debug_log_path.open("w", encoding="utf-8") as df:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        eval_writer = csv.DictWriter(ef, fieldnames=eval_fields)
        writer.writeheader()
        eval_writer.writeheader()
        _debug_log(
            debug_enabled,
            df,
            f"[Init] run={run_name} seed={seed} device={device} epochs={epochs} "
            f"n_traj={train_cfg.get('n_traj', 50)} rollout_steps={rollout_steps} "
            f"num_envs={train_cfg.get('num_envs_per_gpu', 128)} minibatches={num_minibatches} "
            f"accum_grad={gradient_accumulation_steps} "
            f"n_encode_layers={model_cfg.get('n_encode_layers', 3)} "
            f"initial_env_pool_time_s={initial_env_pool_time_s:.3f} "
            f"eval_interval={eval_interval} eval_n_traj={eval_cfg.get('eval_n_traj', 8)} "
            f"eval_batch_size={eval_cfg.get('eval_batch_size', 1)} "
            f"eval_info_level={eval_cfg.get('eval_info_level', 'light')} "
            f"pbrs_annealing={cfg.get('pbrs', {}).get('annealing', {})}",
        )
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            pbrs_scale = pbrs_scale_for_epoch(cfg, epoch, epochs)
            set_pbrs_reward_scale(envs, pbrs_scale)
            agent.train()
            batch = collect_rollout(
                agent,
                envs,
                rollout_steps=rollout_steps,
                decode_mode="sample",
                device=device,
                seed=seed + epoch * 100_000,
                profile_timing=profile_timing,
            )
            returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
            advantages = returns - batch.values
            valid = batch.valid
            adv_vals = advantages[valid]
            if adv_vals.numel() > 1:
                advantages = (advantages - adv_vals.mean()) / (adv_vals.std(unbiased=False) + 1e-8)
            losses = []
            num_envs = int(batch.actions.size(1))
            minibatches = min(num_minibatches, num_envs)
            env_order = np.arange(num_envs, dtype=np.int64)
            if profile_timing:
                _sync_cuda(device)
            ppo_start = time.perf_counter()
            total_steps = int(batch.actions.size(0))
            chunk_size = ppo_step_chunk_size if ppo_step_chunk_size > 0 else total_steps
            chunk_size = max(1, min(chunk_size, total_steps))
            for _ in range(ppo_epochs):
                np.random.shuffle(env_order)
                split_indices = [indices for indices in np.array_split(env_order, minibatches) if indices.size > 0]
                for group_start in range(0, len(split_indices), gradient_accumulation_steps):
                    accum_group = split_indices[group_start : group_start + gradient_accumulation_steps]
                    if not accum_group:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    group_policy = 0.0
                    group_value = 0.0
                    group_entropy = 0.0
                    group_size = float(len(accum_group))
                    for env_indices in accum_group:
                        weighted_policy = 0.0
                        weighted_value = 0.0
                        weighted_entropy = 0.0
                        for step_start in range(0, total_steps, chunk_size):
                            step_end = min(step_start + chunk_size, total_steps)
                            chunk_weight = float(step_end - step_start) / max(float(total_steps), 1.0)
                            loss, policy_loss, value_loss, entropy = evaluate_policy_loss(
                                agent,
                                batch,
                                returns,
                                advantages.detach(),
                                cfg,
                                device,
                                env_indices=env_indices,
                                step_start=step_start,
                                step_end=step_end,
                            )
                            (loss * chunk_weight / group_size).backward()
                            weighted_policy += policy_loss.item() * chunk_weight
                            weighted_value += value_loss.item() * chunk_weight
                            weighted_entropy += entropy.item() * chunk_weight
                        group_policy += weighted_policy / group_size
                        group_value += weighted_value / group_size
                        group_entropy += weighted_entropy / group_size
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    losses.append((group_policy, group_value, group_entropy))
            if profile_timing:
                _sync_cuda(device)
            ppo_update_time_s = time.perf_counter() - ppo_start
            reward_mean = float(batch.rewards[batch.valid].mean().detach().cpu().item()) if batch.valid.any() else 0.0
            loss_arr = np.asarray(losses, dtype=float)
            train_summary = summarize_train_infos(batch.final_infos)
            if epoch % debug_log_every == 0:
                _debug_log(
                    debug_enabled,
                    df,
                    "[Train] "
                    f"epoch={epoch}/{epochs} samples={pool.sample_count} "
                    f"reward={_format_float(reward_mean)} "
                    f"policy_loss={_format_float(loss_arr[:, 0].mean())} "
                    f"value_loss={_format_float(loss_arr[:, 1].mean())} "
                    f"entropy={_format_float(loss_arr[:, 2].mean())} "
                    f"train_fr={_format_float(train_summary['train_feasible_rate'])} "
                    f"train_obj={_format_float(train_summary['train_avg_best_objective_distance_km'])} "
                    f"train_veh={_format_float(train_summary['train_avg_vehicle_count'])} "
                    f"served={_format_float(train_summary['train_avg_served_customers'])} "
                    f"pbrs_scale={pbrs_scale:.4f} "
                    f"timing_reset={batch.timings.get('rollout_reset_time_s', 0.0):.3f}s "
                    f"timing_model={batch.timings.get('rollout_model_action_time_s', 0.0):.3f}s "
                    f"timing_env={batch.timings.get('rollout_env_step_time_s', 0.0):.3f}s "
                    f"timing_ppo={ppo_update_time_s:.3f}s",
                )
            eval_row: dict[str, Any] = {}
            eval_wall_time_s = 0.0
            should_eval = eval_interval > 0 and (epoch % eval_interval == 0 or epoch == epochs)
            if should_eval:
                eval_start = time.perf_counter()
                eval_row = evaluate_fixed_dataset(agent, cfg, seed=seed, epoch=epoch, device=device)
                eval_wall_time_s = time.perf_counter() - eval_start
                eval_writer.writerow({"epoch": epoch, **eval_row})
                ef.flush()
                _debug_log(
                    debug_enabled,
                    df,
                    "[Eval] "
                    f"epoch={epoch}/{epochs} n={eval_row.get('eval_num_instances')} "
                    f"n_traj={eval_row.get('eval_n_traj')} "
                    f"batch={eval_row.get('eval_batch_size')}x{eval_row.get('eval_num_batches')} "
                    f"mode={eval_row.get('eval_decode_mode')} "
                    f"info={eval_row.get('eval_info_level')} "
                    f"fr={_format_float(eval_row.get('eval_feasible_rate'))} "
                    f"obj={_format_float(eval_row.get('eval_avg_objective_distance_km'))} "
                    f"veh={_format_float(eval_row.get('eval_avg_vehicle_count'))} "
                    f"runtime={_format_float(eval_row.get('eval_avg_runtime_s'))} "
                    f"eval_wall={eval_wall_time_s:.3f}s "
                    f"status={eval_row.get('eval_status')}",
                )
            epoch_wall_time_s = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "reward_mean": reward_mean,
                    "policy_loss": float(loss_arr[:, 0].mean()),
                    "value_loss": float(loss_arr[:, 1].mean()),
                    "entropy": float(loss_arr[:, 2].mean()),
                    "samples_seen": pool.sample_count,
                    "num_envs": num_envs,
                    "n_traj": int(train_cfg.get("n_traj", 50)),
                    "rollout_steps": rollout_steps,
                    "num_minibatches": minibatches,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_instances_per_optimizer_step": int(np.ceil(num_envs / max(minibatches, 1))) * gradient_accumulation_steps,
                    "pbrs_scale": pbrs_scale,
                    "initial_env_pool_time_s": initial_env_pool_time_s,
                    "rollout_reset_time_s": batch.timings.get("rollout_reset_time_s", ""),
                    "rollout_stack_obs_time_s": batch.timings.get("rollout_stack_obs_time_s", ""),
                    "rollout_model_action_time_s": batch.timings.get("rollout_model_action_time_s", ""),
                    "rollout_env_step_time_s": batch.timings.get("rollout_env_step_time_s", ""),
                    "rollout_interaction_time_s": batch.timings.get("rollout_interaction_time_s", ""),
                    "rollout_total_time_s": batch.timings.get("rollout_total_time_s", ""),
                    "ppo_update_time_s": ppo_update_time_s,
                    "eval_wall_time_s": eval_wall_time_s,
                    "epoch_wall_time_s": epoch_wall_time_s,
                    "train_feasible_rate": train_summary.get("train_feasible_rate", ""),
                    "train_avg_best_objective_distance_km": train_summary.get("train_avg_best_objective_distance_km", ""),
                    "train_avg_vehicle_count": train_summary.get("train_avg_vehicle_count", ""),
                    "train_avg_served_customers": train_summary.get("train_avg_served_customers", ""),
                    "eval_avg_objective_distance_km": eval_row.get("eval_avg_objective_distance_km", ""),
                    "eval_avg_vehicle_count": eval_row.get("eval_avg_vehicle_count", ""),
                    "eval_feasible_rate": eval_row.get("eval_feasible_rate", ""),
                    "eval_avg_runtime_s": eval_row.get("eval_avg_runtime_s", ""),
                    "eval_num_instances": eval_row.get("eval_num_instances", ""),
                    "eval_n_traj": eval_row.get("eval_n_traj", ""),
                    "eval_batch_size": eval_row.get("eval_batch_size", ""),
                    "eval_num_batches": eval_row.get("eval_num_batches", ""),
                    "eval_decode_mode": eval_row.get("eval_decode_mode", ""),
                    "eval_info_level": eval_row.get("eval_info_level", ""),
                    "eval_save_routes": eval_row.get("eval_save_routes", ""),
                    "eval_status": eval_row.get("eval_status", ""),
                }
            )
            f.flush()
            if epoch % checkpoint_interval == 0 or epoch == epochs:
                save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt", agent, optimizer, cfg, epoch, seed)
    save_checkpoint(ckpt_dir / "checkpoint_final.pt", agent, optimizer, cfg, epochs, seed)
    close_pool = getattr(pool, "close", None)
    if callable(close_pool):
        close_pool(terminate=True)
    return ckpt_dir / "checkpoint_final.pt"
