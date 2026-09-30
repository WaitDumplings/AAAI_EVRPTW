from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import merge_route_sequences


def stack_observations(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = observations[0].keys()
    return {key: np.stack([obs[key] for obs in observations], axis=0) for key in keys}


def tensor_from_array(value: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def sample_actions(agent, obs_batch: dict[str, np.ndarray], decode_mode: str, device: str | torch.device):
    logits_tuple = agent.backbone(obs_batch)
    logits = logits_tuple[0]
    dist = torch.distributions.Categorical(logits=logits)
    if decode_mode == "greedy":
        actions = torch.argmax(logits, dim=-1)
    elif decode_mode == "sample":
        actions = dist.sample()
    else:
        raise ValueError(f"Unknown decode_mode={decode_mode!r}")
    logprob = dist.log_prob(actions)
    entropy = dist.entropy()
    value = agent.critic((logits_tuple[0], logits_tuple[1])).squeeze(-1)
    return actions, logprob, entropy, value, logits


@dataclass
class RolloutBatch:
    observations: list[dict[str, np.ndarray]]
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    route_boundaries: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    entropies: torch.Tensor
    final_infos: list[dict[str, Any]]
    timings: dict[str, float]
    instance_ids: list[str] | None = None
    expert_actions: torch.Tensor | None = None
    expert_valid: torch.Tensor | None = None
    expert_advantages: torch.Tensor | None = None


def reset_envs(envs, seed: int | None = None):
    observations = []
    infos = []
    for idx, env in enumerate(envs):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed) + idx
        obs, info = env.reset(**kwargs)
        observations.append(obs)
        infos.append(info)
    return observations, infos


def current_instance_ids(envs) -> list[str]:
    out = []
    for env in envs:
        instance = getattr(env.unwrapped, "instance", None)
        out.append(str(getattr(instance, "instance_id", "")))
    return out


def route_boundaries_from_actions(
    actions: np.ndarray,
    valid: np.ndarray,
    current_route_customer_count: np.ndarray,
    num_customers: int,
) -> np.ndarray:
    """Flag depot actions that close a non-empty route, aligned to action/reward time."""
    action_arr = np.asarray(actions, dtype=np.int64)
    valid_arr = np.asarray(valid, dtype=bool)
    boundaries = (action_arr == 0) & valid_arr & (current_route_customer_count > 0)
    customer = (action_arr >= 1) & (action_arr <= int(num_customers)) & valid_arr
    current_route_customer_count[customer] += 1
    current_route_customer_count[boundaries] = 0
    return boundaries


def step_envs(envs, actions: np.ndarray):
    observations, rewards, dones, infos = [], [], [], []
    for env, action in zip(envs, actions):
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool))
        infos.append(info)
    return observations, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), infos


def collect_rollout(
    agent,
    envs,
    rollout_steps: int,
    decode_mode: str,
    device: str | torch.device,
    seed: int | None = None,
    profile_timing: bool = False,
    expert_provider=None,
) -> RolloutBatch:
    total_start = time.perf_counter()
    reset_start = time.perf_counter()
    observations, infos = reset_envs(envs, seed=seed)
    reset_time_s = time.perf_counter() - reset_start
    instance_ids = current_instance_ids(envs)
    done = np.zeros((len(envs), envs[0].unwrapped.n_traj), dtype=bool)
    obs_steps: list[dict[str, np.ndarray]] = []
    actions_steps = []
    logprob_steps = []
    reward_steps = []
    done_steps = []
    value_steps = []
    valid_steps = []
    entropy_steps = []
    expert_action_steps = []
    expert_valid_steps = []
    model_action_time_s = 0.0
    env_step_time_s = 0.0
    stack_obs_time_s = 0.0
    num_customers = int(getattr(envs[0].unwrapped, "num_customers", 0))
    current_route_customer_count = np.zeros((len(envs), envs[0].unwrapped.n_traj), dtype=np.int32)
    route_boundary_steps = []

    for _ in range(int(rollout_steps)):
        valid = ~done
        stack_start = time.perf_counter()
        obs_batch = stack_observations(observations)
        stack_obs_time_s += time.perf_counter() - stack_start
        if expert_provider is not None:
            expert_actions_np, expert_valid_np = expert_provider.actions_for_batch(
                instance_ids,
                len(obs_steps),
                obs_batch["action_mask"],
                done,
            )
            expert_action_steps.append(tensor_from_array(expert_actions_np, device).long())
            expert_valid_steps.append(tensor_from_array(expert_valid_np, device).bool())
        if profile_timing:
            _sync_cuda(device)
        model_start = time.perf_counter()
        with torch.no_grad():
            actions, logprob, entropy, value, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
        if profile_timing:
            _sync_cuda(device)
        model_action_time_s += time.perf_counter() - model_start
        action_np = actions.detach().cpu().numpy().astype(np.int64)
        route_boundary_np = route_boundaries_from_actions(
            action_np,
            valid,
            current_route_customer_count,
            num_customers=num_customers,
        )
        env_start = time.perf_counter()
        next_observations, reward_np, step_done, infos = step_envs(envs, action_np)
        env_step_time_s += time.perf_counter() - env_start

        obs_steps.append(obs_batch)
        actions_steps.append(actions.detach())
        logprob_steps.append(logprob.detach())
        entropy_steps.append(entropy.detach())
        reward_steps.append(tensor_from_array(reward_np, device).float())
        done_steps.append(tensor_from_array(step_done, device).bool())
        route_boundary_steps.append(tensor_from_array(route_boundary_np, device).bool())
        value_steps.append(value.detach())
        valid_steps.append(tensor_from_array(valid, device).bool())

        observations = next_observations
        done = done | step_done
        if done.all():
            break

    total_time_s = time.perf_counter() - total_start
    expert_actions = torch.stack(expert_action_steps, dim=0) if expert_action_steps else None
    expert_valid = torch.stack(expert_valid_steps, dim=0) if expert_valid_steps else None
    expert_advantages = None
    if expert_provider is not None and expert_actions is not None:
        expert_advantages = expert_provider.advantages_for_final_infos(
            instance_ids,
            infos,
            len(obs_steps),
            device,
        )
    return RolloutBatch(
        observations=obs_steps,
        actions=torch.stack(actions_steps, dim=0),
        old_logprobs=torch.stack(logprob_steps, dim=0),
        rewards=torch.stack(reward_steps, dim=0),
        dones=torch.stack(done_steps, dim=0),
        route_boundaries=torch.stack(route_boundary_steps, dim=0),
        values=torch.stack(value_steps, dim=0),
        valid=torch.stack(valid_steps, dim=0),
        entropies=torch.stack(entropy_steps, dim=0),
        final_infos=infos,
        timings={
            "rollout_total_time_s": float(total_time_s),
            "rollout_reset_time_s": float(reset_time_s),
            "rollout_stack_obs_time_s": float(stack_obs_time_s),
            "rollout_model_action_time_s": float(model_action_time_s),
            "rollout_env_step_time_s": float(env_step_time_s),
            "rollout_interaction_time_s": float(model_action_time_s + env_step_time_s),
        },
        instance_ids=instance_ids,
        expert_actions=expert_actions,
        expert_valid=expert_valid,
        expert_advantages=expert_advantages,
    )


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.size(0))):
        running = rewards[step] + float(gamma) * running * (~dones[step]).float()
        returns[step] = running
    return returns


def select_best_trajectory(info: dict[str, Any], include_routes: bool = True) -> dict[str, Any]:
    success = np.asarray(info["success"], dtype=bool)
    objective = np.asarray(info["objective_distance_km"], dtype=np.float64)
    served = np.asarray(info["served_customers"], dtype=np.int32)
    if np.any(success):
        candidates = np.where(success)[0]
        selected = int(candidates[np.argmin(objective[candidates])])
        feasible = True
    else:
        max_served = int(served.max()) if served.size else 0
        candidates = np.where(served == max_served)[0]
        selected = int(candidates[np.argmin(objective[candidates])]) if candidates.size else 0
        feasible = False
    row = {
        "selected_traj_idx": selected,
        "feasible": feasible,
        "objective_distance_km": float(objective[selected]),
        "vehicle_count": int(np.asarray(info["vehicle_count"])[selected]),
        "served_customers": int(served[selected]),
    }
    if include_routes and "routes" in info:
        routes = info["routes"][selected]
        route_sequence = merge_route_sequences(routes)
        row["route_sequence_json"] = json.dumps(route_sequence)
        row["routes_json"] = json.dumps(routes)
    return row


def rollout_single_instance(
    agent,
    env,
    decode_mode: str,
    max_steps: int,
    device: str | torch.device,
    seed: int | None = None,
    include_routes: bool = True,
):
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    done = np.zeros(env.unwrapped.n_traj, dtype=bool)
    start = time.perf_counter()
    for _ in range(int(max_steps)):
        obs_batch = stack_observations([obs])
        with torch.no_grad():
            actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
        obs, reward, terminated, truncated, info = env.step(actions.squeeze(0).detach().cpu().numpy().astype(np.int64))
        done = done | np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        if done.all():
            break
    elapsed = time.perf_counter() - start
    row = select_best_trajectory(info, include_routes=include_routes)
    row["runtime_s"] = float(elapsed)
    return row


def rollout_eval_batch(
    agent,
    envs,
    decode_mode: str,
    max_steps: int,
    device: str | torch.device,
    seed: int | None = None,
    include_routes: bool = False,
):
    if not envs:
        return []
    observations, infos = reset_envs(envs, seed=seed)
    n_traj = int(envs[0].unwrapped.n_traj)
    done = np.zeros((len(envs), n_traj), dtype=bool)
    start = time.perf_counter()
    for _ in range(int(max_steps)):
        obs_batch = stack_observations(observations)
        with torch.no_grad():
            actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
        action_np = actions.detach().cpu().numpy().astype(np.int64)
        observations, _, step_done, infos = step_envs(envs, action_np)
        done = done | step_done
        if done.all():
            break
    elapsed = time.perf_counter() - start
    per_instance_runtime = float(elapsed) / max(len(envs), 1)
    rows: list[dict[str, Any]] = []
    for info in infos:
        row = select_best_trajectory(info, include_routes=include_routes)
        row["runtime_s"] = per_instance_runtime
        row["batch_runtime_s"] = float(elapsed)
        rows.append(row)
    return rows
