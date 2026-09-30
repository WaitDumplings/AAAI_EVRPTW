from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import Wrapper
import numpy as np


@dataclass(frozen=True)
class PotentialRewardConfig:
    """PBRS controls for TERRAN-style RL training."""

    use_customer_pbrs: bool = False
    use_repair_distance_pbrs: bool = False
    use_feasible_ratio_pbrs: bool = False
    use_terminal_heuristic: bool = False
    customer_pbrs_mode: str = "progress"  # strict PBRS: gamma * Phi(s_next) - Phi(s)
    gamma: float = 0.99
    alpha: float = 2.0
    beta: float = 0.5
    customer_pbrs_coef: float = 1.0
    customer_progress_budget: float = 0.5
    customer_progress_mix: float = 0.5
    repair_progress_coef: float = 0.5
    feasible_ratio_coef: float = 0.0
    pbrs_clip: float | None = None
    success_bonus: float = 0.1
    failure_penalty: float = 0.5

    def __post_init__(self) -> None:
        mode = self.customer_pbrs_mode.lower()
        if mode in {"serve", "served", "ratio_progress"}:
            mode = "progress"
        if mode not in {"progress", "direct_progress"}:
            raise ValueError("customer_pbrs_mode must be progress or direct_progress")
        object.__setattr__(self, "customer_pbrs_mode", mode)


class PotentialRewardWrapper(Wrapper):
    """Gymnasium wrapper for optional TERRAN PBRS.

    The repair-distance heuristic matches the prior DRL implementation: each
    unserved customer contributes the distance of a one-customer depot-customer-
    depot repair route. Progress is positive when this remaining repair workload
    decreases.
    """

    def __init__(self, env, config: PotentialRewardConfig | None = None, **kwargs: Any) -> None:
        super().__init__(env)
        self.config = config or PotentialRewardConfig(**kwargs)
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_info: dict[str, Any] | None = None
        self._last_finished: np.ndarray | None = None
        self._single_customer_repair_dist: np.ndarray | None = None
        self._node_to_depot_repair_dist: np.ndarray | None = None
        self._total_customer_repair_dist: float = 1.0
        self.reward_scale = 1.0

    def set_reward_scale(self, reward_scale: float) -> None:
        self.reward_scale = max(float(reward_scale), 0.0)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prepare_repair_distance_cache()
        self._last_obs = obs
        self._last_info = info
        self._last_finished = np.zeros_like(info["served_customers"], dtype=bool)
        zero = np.zeros_like(info["served_customers"], dtype=np.float32)
        return obs, self._augment_info(info, zero)

    def step(self, action):
        if self._last_obs is None or self._last_info is None or self._last_finished is None:
            raise RuntimeError("PotentialRewardWrapper.step called before reset.")
        prev_obs = self._last_obs
        prev_info = self._last_info
        prev_finished = self._last_finished.copy()
        prev_repair = self._remaining_repair_ratio()

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        customer = self._customer_pbrs(prev_info, info)
        repair = self._repair_distance_pbrs(prev_repair)
        feasible = self._feasible_ratio_pbrs(prev_obs, obs)
        terminal = self._terminal_heuristic(info, terminated, truncated, prev_finished)
        scale = float(self.reward_scale)
        customer = (scale * customer).astype(np.float32)
        repair = (scale * repair).astype(np.float32)
        feasible = (scale * feasible).astype(np.float32)
        terminal = (scale * terminal).astype(np.float32)
        shaped = base_reward.astype(np.float32) + customer + repair + feasible + terminal

        self._last_obs = obs
        self._last_info = info
        self._last_finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        out_info = dict(info)
        out_info["reward_components"] = {
            "base": base_reward.astype(np.float32).copy(),
            "pbrs_customer": customer.copy(),
            "pbrs_repair_distance": repair.copy(),
            "pbrs_feasible_ratio": feasible.copy(),
            "terminal_heuristic": terminal.copy(),
            "shaped": shaped.copy(),
            "pbrs_scale": np.full_like(shaped, scale, dtype=np.float32),
        }
        return obs, shaped, terminated, truncated, out_info

    def _prepare_repair_distance_cache(self) -> None:
        env = self.unwrapped
        distance = np.asarray(env.distance_km, dtype=np.float32)
        n = int(env.num_customers)
        customer_nodes = np.arange(1, 1 + n, dtype=int)
        single = distance[0, customer_nodes] + distance[customer_nodes, 0]
        self._single_customer_repair_dist = single.astype(np.float32)
        self._node_to_depot_repair_dist = distance[:, 0].astype(np.float32)
        self._total_customer_repair_dist = float(max(float(single.sum()), 1e-6))

    def _remaining_repair_ratio(self) -> np.ndarray:
        env = self.unwrapped
        if self._single_customer_repair_dist is None or self._node_to_depot_repair_dist is None:
            self._prepare_repair_distance_cache()
        assert self._single_customer_repair_dist is not None
        assert self._node_to_depot_repair_dist is not None
        unvisited = (~env.visited[:, 1 : 1 + env.num_customers]).astype(np.float32)
        remaining_customer = unvisited @ self._single_customer_repair_dist
        current_to_depot = self._node_to_depot_repair_dist[env.last]
        return np.maximum((remaining_customer + current_to_depot) / self._total_customer_repair_dist, 0.0).astype(np.float32)

    def _repair_distance_pbrs(self, prev_repair_ratio: np.ndarray) -> np.ndarray:
        cfg = self.config
        reward = np.zeros_like(prev_repair_ratio, dtype=np.float32)
        if not cfg.use_repair_distance_pbrs or cfg.repair_progress_coef == 0.0:
            return reward
        post_repair_ratio = self._remaining_repair_ratio()
        prev_phi = 1.0 - prev_repair_ratio
        post_phi = 1.0 - post_repair_ratio
        reward = float(cfg.repair_progress_coef) * (float(cfg.gamma) * post_phi - prev_phi)
        return self._clip(reward.astype(np.float32))

    def _customer_pbrs(self, prev_info: dict[str, Any], next_info: dict[str, Any]) -> np.ndarray:
        cfg = self.config
        served_prev = np.asarray(prev_info["served_customers"], dtype=np.float32)
        served_next = np.asarray(next_info["served_customers"], dtype=np.float32)
        reward = np.zeros_like(served_next, dtype=np.float32)
        if not cfg.use_customer_pbrs or cfg.customer_pbrs_coef == 0.0:
            return reward
        n = max(float(getattr(self.unwrapped, "num_customers", 1)), 1.0)
        beta = max(float(cfg.beta), 1e-8)
        prev_phi = self._direct_progress_potential(served_prev, n, beta)
        next_phi = self._direct_progress_potential(served_next, n, beta)
        # Scale-invariant progress budget. The linear term prevents early-customer
        # signal from vanishing too aggressively at large scales, while the
        # nonlinear term still gives larger marginal reward near completion.
        served_delta = np.maximum(served_next - served_prev, 0.0) / n
        nonlinear_delta = next_phi - prev_phi
        mix = float(np.clip(cfg.customer_progress_mix, 0.0, 1.0))
        progress_delta = (1.0 - mix) * served_delta + mix * nonlinear_delta
        progress_budget = float(cfg.customer_pbrs_coef) * float(cfg.customer_progress_budget)
        if cfg.customer_pbrs_mode == "progress":
            prev_ratio = (served_prev / n).clip(0.0, 1.0)
            prev_progress = (1.0 - mix) * prev_ratio + mix * prev_phi
            next_ratio = (served_next / n).clip(0.0, 1.0)
            next_progress = (1.0 - mix) * next_ratio + mix * next_phi
            reward = progress_budget * (float(cfg.gamma) * next_progress - prev_progress)
        else:
            reward = progress_budget * progress_delta
        return reward.astype(np.float32)

    def _feasible_ratio_pbrs(self, prev_obs: dict[str, np.ndarray], next_obs: dict[str, np.ndarray]) -> np.ndarray:
        cfg = self.config
        reward = np.zeros(prev_obs["action_mask"].shape[0], dtype=np.float32)
        if not cfg.use_feasible_ratio_pbrs or cfg.feasible_ratio_coef == 0.0:
            return reward
        prev_phi = self._feasible_customer_ratio(prev_obs)
        next_phi = self._feasible_customer_ratio(next_obs)
        reward = float(cfg.feasible_ratio_coef) * (float(cfg.gamma) * next_phi - prev_phi)
        return self._clip(reward.astype(np.float32))

    def _terminal_heuristic(self, next_info: dict[str, Any], terminated: np.ndarray, truncated: np.ndarray, prev_finished: np.ndarray) -> np.ndarray:
        cfg = self.config
        reward = np.zeros_like(np.asarray(next_info["served_customers"], dtype=np.float32))
        if not cfg.use_terminal_heuristic:
            return reward
        now_finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        newly_finished = now_finished & (~prev_finished)
        if not np.any(newly_finished):
            return reward
        success = np.asarray(next_info["success"], dtype=bool)
        served_ratio = np.asarray(next_info["served_customers"], dtype=np.float32) / max(float(getattr(self.unwrapped, "num_customers", 1)), 1.0)
        reward[newly_finished & success] += float(cfg.success_bonus)
        fail = newly_finished & (~success)
        reward[fail] -= float(cfg.failure_penalty) * (1.0 - served_ratio[fail])
        return reward.astype(np.float32)

    @staticmethod
    def _direct_progress_potential(served: np.ndarray, n: float, beta: float) -> np.ndarray:
        served_ratio = (served / n).clip(0.0, 1.0)
        return 1.0 - (1.0 - served_ratio).clip(0.0, 1.0) ** beta

    def _feasible_customer_ratio(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        n = int(getattr(self.unwrapped, "num_customers", 0))
        if n <= 0:
            return np.zeros(mask.shape[0], dtype=np.float32)
        return mask[:, 1 : 1 + n].sum(axis=1).astype(np.float32) / float(n)

    def _clip(self, reward: np.ndarray) -> np.ndarray:
        clip_value = self.config.pbrs_clip
        if clip_value is None or clip_value <= 0:
            return reward.astype(np.float32)
        return np.clip(reward, -float(clip_value), float(clip_value)).astype(np.float32)

    @staticmethod
    def _augment_info(info: dict[str, Any], reward: np.ndarray) -> dict[str, Any]:
        info = dict(info)
        info["reward_components"] = {
            "base": reward.copy(),
            "pbrs_customer": reward.copy(),
            "pbrs_repair_distance": reward.copy(),
            "pbrs_feasible_ratio": reward.copy(),
            "terminal_heuristic": reward.copy(),
            "shaped": reward.copy(),
            "pbrs_scale": np.ones_like(reward, dtype=np.float32),
        }
        return info


__all__ = ["PotentialRewardConfig", "PotentialRewardWrapper"]
