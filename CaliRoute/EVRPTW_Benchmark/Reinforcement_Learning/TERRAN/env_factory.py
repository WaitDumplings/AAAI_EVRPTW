from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from gymnasium import Wrapper

from evrptw_core.io import load_instance
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv, EVRPTWVectorEnvFast
from .pbrs import PotentialRewardConfig, PotentialRewardWrapper


class OnlineInstanceResetWrapper(Wrapper):
    """Refresh the wrapped EVRPTW env with a new sampled instance at reset."""

    def __init__(self, env, instance_sampler: Callable[[], Any]):
        super().__init__(env)
        self.instance_sampler = instance_sampler

    def reset(self, **kwargs: Any):
        options = dict(kwargs.pop("options", {}) or {})
        options["instance"] = self.instance_sampler()
        return self.env.reset(options=options, **kwargs)


def make_terran_env(
    instance_path: str | Path | None = None,
    instance: Any | None = None,
    instance_sampler: Callable[[], Any] | None = None,
    n_traj: int = 50,
    reward_mode: str = "distance",
    pbrs_config: PotentialRewardConfig | None = None,
    **env_kwargs: Any,
):
    """Create the shared EVRPTW env with optional online sampling and PBRS."""
    if instance is None and instance_sampler is None:
        if instance_path is None:
            raise ValueError("Provide instance, instance_path, or instance_sampler.")
        instance = load_instance(instance_path)
    if instance is None and instance_sampler is not None:
        instance = instance_sampler()

    use_fast_env = bool(env_kwargs.pop("use_fast_env", True))
    info_level = str(env_kwargs.pop("info_level", "full"))
    use_jit_mask = bool(env_kwargs.pop("use_jit_mask", True))
    if use_fast_env:
        env = EVRPTWVectorEnvFast(
            instance=instance,
            n_traj=n_traj,
            reward_mode=reward_mode,
            info_level=info_level,
            use_jit_mask=use_jit_mask,
            **env_kwargs,
        )
    else:
        env = EVRPTWVectorEnv(instance=instance, n_traj=n_traj, reward_mode=reward_mode, **env_kwargs)
    if instance_sampler is not None:
        env = OnlineInstanceResetWrapper(env, instance_sampler)
    if pbrs_config is not None and (
        pbrs_config.use_customer_pbrs
        or pbrs_config.use_repair_distance_pbrs
        or pbrs_config.use_feasible_ratio_pbrs
        or pbrs_config.use_terminal_heuristic
    ):
        env = PotentialRewardWrapper(env, pbrs_config)
    return env


__all__ = ["make_terran_env", "OnlineInstanceResetWrapper"]
