from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


METHOD_ALIASES = {
    "ppo": "ppo",
    "online": "ppo",
    "baseline": "ppo",
    "slppo": "slppo",
    "sl_ppo": "slppo",
    "solution_level_ppo": "slppo",
    "solution-level-ppo": "slppo",
    "solution_ppo": "slppo",
    "solution-ppo": "slppo",
    "awbc": "awbc",
    "awbc_ppo": "awbc",
    "awbc-ppo": "awbc",
    "dapg": "dapg",
}


@dataclass(frozen=True)
class MethodPreset:
    """Training-method defaults layered on top of the shared PPO backbone."""

    name: str
    trainer_method: str
    paper_role: str
    ppo_update_epochs: int
    requires_expert: bool
    offline: dict[str, Any] = field(default_factory=dict)
    advantage: dict[str, Any] = field(default_factory=dict)

    def offline_config(self) -> dict[str, Any]:
        out = deepcopy(self.offline)
        out["method"] = self.trainer_method
        return out

    def advantage_config(self) -> dict[str, Any]:
        return deepcopy(self.advantage)


SLPPO_ADVANTAGE_DEFAULTS: dict[str, Any] = {
    "use_group_advantage": True,
    "group_adv_coef": 1.0,
    "group_adv_clip": 3.0,
    "group_adv_std_floor": 5.0,
    "group_infeasible_penalty": 10.0,
    "sl_include_reference_in_group_stats": True,
    "sl_use_memory_incumbent": True,
    "use_reference_advantage": True,
    "reference_adv_coef": 0.50,
    "reference_adv_rho": 1.0,
    "reference_adv_clip": 3.0,
    "reference_success_only": True,
    "reference_advantage_mode": "absolute",
    "use_reference_soft_gate": False,
    "use_reference_memory_gate": False,
    "use_expert_solution_level": True,
    "sl_expert_candidate_weight": 0.60,
    "sl_candidate_clip": 2.0,
    "sl_candidate_std_floor": 5.0,
    "sl_candidate_gap_baseline": "mean",
    "sl_candidate_gap_scale_coef": 1.0,
    "sl_candidate_gap_floor_ratio": 0.01,
    "sl_candidate_quality_gate_eta": 0.05,
    "sl_candidate_margin": 0.005,
    "sl_candidate_gate_eta": 0.05,
    "sl_candidate_use_current_incumbent_gate": True,
    "sl_candidate_use_memory_incumbent_gate": True,
    "sl_use_expert_candidate": True,
}


METHOD_PRESETS: dict[str, MethodPreset] = {
    "ppo": MethodPreset(
        name="ppo",
        trainer_method="ppo",
        paper_role="online PPO backbone / comparison",
        ppo_update_epochs=3,
        requires_expert=False,
        offline={"init_checkpoint_strict": False},
    ),
    "slppo": MethodPreset(
        name="slppo",
        trainer_method="sl_ppo",
        paper_role="main contribution: solution-level supervised PPO",
        ppo_update_epochs=4,
        requires_expert=True,
        offline={
            "init_checkpoint_strict": False,
            "strict_replay": False,
            "sl_coef": 0.50,
            "sl_clip_coef": 0.20,
            "only_success_route_loss": True,
            "sl_expert_candidate_weight": 0.60,
            "sl_expert_logprob_chunk_size": 4096,
            "use_priority_sampler": True,
            "priority_selection_mode": "weighted",
            "priority_mix_rho": 0.50,
            "priority_alpha": 0.70,
        },
        advantage=SLPPO_ADVANTAGE_DEFAULTS,
    ),
    "awbc": MethodPreset(
        name="awbc",
        trainer_method="awbc",
        paper_role="offline comparison: advantage-weighted behavior cloning",
        ppo_update_epochs=3,
        requires_expert=True,
        offline={
            "init_checkpoint_strict": False,
            "strict_replay": False,
            "bc_batch_size": 512,
            "bc_updates_per_epoch": 16,
            "bc_coef": 0.10,
            "awbc_coef": 0.10,
            "awbc_eta": 0.10,
            "awbc_normalize": "eta_objective",
            "awbc_baseline": "mean_successful",
        },
    ),
    "dapg": MethodPreset(
        name="dapg",
        trainer_method="dapg",
        paper_role="offline comparison: demonstration-augmented policy gradient",
        ppo_update_epochs=3,
        requires_expert=True,
        offline={
            "init_checkpoint_strict": False,
            "strict_replay": False,
            "bc_warmup_epochs": 3,
            "bc_warmup_coef": 1.0,
            "bc_batch_size": 512,
            "bc_updates_per_epoch": 16,
            "bc_coef": 0.10,
            "bc_decay": 0.995,
            "min_bc_coef": 0.0,
        },
    ),
}


def canonical_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    try:
        return METHOD_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(METHOD_PRESETS))
        raise ValueError(f"Unsupported method {method!r}; choose one of: {valid}") from exc


def method_preset(method: str) -> MethodPreset:
    return METHOD_PRESETS[canonical_method(method)]
