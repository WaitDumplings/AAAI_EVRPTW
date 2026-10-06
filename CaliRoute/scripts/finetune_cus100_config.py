"""Build the CVRP100/VRPTW100 PPO-init and validation-only finetuning protocol.

The 40-epoch comparisons are constant-schedule local probes. They are not the
first 40 epochs of the long cosine schedule, and their checkpoints are not used
to initialize the long runs. Every SLPPO arm starts from its task's shared PPO
initialization with an unchanged optimized_v2 model architecture.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
for _directory in (CODE_ROOT, Path(__file__).resolve().parent):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from caliroute.methods import method_preset
from caliroute.optimization import apply_optimization_profile
from offline2online.training_schedule import schedule_for_epoch
from run_slppo_comparison import build_configs


def build_config(
    *,
    problem: str,
    phase: str,
    run_name: str,
    output_dir: str | Path,
    data_root: str | Path,
    init_checkpoint: str | Path | None = None,
    seed: int = 3009,
) -> dict[str, Any]:
    """Return a config without creating files or reading datasets/checkpoints.

    ``output_dir`` is the directory for this individual phase; its evaluation
    and monitoring outputs are isolated there. Checkpoint existence and model
    compatibility are checked when the launcher loads it, allowing configs to
    be prepared before the shared PPO initialization has finished.
    """
    if problem not in {"cvrp", "vrptw"}:
        raise ValueError("problem must be cvrp or vrptw")
    if phase not in {"ppo_init", "control", "candidate", "long_control", "long_candidate"}:
        raise ValueError("unsupported Cus100 finetuning phase")
    if not isinstance(run_name, str) or not run_name.strip() or Path(run_name).name != run_name:
        raise ValueError("run_name must be a nonempty directory name")
    root = Path(data_root).expanduser().resolve()
    if {"test", "test_release"}.intersection(part.lower() for part in root.parts):
        raise ValueError("data_root must contain train and val datasets, not frozen test data")
    output = Path(output_dir).expanduser().resolve()
    is_init = phase == "ppo_init"
    is_long = phase.startswith("long_")
    is_candidate = phase in {"candidate", "long_candidate"}
    if is_init and init_checkpoint is not None:
        raise ValueError("ppo_init starts from random weights; omit init_checkpoint")
    if not is_init and (init_checkpoint is None or not str(init_checkpoint).strip()):
        raise ValueError("SLPPO phases require the shared PPO init_checkpoint")

    epochs = 100 if is_init else 1000 if is_long else 40
    # The shared SLPPO builder requires a path. PPO initialization replaces its
    # entire offline section below, so this temporary value never enters the
    # returned PPO config or triggers a checkpoint load.
    args = Namespace(
        problem=problem, customers=100, data_root=root,
        init_checkpoint=init_checkpoint if init_checkpoint is not None else output,
        epochs=epochs, num_envs=32, n_traj=50, learning_rate=5e-5,
        eval_interval=20, eval_batch_size=32, seed=int(seed),
        eval_limit=None, expert_limit=None,
    )
    cfg = apply_optimization_profile(build_configs(args, output)["optimized"], "optimized_v2")
    cfg["run_name"] = run_name
    cfg["training"].update({
        "epochs": epochs,
        "rollout_steps": 201,
        "ppo_step_chunk_size": 8,
        "ppo_update_epochs": method_preset("ppo").ppo_update_epochs if is_init else 3 if is_candidate else 4,
        "checkpoint_interval": 50 if is_init or is_long else 40,
        "latest_checkpoint_interval": 5,
        "learning_rate": 5e-5,
        "lr_schedule": "warmup_cosine" if is_init or is_long else "constant",
        "lr_warmup_epochs": 10 if is_init else 20 if is_long else 0,
        "lr_min": 1e-5 if is_init or is_long else 5e-5,
        "ent_coef": .01,
        "entropy_initial_coef": .01,
        "entropy_final_coef": .002 if is_init or is_long else .01,
        "monitor_interval": 10,
        "monitor_gradient_components": True,
        "monitor_output_dir": str(output / "monitoring"),
        "monitor_target_kl": .02,
        "amp_init_scale": 4096.,
    })
    cfg["evaluation"].update({
        "eval_max_steps": 201,
        "eval_output_dir": str(output / "evaluations"),
    })
    if is_init:
        # Changing method alone is insufficient: the trainer independently
        # checks reference/candidate flags and may otherwise load expert data.
        cfg["offline"] = method_preset("ppo").offline_config()
        cfg["offline"].update({
            "init_checkpoint_strict": True,
            "sl_coef": 0.,
            "policy_replay_enabled": False,
            "use_priority_sampler": False,
            "share_expert_static_observations": False,
        })
        cfg["advantage"] = {
            "use_group_advantage": False,
            "use_reference_advantage": False,
            "use_expert_solution_level": False,
        }
        cfg["training"].update({
            "share_ppo_sl_forward": False,
            "cache_expert_route_encoding": False,
        })
    else:
        cfg["offline"].update({
            "init_checkpoint_path": str(Path(init_checkpoint).expanduser().resolve()),
            "init_checkpoint_strict": True,
            "sl_coef": .35 if is_candidate else .5,
        })
    cfg["experiment_protocol"] = {
        "phase": phase,
        "selection_split": "val",
        "seed": int(seed),
        "world_size": 2,
        "global_instances_per_rollout": 64,
        "global_trajectories_per_rollout": 3200,
        "global_instances_per_optimizer_step": 16,
        "initialization": "random" if is_init else "shared_task_PPO_init_weights_only",
        "schedule_scope": (
            "full_horizon_warmup_cosine" if is_init or is_long else
            "40_epoch_local_probe_with_constant_LR_and_entropy; not_a_long_schedule_prefix"
        ),
    }
    # Exercise the actual schedule parser, including its horizon constraints.
    schedule_for_epoch(cfg, 1)
    schedule_for_epoch(cfg, epochs)
    return cfg
