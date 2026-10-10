"""Portable, versioned candidates without changing the existing training engine.

``aaai_graph_v1`` reproduces the recorded Graph/explore training configuration
for VRPTW100 and EVRPTW100. It is a candidate, not a definition of experiment E1.
The CURRENT encoder is the matching modern control, not historical original.
This module only resolves configuration; it does not open datasets, write files,
load learned state, select GPUs, or start training.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from caliroute.input_normalization import input_normalization_signature
from caliroute.recipe_components import describe_components


CODE_ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAME = "aaai_graph_v1"
RECIPE_PATH = CODE_ROOT / "configs" / "recipes" / f"{RECIPE_NAME}.yaml"
HARDWARE_PATHS = {
    name: CODE_ROOT / "configs" / "hardware" / f"aaai_{name}.yaml"
    for name in ("rtx48_single", "2080ti_dual")
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mapping(path):
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError(f"Unsupported recipe/profile schema: {path}")
    return result


def _source_record(path):
    return {"path": str(path.relative_to(CODE_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_recipe_config(*, problem, customers=100, encoder="graph", seed=3011,
                        epochs=1500, data_root, output_dir, run_name,
                        world_size=1, global_batch=None, ppo_chunk_size=None,
                        expert_chunk_size=None, eval_batch_size=None,
                        hardware="rtx48_single"):
    """Resolve a complete trainer config while keeping research budgets explicit.

    ``data_root`` is the AAAI_Dataset directory containing ``dataset/``.
    Hardware names constrain rank count; they never silently change global batch,
    trajectory count, precision, LR, optimizer passes, or minibatches. Explicit
    batch overrides are research variants and are recorded as such. Batches above
    64 need a separately specified replay budget and are deliberately rejected.
    Every launch still needs a full-allocation preflight on its actual hardware.
    """
    recipe = _mapping(RECIPE_PATH)
    if problem not in recipe["tasks"]:
        raise ValueError("aaai_graph_v1 supports only vrptw and evrptw; CVRP needs separate validation")
    if isinstance(customers, bool) or not isinstance(customers, int) or customers != 100:
        raise ValueError("aaai_graph_v1 supports only Cus100; Cus15/Cus50 need separate validation")
    if encoder not in recipe["supported_encoders"]:
        raise ValueError("encoder must be graph or current (the modern control, not original)")
    if hardware not in HARDWARE_PATHS:
        raise ValueError(f"hardware must be one of {', '.join(HARDWARE_PATHS)}")
    _positive_integer(world_size, "world_size")
    profile = _mapping(HARDWARE_PATHS[hardware])
    if world_size != profile["world_size"]:
        raise ValueError(f"hardware={hardware} requires world_size={profile['world_size']}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    _positive_integer(epochs, "epochs")
    if not isinstance(run_name, str) or not run_name.strip() or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a nonempty filename without directory components")

    task = recipe["tasks"][problem]
    research = recipe["research_defaults"]
    execution = profile["execution"]
    values = {"global_batch": task["global_batch"] if global_batch is None else global_batch,
              "ppo_chunk_size": execution["ppo_chunk_size"] if ppo_chunk_size is None else ppo_chunk_size,
              "expert_chunk_size": execution["expert_chunk_size"] if expert_chunk_size is None else expert_chunk_size,
              "eval_batch_size": execution["eval_batch_size"] if eval_batch_size is None else eval_batch_size}
    for key, value in values.items():
        _positive_integer(value, key)
    batch, chunk, expert_chunk, eval_batch = (values[key] for key in
        ("global_batch", "ppo_chunk_size", "expert_chunk_size", "eval_batch_size"))
    cfg = copy.deepcopy(recipe["config"])
    minibatches = cfg["training"]["num_minibatches"]
    if batch % (world_size * minibatches):
        raise ValueError(f"global_batch must be divisible by world_size * num_minibatches ({world_size * minibatches})")
    if batch > research["maximum_global_batch_without_replay_cap_change"]:
        raise ValueError("global_batch above 64 requires a separate global replay-candidate budget; this is a recipe budget limit, not a model limit")
    if chunk > task["rollout_steps"]:
        raise ValueError(f"ppo_chunk_size must not exceed rollout_steps={task['rollout_steps']}")
    if eval_batch > task["validation_instances"]:
        raise ValueError("eval_batch_size must not exceed the 1000 validation instances")
    for name in ("global_exploration_instances", "global_policy_replay_max_new_routes"):
        if research[name] % world_size:
            raise ValueError(f"{name} must be divisible by world_size")

    data_root = Path(data_root).expanduser().resolve()
    if data_root.name == "dataset":
        raise ValueError("data_root must be AAAI_Dataset, not its dataset child")
    output = Path(output_dir).expanduser().resolve()
    train = data_root / "dataset" / problem / "train" / f"Cus{customers}"
    val = data_root / "dataset" / problem / "val" / f"Cus{customers}"
    cfg.update(run_name=run_name, dataset_name=task["dataset_name"])
    cfg["data"].update(problem_type=problem, num_customers=customers,
        num_charging_stations=task["physical_charging_stations"], train_dataset_path=str(train))
    cfg["training"].update(epochs=epochs, post_init_seed=seed,
        num_envs_per_gpu=batch // world_size, ppo_step_chunk_size=chunk,
        rollout_steps=task["rollout_steps"],
        require_complete_feasible_rollouts=task["require_complete_feasible_rollouts"],
        monitor_output_dir=str(output / "monitoring"))
    cfg["offline"].update(expert_dataset_path=str(train),
        expert_solution_path=str(train / "expert_solutions.csv"),
        sl_expert_logprob_chunk_size=expert_chunk,
        exploration_instances=research["global_exploration_instances"] // world_size,
        policy_replay_max_new_routes=research["global_policy_replay_max_new_routes"] // world_size)
    cfg["advantage"]["sl_expert_logprob_chunk_size"] = expert_chunk
    cfg["evaluation"].update(eval_path=str(val), gurobi_summary_path=str(val / "gurobi_summary.csv"),
        eval_output_dir=str(output / "evaluations"), eval_max_steps=task["rollout_steps"],
        eval_batch_size=eval_batch, eval_seed=17000000 + seed)
    if encoder == "current":
        cfg["model"].update(use_joint_graph_encoder=False, use_edge_relation_encoder=True)
        cfg["model"].pop("joint_graph_edge_dim")
        cfg["model"].pop("joint_graph_dropout")

    model, training, offline, evaluation = (cfg[key] for key in ("model", "training", "offline", "evaluation"))
    integration_keys = ("use_typed_static_fusion", "use_edge_relation_encoder", "use_edge_state_updates",
        "use_edge_value_messages", "use_resource_decoder", "edge_relation_dim", "decoder_observation_mode",
        "agda_physical_candidate_features", "agda_smooth_distance_features", "use_joint_graph_encoder")
    integration = {"schema": "physical_model_integration_v1",
        **{key: model[key] for key in integration_keys},
        "joint_graph_edge_dim": model.get("joint_graph_edge_dim"),
        "active_edge_state_dim": model["joint_graph_edge_dim"] if encoder == "graph" else model["edge_relation_dim"]}
    normalization = input_normalization_signature(cfg)
    normalization.update(schema="routing_input_normalization_v1",
        use_physical_input_context=model["use_physical_input_context"],
        physical_input_context_hidden_dim=model["physical_input_context_hidden_dim"])
    overrides = []
    for parameter, reference, used in (
        ("seed", research["seed"], seed), ("epochs", research["epochs"], epochs),
        ("global_batch", task["global_batch"], batch)):
        if reference != used:
            overrides.append({"parameter": parameter, "reference": reference, "used": used})
    hardware_overrides = [dict(parameter=key, profile=execution[key], used=values[key])
        for key in execution if execution[key] != values[key]]
    protocol = dict(
        recipe=RECIPE_NAME, recipe_status=recipe["status"], schema="aaai_recipe_protocol_v1",
        recipe_sources=[_source_record(RECIPE_PATH), _source_record(HARDWARE_PATHS[hardware])],
        reference_training_commit=recipe["reference_training_commit"],
        reference_provenance=recipe["reference_provenance"],
        reference_run=task[f"reference_{encoder}_run"],
        phase=f"{problem}{customers}_{RECIPE_NAME}", task=f"{problem}{customers}",
        arm=encoder, encoder_variant=encoder, implementation="explore", seed=seed, epochs=epochs,
        initialization_mode="scratch", ppo_warmup_epochs=0, source_init_checkpoint=None,
        source_init_epoch=None, source_initialization_experiment=None,
        initialization="Random model initialization, empty optimizer/archive/statistics; SL-PPO starts at epoch 1. No learned checkpoint or PPO/BC warmup.",
        world_size=world_size, hardware=hardware, require_preflight_health=True,
        hardware_validation=copy.deepcopy(profile["validation"]), hardware_overrides=hardware_overrides,
        research_overrides=overrides, pending_validation=copy.deepcopy(recipe["pending_validation"]),
        nonreference_global_batch=batch != task["global_batch"],
        eval_batch_changes_sampling_stream=eval_batch != execution["eval_batch_size"],
        eval_batch_caveat="Changing evaluation batch size may change the sampled trajectories despite using the same seed; use the same value for compared arms.",
        global_batch=batch, n_traj=training["n_traj"], num_minibatches=minibatches,
        ppo_update_epochs=training["ppo_update_epochs"], target_kl=training["target_kl"],
        learning_rate=training["learning_rate"], lr_schedule=training["lr_schedule"],
        ppo_chunk_size=chunk, expert_chunk_size=expert_chunk,
        attempted_optimizer_steps_per_epoch=training["ppo_update_epochs"] * minibatches,
        global_instances_per_rollout=batch, global_trajectories_per_rollout=batch * training["n_traj"],
        global_instances_per_optimizer_step=batch // minibatches,
        batch_controls=dict(instances=batch, per_rank_instances=batch // world_size,
            trajectories=training["n_traj"], minibatches=minibatches,
            ppo_passes=training["ppo_update_epochs"], rollout_steps=training["rollout_steps"]),
        eval_interval=evaluation["eval_interval"], eval_n_traj=evaluation["eval_n_traj"],
        eval_batch_size=eval_batch, train_instances=task["train_instances"],
        validation_instances=task["validation_instances"], selection_split="val", test_enabled=False,
        physical_charging_stations=task["physical_charging_stations"], charging_mode=cfg["env"]["charging_mode"],
        env_action_limit=cfg["env"]["max_steps_factor"] * (customers + 1 + task["physical_charging_stations"]),
        input_normalization_signature=normalization, model_integration=integration,
        architecture=dict(use_joint_graph_encoder=model["use_joint_graph_encoder"],
            joint_graph_edge_dim=model.get("joint_graph_edge_dim"),
            joint_graph_dropout=model.get("joint_graph_dropout"), training_bundle="explore"),
        extra_search_enabled=offline["branch_exploration_enabled"] or offline["exploration_enabled"],
        search_budget=dict(interval=offline["exploration_interval"],
            max_instances_per_rank=offline["exploration_instances"],
            trajectories_per_instance=offline["exploration_trajectories"],
            max_global_trajectories=research["global_exploration_instances"] * offline["exploration_trajectories"]),
        global_policy_replay_max_new_routes=research["global_policy_replay_max_new_routes"],
        global_policy_replay_candidate_budget=min(offline["policy_replay_max_candidates"],
            int((batch // world_size) * offline["policy_replay_fraction"])) * world_size,
        replay_schedule="The inherited archive weight warmup/ramp remains 25/75 epochs; it is not a PPO-only initialization stage.",
        distributed_execution=("single-process objective; one sampler and policy archive" if world_size == 1 else
            "mean of rank-local masked objectives; scaled gradient averaging at optimizer boundaries; rank-local samplers and policy archives"),
        distributed_caveat="Equal global budgets do not make independent rank samplers, archives and rank-local loss reductions bitwise equivalent to a single GPU.",
        comparison_scope="Graph versus CURRENT changes the static graph encoder and its edge interface; both use the same modern explore training algorithm. CURRENT is not historical original.",
        compute_caveat="Both encoders retain extra exploration rollouts; nominal PPO epochs are not a complete compute budget.",
        initial_evaluation_equivalence_group=None,
        input_units="Fixed distance unit 43.638668060302734 km; explicit road distance/time/energy matrices remain authoritative.",
        reward_failure_guard="The inherited 1000 km failed-termination penalty is an explicit guard, not a universal feasibility bound.",
        kl_caveat="Recorded replay-action KL is an estimator, not a hard post-update bound; target_kl is disabled.",
    )
    cfg["experiment_protocol"] = protocol
    protocol["resolved_components"] = describe_components(cfg)
    return cfg
