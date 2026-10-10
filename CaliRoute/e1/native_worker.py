"""Process-isolated native E1 training, recovery and validation.

RADAR calls the frozen ACVRPTrainer._train_one_batch unchanged. RRNCO calls
RRNet.shared_step and its configure_optimizers unchanged; only Lightning's
logging sink is replaced because the outer loop records explicit exposure
budgets and checkpoints. Neither native objective is implemented here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import fcntl
import math
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from e1.native import atomic_json, file_hash, json_hash, verify_files
from e1.validator import routes_from_sequence
from e1.evaluation import select_candidates, summarize_rows


def now():
    return datetime.now(timezone.utc).isoformat()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be exactly resumed on a CPU-only worker")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def evaluation_rng(seed):
    before = rng_state()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: yield
    finally: restore_rng(before)


def load_data(run, split):
    path = run / "native_data" / f"{split}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != metadata["npz_sha256"]:
        raise ValueError(f"Changed native dataset: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, metadata


def instance_for(arrays, metadata, index):
    return SimpleNamespace(num_customers=arrays["demand"].shape[1],
        distance_matrix_km=arrays["distance_matrix"][index], demands_cm3=arrays["demand"][index],
        vehicle={"cargo_capacity_cm3": float(arrays["capacity"][index])},
        instance_id=metadata["instances"][index]["instance_id"],
        region_id=metadata["instances"][index]["territory_id"],
        metadata={"service_territory_id": metadata["instances"][index]["territory_id"]})


def action_routes(sequence, *, implicit_final_return=False):
    """Expose native omitted boundaries; remove only zero-cost depot padding.

    RRNCO natively prices an implicit last-node->depot edge. Appending that
    boundary is its representation conversion, not an extra sampled solution.
    Missing customers still fail the independent validator.
    """
    result = [int(x) for x in sequence]
    if not result or result[0] != 0: result.insert(0, 0)
    if implicit_final_return and result[-1] != 0: result.append(0)
    result = [node for index, node in enumerate(result) if index == 0 or node != 0 or result[index-1] != 0]
    return routes_from_sequence(result)


def _guard_step(optimizer, args, kwargs):
    gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise FloatingPointError("Native optimizer has missing or nonfinite gradients; update refused")


class RadarRunner:
    def __init__(self, cfg, run, device):
        source = run / "native_source"
        sys.path[:0] = [str(source / "acvrp"), str(source)]
        from ACVRPTrainer import ACVRPTrainer
        from ACVRPEnv import ACVRPEnv
        from utils.utils import set_result_folder
        set_result_folder(str(run / "native_logs"))
        self.cfg, self.device, self.env_class = cfg, device, ACVRPEnv
        n, width, heads = cfg["num_customers"], 256, 8
        model = dict(embedding_dim=width, sqrt_embedding_dim=width ** .5,
            encoder_layer_num=5, qkv_dim=32, sqrt_qkv_dim=32 ** .5, head_num=heads,
            init="svd", att_type="normal", logit_clipping=10, ff_hidden_dim=512,
            ms_hidden_dim=16, ms_layer1_init=(1/2) ** .5, ms_layer2_init=(1/16) ** .5,
            eval_type="softma", one_hot_seed_cnt=n)
        env = dict(node_cnt=n, pomo_size=cfg["n_traj"], fixed_data_path=str(run / "native_data/train.npz"),
                   fixed_shuffle=True, fixed_seed=cfg["training_seed"])
        optimizer = dict(optimizer=dict(lr=cfg["native_learning_rate"], weight_decay=cfg["native_weight_decay"]),
                         scheduler=dict(milestones=cfg["native_lr_milestones_data_passes"], gamma=.1))
        trainer = dict(use_cuda=device.type == "cuda", cuda_device_num=device.index or 0,
                       epochs=1, train_episodes=1, train_batch_size=cfg["batch_size"], model_load=dict(enable=False))
        self.trainer = ACVRPTrainer(env, model, optimizer, trainer)
        self.model, self.optimizer, self.scheduler = self.trainer.model, self.trainer.optimizer, self.trainer.scheduler
        self.optimizer.register_step_pre_hook(_guard_step)
        self.dataset_size = self.trainer.env.fixed_dataset_size
        self.data_pass = 1
        self._start_epoch()
        self.last_decisions = 0
        original_step = self.trainer.env.step
        def record_step(action):
            self.last_decisions += int((~self.trainer.env.finished).sum().item())
            if self.trainer.env.selected_count >= cfg["decode_cap"]:
                raise RuntimeError("Native RADAR rollout exceeded complete-solution decoding cap")
            return original_step(action)
        self.trainer.env.step = record_step
        self.architecture = model

    def _start_epoch(self):
        # Preserve the historical native ordering of scheduler.step before an
        # epoch; scheduler state is subsequently saved/restored in full.
        self.scheduler.step()
        self.trainer.env.start_fixed_epoch(self.data_pass)

    def available_batch(self):
        env = self.trainer.env
        if env.fixed_cursor == env.fixed_dataset_size:
            self.data_pass += 1; self._start_epoch()
        return env.fixed_dataset_size - env.fixed_cursor

    def train_batch(self, size):
        self.last_decisions = 0
        score, loss = self.trainer._train_one_batch(size)
        if not math.isfinite(score) or not math.isfinite(loss):
            raise FloatingPointError("Nonfinite native RADAR training statistics")
        return dict(loss=loss, best_training_cost_km=score, decision_steps=self.last_decisions,
                    completed_rollouts=int(self.trainer.env.finished.sum().item()))

    def sampler_state(self):
        return dict(data_pass=self.data_pass, cursor=self.trainer.env.fixed_cursor,
                    order=self.trainer.env.fixed_order.copy())

    def restore_sampler(self, state):
        self.data_pass = int(state["data_pass"])
        self.trainer.env.fixed_cursor = int(state["cursor"])
        self.trainer.env.fixed_order = state["order"].copy()

    def decode(self, arrays, start, stop, k):
        env = self.env_class(node_cnt=self.cfg["num_customers"], pomo_size=k)
        distance = torch.from_numpy(arrays["distance_matrix"][start:stop]).to(self.device)
        demand = torch.from_numpy(arrays["demand"][start:stop] / arrays["capacity"][start:stop, None]).to(self.device)
        env.load_problems_manual(distance, demand)
        reset, _, _ = env.reset()
        self.model.eval(); self.model.pre_forward(reset)
        state, reward, done = env.pre_step()
        while not done and env.selected_count < self.cfg["decode_cap"]:
            action, _ = self.model(state)
            state, reward, done = env.step(action)
        actions = env.selected_node_list.detach().cpu().numpy()
        costs = (-reward).detach().cpu().numpy() if reward is not None else np.full((stop-start, k), np.nan)
        completed = env.finished.detach().cpu().numpy()
        return actions, costs, completed


class RrncoRunner:
    def __init__(self, cfg, run, device):
        source = run / "native_source"
        sys.path.insert(0, str(source))
        from rrnco.envs.rcvrp.env import RCVRPEnv
        from rrnco.envs.rcvrp.fixed_generator import FixedRCVRPGenerator
        from rrnco.models.policy import RRNetPolicy
        from rrnco.models.rl import RRNet
        self.cfg, self.device = cfg, device
        self.generator = FixedRCVRPGenerator(str(run / "native_data/train.npz"), sample_with_replacement=True)
        self.env = RCVRPEnv(generator=self.generator, normalize=True)
        self.architecture = dict(env_name="rcvrp", embed_dim=128, num_heads=8,
            num_encoder_layers=6, normalization="instance", use_graph_context=False, nab_type="gating",
            init_embedding_kwargs=dict(use_coords=True, use_polar_feats=True, use_dist=True,
                                       use_matnet_init=False, sample_type="prob", sample_size=25))
        policy = RRNetPolicy(**self.architecture)
        self.model = RRNet(self.env, policy=policy, num_augment=1, num_starts=cfg["n_traj"],
            no_aug_coords=False, batch_size=cfg["batch_size"], train_data_size=self.generator.dataset_size,
            val_data_size=1, test_data_size=1, optimizer_kwargs=dict(lr=cfg["native_learning_rate"],
            weight_decay=cfg["native_weight_decay"]), lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs=dict(milestones=cfg["native_lr_milestones_data_passes"], gamma=.1)).to(device)
        # Logging only: shared_step/calculate_loss, policy, baseline and optimizer
        # creation remain native. No Lightning data generation or test hook runs.
        self.model.log_dict = lambda *args, **kwargs: None
        optimizers, scheduler = self.model.configure_optimizers()
        self.optimizer, self.scheduler = optimizers[0], scheduler["scheduler"]
        self.optimizer.register_step_pre_hook(_guard_step)
        self.dataset_size, self.seen = self.generator.dataset_size, 0
        self.last_decisions, self.last_output = 0, None
        self.batch_decision_cap = None
        self.model.policy.register_forward_hook(self._capture)
        original_step = self.env.step
        def record_step(td, *args, **kwargs):
            done = td.get("done", torch.zeros(td.batch_size, dtype=torch.bool, device=device))
            self.last_decisions += int((~done).sum().item())
            if self.batch_decision_cap is not None and self.last_decisions > self.batch_decision_cap:
                raise RuntimeError("Native RRNet rollout exceeded complete-solution decoding cap")
            return original_step(td, *args, **kwargs)
        self.env.step = record_step

    def _capture(self, module, inputs, output):
        self.last_output = output

    def available_batch(self):
        # Preserve the native fixed generator's with-replacement sampling. A
        # counter boundary controls the original per-data-pass LR scheduler.
        return self.dataset_size - (self.seen % self.dataset_size)

    def train_batch(self, size):
        self.last_decisions = 0
        self.batch_decision_cap = size * self.cfg["n_traj"] * self.cfg["decode_cap"]
        batch = self.generator([size]).to(self.device)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        result = self.model.shared_step(batch, 0, "train")
        loss = result["loss"]
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite native RRNet loss")
        loss.backward(); self.optimizer.step()
        from rl4co.utils.ops import unbatchify
        reward = unbatchify(self.last_output["reward"], self.cfg["n_traj"])
        score = -reward.max(-1).values.mean().item()
        self.seen += size
        if self.seen % self.dataset_size == 0: self.scheduler.step()
        self.last_output = None
        return dict(loss=loss.item(), best_training_cost_km=score, decision_steps=self.last_decisions,
                    completed_rollouts=size*self.cfg["n_traj"])

    def sampler_state(self):
        return dict(seen=self.seen, sampling="native_fixed_generator_with_replacement", numpy_rng_saved_separately=True)

    def restore_sampler(self, state):
        self.seen = int(state["seen"])

    def decode(self, arrays, start, stop, k):
        from tensordict import TensorDict
        from rl4co.utils.ops import unbatchify
        self.batch_decision_cap = None
        td = TensorDict({key: torch.from_numpy(arrays[key][start:stop]).to(self.device)
                         for key in ("depot", "locs", "demand", "capacity", "distance_matrix")}, batch_size=[stop-start])
        td["demand"] = td["demand"] / td["capacity"][:, None]
        self.model.eval()
        out = self.model.policy(self.env.reset(td), self.env, phase="test", num_starts=k,
                                max_steps=self.cfg["decode_cap"]-1, decode_type="multistart_greedy")
        actions = unbatchify(out["actions"], k).detach().cpu().numpy()
        costs = -unbatchify(out["reward"], k).detach().cpu().numpy()
        n = self.cfg["num_customers"]
        completed = np.stack([(actions == node).sum(-1) == 1 for node in range(1, n+1)]).all(0)
        self.last_output = None
        return actions, costs, completed


def evaluate_runner(runner, cfg, run, *, split, checkpoint_id, output, inference_seed):
    arrays, metadata = load_data(run, split)
    protocol = cfg.get("experiment_protocol", {})
    count = metadata["count"]
    if not protocol.get("smoke_only", False) and count != 1000:
        raise ValueError("Formal native validation/test requires all 1000 instances")
    limit = cfg.get("eval_limit")
    if limit is not None:
        if not protocol.get("hardware_preflight_only", False) or split != "val":
            raise ValueError("Native eval_limit is permitted only for hardware preflight validation")
        count = min(count, int(limit))
    rows, wall_start = [], time.perf_counter()
    model_was_training = runner.model.training
    with evaluation_rng(inference_seed), torch.no_grad():
        for start in range(0, count, cfg["eval_batch_size"]):
            stop = min(start + cfg["eval_batch_size"], count)
            if runner.device.type == "cuda": torch.cuda.synchronize(runner.device)
            begin = time.perf_counter()
            actions, costs, completed = runner.decode(arrays, start, stop, cfg["eval_k"])
            if runner.device.type == "cuda": torch.cuda.synchronize(runner.device)
            batch_rows = []
            for local, index in enumerate(range(start, stop)):
                instance = instance_for(arrays, metadata, index)
                candidates = []
                for attempt in range(actions.shape[1]):
                    failure = None
                    try:
                        routes = action_routes(actions[local, attempt].tolist(), implicit_final_return=cfg["method"] == "rrnco")
                    except (ValueError, IndexError) as exc:
                        routes, failure = [], str(exc)
                    reported = float(costs[local, attempt])
                    candidates.append(dict(routes=routes, completed=bool(completed[local, attempt]),
                        truncated=not bool(completed[local, attempt]),
                        reported_cost=reported if math.isfinite(reported) else None,
                        failure_reason=failure))
                row = select_candidates(instance, candidates, method=cfg["method"], run_id=cfg.get("run_id", run.name),
                    training_seed=cfg["training_seed"], inference_seed=inference_seed,
                    checkpoint_id=checkpoint_id, requested_k=cfg["eval_k"],
                    decode_mode="native_greedy_multistart", timing_batch_size=stop-start)
                row.update(split=split, num_augment=1, implicit_final_return=cfg["method"] == "rrnco",
                           offline_dataset_conversion_excluded_from_timing=True)
                batch_rows.append(row)
            if runner.device.type == "cuda": torch.cuda.synchronize(runner.device)
            batch_time = time.perf_counter() - begin
            for row in batch_rows:
                row.update(runtime=batch_time/(stop-start), batch_runtime_s=batch_time)
            rows.extend(batch_rows)
    runner.model.train(model_was_training)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output)
    feasible = [row["recomputed_cost_km"] for row in rows if row["feasible"]]
    return dict(**summarize_rows(rows), feasible_instances=len(feasible),
                actual_candidates=sum(row["actual_K"] for row in rows),
                wall_time_s=time.perf_counter()-wall_start, output=str(output), output_sha256=file_hash(output),
                checkpoint_id=checkpoint_id, split=split, evaluation_batch_size=cfg["eval_batch_size"], selection_rule="maximize_feasible_count_then_minimize_feasible_mean_km")


def runtime_versions(method):
    names = ("torch", "numpy") + (("rl4co", "tensordict", "torchrl", "lightning") if method == "rrnco" else ())
    return {name: importlib.metadata.version(name) for name in names}


def save_checkpoint(runner, cfg, run, manifest, counters, best_metric, *, name):
    folder = run / "native_checkpoints"; folder.mkdir(exist_ok=True)
    path = folder / name
    payload = dict(schema="aaai_e1_native_checkpoint_v1", method=cfg["method"],
        model=runner.model.state_dict(), optimizer=runner.optimizer.state_dict(), scheduler=runner.scheduler.state_dict(),
        scaler=None, critic=None, rng=rng_state(), sampler=runner.sampler_state(), counters=deepcopy(counters),
        best_metric=best_metric, config_sha256=manifest["config_sha256"],
        source_sha256=manifest["source"]["sha256"], harness_sha256=manifest["harness"]["sha256"],
        dataset_hashes={key:value["npz_sha256"] for key,value in manifest["datasets"].items()},
        architecture=runner.architecture, runtime_versions=runtime_versions(cfg["method"]), saved_at=now())
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    backup = Path(cfg["backup_dir"]); backup.mkdir(parents=True, exist_ok=True)
    backup_temp = backup / (name + ".tmp")
    shutil.copy2(path, backup_temp)
    with backup_temp.open("rb") as stream: os.fsync(stream.fileno())
    digest = file_hash(path)
    if file_hash(backup_temp) != digest: raise IOError("Native checkpoint backup verification failed")
    os.replace(backup_temp, backup / name)
    return dict(path=str(path), sha256=digest, backup_path=str(backup / name), backup_verified=True)


def restore_checkpoint(runner, cfg, manifest, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("config_sha256",manifest["config_sha256"]), ("source_sha256",manifest["source"]["sha256"]),
                           ("harness_sha256",manifest["harness"]["sha256"])):
        if state.get(key) != expected: raise ValueError(f"Native checkpoint mismatch: {key}")
    if state["dataset_hashes"] != {key:value["npz_sha256"] for key,value in manifest["datasets"].items()}:
        raise ValueError("Native checkpoint dataset identity changed")
    if state.get("runtime_versions") != runtime_versions(cfg["method"]):
        raise ValueError("Native checkpoint framework versions changed; exact recovery is not established")
    runner.model.load_state_dict(state["model"], strict=True)
    runner.optimizer.load_state_dict(state["optimizer"])
    runner.scheduler.load_state_dict(state["scheduler"])
    runner.restore_sampler(state["sampler"])
    restore_rng(state["rng"])
    return state


def run_training(runner, cfg, run, manifest, *, resume, max_updates, checkpoint=None):
    counters = dict(instance_exposures=0, complete_rollouts=0, decision_steps=0, optimizer_updates=0,
                    training_wall_time_s=0., evaluation_wall_time_s=0.)
    best_metric = None
    if resume:
        recovery = Path(checkpoint) if checkpoint else run / "native_checkpoints/last.pt"
        state = restore_checkpoint(runner, cfg, manifest, recovery)
        counters, best_metric = state["counters"], state["best_metric"]
        with (run / "native_resume.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(checkpoint=str(recovery), sha256=file_hash(recovery),
                counters=counters, resumed_at=now()), allow_nan=False)+"\n")
    started_updates = counters["optimizer_updates"]
    while counters["instance_exposures"] < cfg["instance_exposures"]:
        if max_updates is not None and counters["optimizer_updates"] - started_updates >= max_updates: break
        before_exposures = counters["instance_exposures"]
        next_eval = (before_exposures // cfg["eval_interval_exposures"] + 1)*cfg["eval_interval_exposures"]
        size = min(cfg["batch_size"], cfg["instance_exposures"]-before_exposures,
                   next_eval-before_exposures, runner.available_batch())
        begin = time.perf_counter()
        row = runner.train_batch(size)
        if runner.device.type == "cuda": torch.cuda.synchronize(runner.device)
        elapsed = time.perf_counter()-begin
        counters["instance_exposures"] += size
        counters["optimizer_updates"] += 1
        counters["complete_rollouts"] += row["completed_rollouts"]
        counters["decision_steps"] += row["decision_steps"]
        counters["training_wall_time_s"] += elapsed
        row.update(counters, timestamp=now(), batch_instances=size, n_traj=cfg["n_traj"],
                   update_wall_time_s=elapsed, lr=runner.optimizer.param_groups[0]["lr"],
                   nominal_data_passes=counters["instance_exposures"]/runner.dataset_size)
        with (run / "native_train.jsonl").open("a") as stream: stream.write(json.dumps(row, allow_nan=False)+"\n")
        evaluate_now = counters["instance_exposures"] % cfg["eval_interval_exposures"] == 0 or counters["instance_exposures"] == cfg["instance_exposures"]
        if evaluate_now:
            token = f"exposures_{counters['instance_exposures']:08d}"
            result = evaluate_runner(runner, cfg, run, split="val", checkpoint_id=token,
                output=run / "native_evaluations" / f"{token}.jsonl", inference_seed=cfg["inference_seed"])
            counters["evaluation_wall_time_s"] += result["wall_time_s"]
            metric = (result["feasible_instances"], (-result["mean_distance_km"] if result["mean_distance_km"] is not None else -1e300))
            result.update(instance_exposures=counters["instance_exposures"], optimizer_updates=counters["optimizer_updates"])
            with (run / "native_validation.jsonl").open("a") as stream: stream.write(json.dumps(result, allow_nan=False)+"\n")
            if best_metric is None or metric > tuple(best_metric):
                best_metric = metric
                save_checkpoint(runner, cfg, run, manifest, counters, best_metric, name="best.pt")
        if evaluate_now or counters["optimizer_updates"] % cfg["checkpoint_interval_updates"] == 0:
            save_checkpoint(runner, cfg, run, manifest, counters, best_metric, name="last.pt")
        atomic_json(run / "native_status.json", dict(state="running", method=cfg["method"], counters=counters,
                    total_instance_exposures=cfg["instance_exposures"], updated=now(), pretrained_initialization=False))
        print(f"[{cfg['method']}] exposures={counters['instance_exposures']}/{cfg['instance_exposures']} updates={counters['optimizer_updates']} loss={row['loss']:.5f} native_cost={row['best_training_cost_km']:.4f}", flush=True)
    checkpoint = save_checkpoint(runner, cfg, run, manifest, counters, best_metric, name="last.pt")
    done = counters["instance_exposures"] == cfg["instance_exposures"]
    atomic_json(run / "native_status.json", dict(state="completed" if done else "paused", method=cfg["method"],
        counters=counters, total_instance_exposures=cfg["instance_exposures"], updated=now(),
        checkpoint=checkpoint, training_complete=done, test_complete=False,
        pretrained_initialization=False, best_metric=list(best_metric) if best_metric is not None else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate")); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int); parser.add_argument("--checkpoint")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--split", default="val", choices=("val", "test")); parser.add_argument("--output")
    args = parser.parse_args(); run = Path(args.run_dir).resolve()
    if args.eval_batch_size is not None and (args.command != "evaluate" or args.eval_batch_size <= 0):
        raise ValueError("--eval-batch-size must be positive and is available only for evaluation")
    lock_handle = (run / "native_worker.lock").open("a")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((run / "native_resolved.json").read_text())
    manifest = json.loads((run / "native_manifest.json").read_text())
    if json_hash(cfg) != manifest["config_sha256"]: raise ValueError("Native resolved config changed")
    verify_files(run / "native_source", manifest["source"]); verify_files(run / "native_harness", manifest["harness"])
    for split in ("train", "val"):
        _, metadata = load_data(run, split)
        if metadata != manifest["datasets"][split]: raise ValueError(f"Dataset sidecar changed: {split}")
    torch.set_num_threads(cfg["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    random.seed(cfg["training_seed"]); np.random.seed(cfg["training_seed"]); torch.manual_seed(cfg["training_seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg["training_seed"])
    runner = (RadarRunner if cfg["method"] == "radar" else RrncoRunner)(cfg, run, device)
    atomic_json(run / "native_architecture.json", dict(method=cfg["method"], architecture=runner.architecture,
                parameters=sum(p.numel() for p in runner.model.parameters()), native_objective=True))
    if args.command == "train":
        if args.checkpoint is not None and not args.resume: raise ValueError("Training checkpoint requires --resume")
        run_training(runner, cfg, run, manifest, resume=args.resume, max_updates=args.max_updates, checkpoint=args.checkpoint)
    else:
        if args.checkpoint is None: raise ValueError("Evaluation requires an explicit selected checkpoint")
        restore_checkpoint(runner, cfg, manifest, Path(args.checkpoint))
        evaluation_cfg = deepcopy(cfg)
        if args.eval_batch_size is not None:
            evaluation_cfg["eval_batch_size"] = args.eval_batch_size
        result = evaluate_runner(runner, evaluation_cfg, run, split=args.split, checkpoint_id=file_hash(args.checkpoint),
            output=args.output or run / "native_evaluations" / f"final_{args.split}.jsonl", inference_seed=cfg["inference_seed"])
        atomic_json(run / "native_evaluation_status.json", dict(state="completed", updated=now(), **result))


if __name__ == "__main__":
    try: main()
    except Exception as error:
        if isinstance(error, BlockingIOError):
            # A second dispatcher must not overwrite the live worker's status.
            raise
        if "--run-dir" in sys.argv:
            run = Path(sys.argv[sys.argv.index("--run-dir")+1])
            command = sys.argv[1] if len(sys.argv) > 1 else None
            failure = dict(state="failed", error=repr(error), updated=now())
            if command != "evaluate": failure["training_complete"] = False
            status_name = "native_evaluation_status.json" if command == "evaluate" else "native_status.json"
            atomic_json(run / status_name, failure)
        raise
