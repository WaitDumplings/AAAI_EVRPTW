"""E1 interfaces to frozen native RRNCO/RADAR implementations.

The source architecture and native REINFORCE/POMO training steps are retained.
This module owns data conversion, source freezing and subprocess dispatch only;
no controlled PPO model, expert data, or RDI/AGDA/SL-PPO plugin is imported.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = {
    "rrnco": Path("/data/Maojie/Github2/RRNCO/real-routing-nco"),
    "radar": Path("/data/Maojie/Github2/RADAR/RADAR"),
}
EXTERNAL_SOURCES = dict(DEFAULT_SOURCES)
for _method in DEFAULT_SOURCES:
    _vendored = ROOT / "benchmarks/native_sources" / _method
    if _vendored.is_dir():
        DEFAULT_SOURCES[_method] = _vendored


def _default_python(method):
    override = os.environ.get(f"E1_{method.upper()}_PYTHON")
    if override:
        return str(Path(override).expanduser())
    if all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES[method]):
        return sys.executable
    external = EXTERNAL_SOURCES[method] / ".venv/bin/python"
    return str(external if method == "rrnco" and external.is_file() else sys.executable)


REQUIRED = {
    "rrnco": ("rrnco/models/rl.py", "rrnco/models/policy.py", "rrnco/envs/rcvrp/env.py",
              "rrnco/envs/rcvrp/fixed_generator.py", "configs/experiment/rrnet.yaml"),
    "radar": ("acvrp/ACVRPTrainer.py", "acvrp/ACVRPModel.py", "acvrp/ACVRPModel_LIB.py",
              "acvrp/ACVRPEnv.py", "ACVRProblemDef.py", "utils/utils.py"),
}
DEPENDENCIES = {"rrnco": ("torch", "numpy", "rl4co", "tensordict", "torchrl", "lightning", "hydra", "orjson", "matplotlib", "einops", "pandas"),
                "radar": ("torch", "numpy", "matplotlib")}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def native_status(method, source_root=None, python_executable=None):
    if method not in REQUIRED:
        raise ValueError("native method must be rrnco or radar")
    source = Path(source_root or DEFAULT_SOURCES[method]).resolve()
    interpreter = str(python_executable or _default_python(method))
    missing = [name for name in REQUIRED[method] if not (source / name).is_file()]
    modules = {}
    error = None
    try:
        probe = "import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n) is not None for n in " + repr(DEPENDENCIES[method]) + "}))"
        output = subprocess.run([interpreter, "-c", probe], capture_output=True, text=True, check=True, timeout=30)
        modules = json.loads(output.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        error = str(exc)
    return dict(method=method, source_root=str(source), python_executable=interpreter,
                missing_source_files=missing, dependencies=modules, dependency_probe_error=error,
                source_available=not missing, dependencies_available=bool(modules) and all(modules.values()),
                smoke_passed=False, algorithm="native_REINFORCE_POMO", expert_data_used=False,
                distributed_support="single_process_only", pretrained_initialization=False,
                interpreter_environment_variable=f"E1_{method.upper()}_PYTHON",
                requirements_lock=str(ROOT / "e1" / f"requirements-{method}.lock.txt"),
                setup_hint=f"Create a Python 3.12 venv, install e1/requirements-{method}.lock.txt, then export E1_{method.upper()}_PYTHON=/path/to/venv/bin/python. Native source is vendored; no pretrained weights are required.")


def _source_files(method, source):
    if method == "rrnco":
        candidates = [*source.joinpath("rrnco").rglob("*.py"), *source.joinpath("configs").rglob("*.yaml")]
        candidates += [source / name for name in ("train.py", "test.py", "pyproject.toml", "LICENSE", "README.md")]
    else:
        candidates = [*source.joinpath("acvrp").glob("*.py"), *source.joinpath("utils").rglob("*.py"),
                      source / "ACVRProblemDef.py", source / "README.md"]
    return sorted({p for p in candidates if p.is_file() and "__pycache__" not in p.parts})


def freeze_source(method, source_root, target):
    source, target = Path(source_root).resolve(), Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen native source: {target}")
    for name in REQUIRED[method]:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    target.mkdir(parents=True)
    records = []
    for path in _source_files(method, source):
        relative = path.relative_to(source)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records.append(dict(path=relative.as_posix(), sha256=file_hash(dest)))
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    upstream = source / "SOURCE_MANIFEST.json"
    return dict(upstream_provenance=json.loads(upstream.read_text()) if upstream.is_file() else None, upstream_manifest_sha256=file_hash(upstream) if upstream.is_file() else None, source_root=str(source), git_head=result.stdout.strip() if result.returncode == 0 else None,
                files=records, sha256=json_hash(records), includes_uncommitted_adapters=True)


def verify_files(root, manifest):
    for item in manifest["files"]:
        path = Path(root) / item["path"]
        if not path.is_file() or file_hash(path) != item["sha256"]:
            raise ValueError(f"Frozen source/data fingerprint changed: {path}")


def convert_cvrp_instances(instances, *, expected_customers=100):
    """Retain directed km and raw demand/capacity; only coordinates use native minmax."""
    arrays = {key: [] for key in ("depot", "locs", "demand", "capacity", "distance_matrix")}
    identities, seen = [], set()
    for instance in instances:
        n = int(instance.num_customers)
        identity = str(instance.instance_id)
        if not identity or identity in seen:
            raise ValueError(f"Missing or duplicate instance identity: {identity}")
        seen.add(identity)
        if n != expected_customers or instance.num_charging_stations != 0:
            raise ValueError("Native E1 input must have the declared customer count and no charging stations")
        distance = np.asarray(instance.distance_matrix_km, dtype=np.float32)
        demand = np.asarray(instance.demands_cm3, dtype=np.float32)
        capacity = float(instance.vehicle["cargo_capacity_cm3"])
        coordinates = np.concatenate((np.asarray(instance.depot)[None], np.asarray(instance.customers)), axis=0).astype(np.float32)
        if distance.shape != (n+1, n+1) or not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError("Native CVRP requires a finite nonnegative authoritative road matrix")
        if not np.all(np.diag(distance) == 0):
            raise ValueError("Zero diagonal is required by native RRNCO's min-max reward inversion")
        if not np.isfinite(capacity) or capacity <= 0 or demand.shape != (n,) or not np.isfinite(demand).all() or (demand < 0).any() or (demand > capacity).any():
            raise ValueError("Invalid demand/capacity or individually infeasible customer")
        if coordinates.shape != (n+1, 2) or not np.isfinite(coordinates).all():
            raise ValueError("Invalid depot/customer coordinates")
        low, high = coordinates.min(0), coordinates.max(0)
        coordinates = (coordinates-low) / np.where(high > low, high-low, 1.)
        for key, value in dict(depot=coordinates[0], locs=coordinates[1:], demand=demand,
                               capacity=np.float32(capacity), distance_matrix=distance).items():
            arrays[key].append(value)
        from e1.assets import _fingerprint
        identities.append(dict(instance_id=identity, semantic_sha256=_fingerprint(instance), territory_id=str(instance.metadata.get("service_territory_id", instance.metadata.get("source_territory_id", getattr(instance, "region_id", ""))))))
    if not identities:
        raise ValueError("Empty native dataset")
    return {name: np.stack(values) for name, values in arrays.items()}, identities


def export_native_dataset(source_path, output_path, *, split, expected_customers=100, limit=None):
    from offline2online.instance_adapter import iter_instance_payloads, adapt_instance_payload
    import itertools
    source = Path(source_path).resolve()
    bundle = source / "instances.pkl" if source.is_dir() else source
    payloads = iter_instance_payloads(bundle)
    if limit is not None:
        payloads = itertools.islice(payloads, int(limit))
    instances = (adapt_instance_payload(item, problem_type="cvrp", strict_road_metric=True) for item in payloads)
    arrays, identities = convert_cvrp_instances(instances, expected_customers=expected_customers)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with target.open("wb") as stream:
        np.savez(stream, **arrays)
    manifest = dict(split=split, source_path=str(bundle), source_sha256=file_hash(bundle),
                    npz_sha256=file_hash(target), instances=identities, count=len(identities),
                    customers=expected_customers, distance_unit="km", directed=True,
                    demand_storage="raw_demand_and_raw_capacity", coordinate_transform="native_per_axis_minmax")
    atomic_json(target.with_suffix(".json"), manifest)
    return manifest


def _config(config):
    cfg = deepcopy(config)
    # A recovery artifact is an execution choice, not a new training recipe.
    cfg.pop("resume_checkpoint", None)
    method = cfg["method"]
    if method not in REQUIRED:
        raise ValueError("Only native rrnco/radar are handled here")
    cfg.setdefault("training_seed", 3009)
    cfg.setdefault("instance_exposures", 96000)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("n_traj", 50)
    cfg.setdefault("eval_k", 50)
    cfg.setdefault("eval_interval_exposures", 3200)
    cfg.setdefault("eval_batch_size", 16)
    cfg.setdefault("inference_seed", cfg.get("eval_seed", 17000000 + int(cfg["training_seed"])))
    if cfg.get("checkpoint_backup_dir"):
        cfg.setdefault("backup_dir", cfg["checkpoint_backup_dir"])
    cfg.setdefault("checkpoint_interval_updates", 50)
    cfg.setdefault("num_customers", 100)
    cfg.setdefault("world_size", 1)
    cfg.setdefault("cpu_threads", 1)
    if cfg.get("eval_limit") is not None:
        if not cfg.get("experiment_protocol", {}).get("hardware_preflight_only", False):
            raise ValueError("Native eval_limit is allowed only for hardware_preflight_only; formal validation/test evaluates all instances")
        if isinstance(cfg["eval_limit"], bool) or not isinstance(cfg["eval_limit"], int) or cfg["eval_limit"] <= 0:
            raise ValueError("eval_limit must be a positive integer")
    if cfg["world_size"] != 1:
        raise ValueError("Native E1 runner currently supports one GPU/process; it must not masquerade as DDP")
    for key in ("instance_exposures", "batch_size", "n_traj", "eval_k", "eval_interval_exposures", "eval_batch_size", "checkpoint_interval_updates", "num_customers", "cpu_threads"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["n_traj"] < 2 or cfg["n_traj"] > cfg["num_customers"] or cfg["eval_k"] > cfg["num_customers"]:
        raise ValueError("Native multistart counts must fit customer count; training needs at least two starts")
    for key in ("train_path", "val_path"):
        cfg[key] = str(Path(cfg[key]).resolve())
    if cfg["train_path"] == cfg["val_path"]:
        raise ValueError("Training and validation paths must differ")
    cfg["native_source_root"] = str(Path(cfg.get("native_source_root", DEFAULT_SOURCES[method])).resolve())
    cfg["native_python"] = str(cfg.get("native_python", _default_python(method)))
    cfg["native_algorithm"] = "native_REINFORCE_POMO"
    cfg["num_augment"] = 1
    cfg["initialization"] = "scratch"
    cfg["native_learning_rate"] = 4e-4
    cfg["native_weight_decay"] = 1e-6
    cfg["native_lr_milestones_data_passes"] = [180, 195] if method == "rrnco" else [2001, 2101]
    cfg["decode_cap"] = 2 * cfg["num_customers"] + 1
    cfg["epoch_semantics"] = "one optimizer update per native batch; data_pass is floor(train instance exposures / train pool size)"
    return cfg


def prepare_native(config, run_dir):
    run = Path(run_dir).resolve()
    if (run / "native_resolved.json").exists():
        raise FileExistsError("A native run already exists; use resume, not scratch initialization")
    cfg = _config(config)
    readiness = native_status(cfg["method"], cfg["native_source_root"], cfg["native_python"])
    if not readiness["source_available"] or not readiness["dependencies_available"]:
        raise RuntimeError(f"Native assets/dependencies unavailable: {readiness}")
    run.mkdir(parents=True, exist_ok=True)
    source = freeze_source(cfg["method"], cfg["native_source_root"], run / "native_source")
    harness = run / "native_harness" / "e1"
    harness.mkdir(parents=True)
    records = []
    for name in ("native.py", "native_worker.py", "validator.py", "evaluation.py", f"requirements-{cfg["method"]}.lock.txt"):
        shutil.copy2(ROOT / "e1" / name, harness / name)
        records.append(dict(path=f"e1/{name}", sha256=file_hash(harness / name)))
    (harness / "__init__.py").write_text("")
    records.append(dict(path="e1/__init__.py", sha256=file_hash(harness / "__init__.py")))
    manifests = {split: export_native_dataset(cfg[f"{split}_path"], run / "native_data" / f"{split}.npz",
                 split=split, expected_customers=cfg["num_customers"], limit=cfg.get(f"smoke_{split}_limit")) for split in ("train", "val")}
    protocol = cfg.get("experiment_protocol", {})
    if not protocol.get("smoke_only", False) and manifests["val"]["count"] != 1000:
        raise ValueError("Formal native validation requires all 1000 instances (use smoke_only only for a separate smoke run)")
    train_ids = {row["instance_id"] for row in manifests["train"]["instances"]}
    train_fingerprints = {row["semantic_sha256"] for row in manifests["train"]["instances"]}
    if (train_ids & {row["instance_id"] for row in manifests["val"]["instances"]}
            or train_fingerprints & {row["semantic_sha256"] for row in manifests["val"]["instances"]}):
        raise ValueError("Train and validation instance IDs or semantic fingerprints overlap")
    cfg["backup_dir"] = str(Path(cfg.get("backup_dir", run.parent / "checkpoint_backups" / run.name)).resolve())
    if Path(cfg["backup_dir"]) == run / "native_checkpoints":
        raise ValueError("Checkpoint backup must be an independent directory")
    manifest = dict(config_sha256=json_hash(cfg), source=source,
                    harness=dict(files=records, sha256=json_hash(records)), datasets=manifests)
    atomic_json(run / "native_manifest.json", manifest)
    atomic_json(run / "native_resolved.json", cfg)
    atomic_json(run / "native_status.json", dict(state="prepared", method=cfg["method"], trained=False, smoke_passed=False))
    return dict(config=cfg, manifest=manifest, run_dir=str(run))


def _dispatch(run_dir, command, *, device="cpu", resume=False, max_updates=None, checkpoint=None,
              split="val", output_path=None, eval_batch_size=None):
    run = Path(run_dir).resolve()
    cfg = json.loads((run / "native_resolved.json").read_text())
    manifest = json.loads((run / "native_manifest.json").read_text())
    if json_hash(cfg) != manifest["config_sha256"]:
        raise ValueError("Resolved native configuration changed")
    verify_files(run / "native_source", manifest["source"])
    verify_files(run / "native_harness", manifest["harness"])
    args = [cfg["native_python"], str(run / "native_harness/e1/native_worker.py"), command,
            "--run-dir", str(run), "--device", device]
    if resume: args += ["--resume"]
    if max_updates is not None: args += ["--max-updates", str(max_updates)]
    if checkpoint is not None: args += ["--checkpoint", str(Path(checkpoint).resolve())]
    if command == "evaluate": args += ["--split", split]
    if eval_batch_size is not None:
        if command != "evaluate": raise ValueError("Evaluation batch override cannot change native training")
        args += ["--eval-batch-size", str(eval_batch_size)]
    if output_path is not None: args += ["--output", str(Path(output_path).resolve())]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        environment["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
        args[args.index("--device") + 1] = "cuda:0"
    else:
        raise ValueError("Native device must be cpu or cuda[:index]")
    result = subprocess.run(args, cwd=run, env=environment)
    if result.returncode:
        status_name = "native_status.json" if command == "train" else "native_evaluation_status.json"
        raise RuntimeError(f"Native {cfg['method']} worker exited {result.returncode}; see {status_name}")
    return json.loads((run / ("native_status.json" if command == "train" else "native_evaluation_status.json")).read_text())


def train(config, run_dir, *, resume=False, device="cpu", max_updates=None):
    run = Path(run_dir)
    recovery = config.get("resume_checkpoint")
    if recovery is not None and not resume:
        raise ValueError("resume_checkpoint requires resume=True; scratch training never loads weights")
    if not resume:
        prepare_native(config, run)
    else:
        previous = json.loads((run / "native_resolved.json").read_text())
        requested = _config(config)
        for key, value in requested.items():
            if previous.get(key) != value:
                raise ValueError(f"Native resume configuration differs: {key}")
    return _dispatch(run, "train", device=device, resume=resume, max_updates=max_updates, checkpoint=recovery)


def evaluate(config, run_dir, checkpoint, *, split="val", device="cpu", output_path=None, eval_batch_size=None):
    if eval_batch_size is not None and (isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int) or eval_batch_size <= 0):
        raise ValueError("eval_batch_size must be a positive integer")
    if split not in {"val", "test"}:
        raise ValueError("Native final evaluation split must be val or test")
    run = Path(run_dir)
    if split == "test" and not (run / "native_data/test.npz").exists():
        cfg = json.loads((run / "native_resolved.json").read_text())
        if "test_path" not in config:
            raise ValueError("Explicit test_path is required; training never reads test")
        export_native_dataset(config["test_path"], run / "native_data/test.npz", split="test", expected_customers=cfg["num_customers"])
    return _dispatch(run, "evaluate", device=device, checkpoint=checkpoint, split=split, output_path=output_path, eval_batch_size=eval_batch_size)
