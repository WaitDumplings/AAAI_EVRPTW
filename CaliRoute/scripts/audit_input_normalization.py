#!/usr/bin/env python3
"""Sample train/val input units; does not evaluate model or solver performance.

Checks first/middle/last records of each available task/size/split. Streaming a
bundle to locate those records does not validate the remaining records. The
report preserves saved-vs-reconstructed time discrepancies explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliroute.input_normalization import (
    NODE_INPUT_CONTEXT_FEATURES, GRAPH_INPUT_CONTEXT_FEATURES,
    input_normalization_signature,
)
from offline2online.instance_adapter import iter_instance_payloads, adapt_instance_payload
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env import EVRPTWVectorEnv
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env_fast import EVRPTWVectorEnvFast


def compare_arrays(left, right, atol=1e-4, rtol=1e-6):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        return {"pass": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    finite_left, finite_right = np.isfinite(left), np.isfinite(right)
    finite = finite_left & finite_right
    delta = np.abs(left[finite] - right[finite])
    bad = delta > atol + rtol * np.abs(left[finite])
    same_nonfinite = np.array_equal(finite_left, finite_right) and np.array_equal(left[~finite], right[~finite])
    # matching infinities are an explicit unreachable marker; NaNs never pass.
    ok = same_nonfinite and not np.isnan(left).any() and not np.isnan(right).any() and not bad.any()
    return {"pass": bool(ok), "finite_entries": int(finite.sum()),
            "nonfinite_entries": int((~finite_left).sum()),
            "max_abs_error": float(delta.max()) if delta.size else 0.,
            "max_relative_error": float((delta / np.maximum(np.abs(left[finite]), 1e-8)).max()) if delta.size else 0.,
            "entries_outside_tolerance": int(bad.sum()), "atol": atol, "rtol": rtol}


def quantiles(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"count": len(values), "finite_count": len(finite),
            "min_p01_p50_p99_max": np.quantile(finite, [0, .01, .5, .99, 1]).tolist() if finite.size else []}


def metadata_count(directory):
    for name in ("metadata.json", "public_metadata.json"):
        path = directory / name
        if path.exists():
            obj = json.loads(path.read_text())
            count = obj.get("num_instances", obj.get("dataset_instances"))
            if count is not None:
                return int(count)
    raise ValueError(f"No declared instance count in {directory}")


def check_instance(payload, problem, length_unit, reward_unit):
    instance = adapt_instance_payload(payload, problem_type=problem)
    common = dict(instance=instance, n_traj=1, observation_distance_scale_km=length_unit,
                  reward_distance_scale_km=reward_unit)
    legacy = EVRPTWVectorEnv(**common)
    slow = EVRPTWVectorEnv(**common, observation_coordinate_mode="depot_fixed", observation_input_context=True)
    fast = EVRPTWVectorEnvFast(**common, observation_coordinate_mode="depot_fixed", observation_input_context=True,
                              use_jit_mask=False)
    old, _ = legacy.reset(seed=17)
    obs, _ = slow.reset(seed=17)
    fast_obs, _ = fast.reset(seed=17)
    mismatches = [name for name in obs if not np.array_equal(obs[name], fast_obs[name])]
    other_changed = [name for name in old if name not in {"cus_loc", "depot_loc", "rs_loc"}
                     and not np.array_equal(old[name], obs[name])]
    physical = {
        "distance_reconstruction_km": compare_arrays(slow.distance_km, obs["edge_distance"] * length_unit),
        "time_reconstruction_s": compare_arrays(slow.travel_time_s, obs["edge_time"] * slow.horizon_s),
        "energy_reconstruction_kwh": compare_arrays(slow.energy_kwh, obs["edge_energy"] * slow.battery_capacity_kwh),
        "time_window_reconstruction_s": compare_arrays(slow.tw_s, obs["time_window"] * slow.horizon_s + slow.working_start_s),
    }
    for name in ("travel_time_matrix_s", "time_matrix_s", "travel_time_matrix"):
        if name in payload:
            physical["saved_vs_environment_time_s"] = compare_arrays(payload[name], slow.travel_time_s)
            break
    node_names = ("depot_loc", "cus_loc", "rs_loc")
    all_xy = np.concatenate([obs[name] for name in node_names])
    physical["coordinate_relative_km"] = compare_arrays(slow.coords_raw - slow.coords_raw[0], all_xy * length_unit)
    resources = {"speed_kmh": slow.speed_kmh, "horizon_s": slow.horizon_s,
                 "battery_capacity_kwh": slow.battery_capacity_kwh, "cargo_capacity_cm3": slow.cargo_capacity_cm3,
                 "energy_kwh_per_km": slow.energy_per_km, "distance_unit_km": length_unit,
                 "reward_unit_km": reward_unit}
    resources = {key: value if np.isfinite(value) else str(value) for key, value in resources.items()}
    # One identical feasible action is a dynamics check, not a route-quality evaluation.
    feasible = np.flatnonzero(obs["action_mask"][0, 1:1 + instance.num_customers]) + 1
    step = {"checked": False}
    if feasible.size:
        destination = int(feasible[0])
        records = [env.step([destination]) for env in (legacy, slow, fast)]
        reward_equal = np.array_equal(records[0][1], records[1][1]) and np.array_equal(records[1][1], records[2][1])
        state_equal = all(np.array_equal(getattr(legacy, name), getattr(slow, name))
                          and np.array_equal(getattr(slow, name), getattr(fast, name))
                          for name in ("current_time_s", "battery_used_kwh", "load_cm3", "objective_distance_km"))
        mask_equal = np.array_equal(records[0][0]["action_mask"], records[1][0]["action_mask"])
        fast_equal = all(np.array_equal(records[1][0][key], records[2][0][key]) for key in records[1][0])
        step = {"checked": True, "destination": destination, "reward_equal": bool(reward_equal),
                "physical_state_equal": bool(state_equal), "legacy_mask_equal": bool(mask_equal),
                "fast_all_observations_equal": bool(fast_equal)}
    arrays = {key: obs[key] for key in ("edge_distance", "edge_time", "edge_energy", "demand", "time_window", "service_time")}
    arrays["depot_relative_xy"] = all_xy
    for i, name in enumerate(NODE_INPUT_CONTEXT_FEATURES):
        arrays["node/" + name] = obs["node_input_context"][:, i]
    for i, name in enumerate(GRAPH_INPUT_CONTEXT_FEATURES):
        arrays["graph/" + name] = obs["graph_input_context"][i:i+1]
    context_finite = bool(np.isfinite(obs["node_input_context"]).all() and np.isfinite(obs["graph_input_context"]).all())
    passed = (not mismatches and not other_changed and context_finite and all(row["pass"] for row in physical.values())
              and (not step["checked"] or all(value for key, value in step.items() if key not in {"checked", "destination"})))
    return {"instance_id": instance.instance_id, "territory_id": instance.metadata.get("service_territory_id", instance.region_id),
            "num_customers": instance.num_customers, "num_charging_stations": instance.num_charging_stations,
            "saved_time_unit": instance.metadata.get("saved_time_unit"), "coordinate_unit": "km (dataset generator convention)",
            "resources": resources, "physical_checks": physical, "context_finite": context_finite,
            "fast_slow_initial_mismatches": mismatches, "legacy_noncoordinate_changes": other_changed,
            "single_action_dynamics": step, "pass": bool(passed)}, arrays


def run_audit(dataset_root, length_unit, reward_unit):
    cohorts = []
    for problem in ("cvrp", "vrptw", "evrptw"):
        for split in ("train", "val"):
            for size in (15, 50, 100):
                directory = dataset_root / problem / split / f"Cus{size}"
                if not (directory / "instances.pkl").exists():
                    cohorts.append({"dataset": str(directory), "status": "missing"})
                    continue
                count = metadata_count(directory)
                positions = {0, count // 2, count - 1}
                rows, features = [], {}
                for position, payload in enumerate(iter_instance_payloads(directory)):
                    if position not in positions:
                        continue
                    result, arrays = check_instance(payload, problem, length_unit, reward_unit)
                    rows.append({"bundle_index": position, **result})
                    for key, value in arrays.items():
                        features.setdefault(key, []).append(np.asarray(value).reshape(-1))
                    if len(rows) == len(positions):
                        break
                complete = {row["bundle_index"] for row in rows} == positions
                entry = {"dataset": str(directory), "problem": problem, "split": split, "num_customers": size,
                         "declared_count": count, "requested_indices": sorted(positions), "sample_count": len(rows),
                         "complete_requested_sample": complete, "instances": rows,
                         "feature_quantiles": {key: quantiles(np.concatenate(values)) for key, values in features.items()},
                         "pass": complete and all(row["pass"] for row in rows)}
                cohorts.append(entry)
                print(f'{problem}/{split}/Cus{size}: {len(rows)} sampled, pass={entry["pass"]}', flush=True)
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "AAAI_Dataset/dataset")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--distance-unit", type=float, default=43.638668060302734)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results/audit/input_normalization_20261007.json")
    args = parser.parse_args()
    unit, reward_unit = args.distance_unit, args.distance_unit
    provenance = {"distance_unit_source": "CLI/default fixed 43.638668060302734 km from VRPTW100 epoch300 init"}
    if args.source_config:
        raw = args.source_config.read_bytes()
        cfg = yaml.safe_load(raw)
        unit = cfg["env"]["observation_distance_scale_km"]
        reward_unit = cfg["env"]["reward_distance_scale_km"]
        provenance = {"distance_unit_source": str(args.source_config.resolve()), "config_sha256": hashlib.sha256(raw).hexdigest()}
    signature = input_normalization_signature({"env": {"observation_coordinate_mode": "depot_fixed",
        "observation_input_context": True, "observation_distance_scale_km": unit}})
    cohorts = run_audit(args.dataset_root, float(unit), float(reward_unit))
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "first/middle/last per available train/val task/size; input audit only, not full data or policy-quality validation",
              "signature": signature, "provenance": provenance, "cohorts": cohorts,
              "checked_instances": sum(row.get("sample_count", 0) for row in cohorts),
              "pass": all(row.get("pass", row.get("status") == "missing") for row in cohorts)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f'Report: {args.output}, checked={report["checked_instances"]}, pass={report["pass"]}', flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
