from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from evrptw_hierarchy.configs.config import load_yaml, vehicle_from_config
from evrptw_hierarchy.core.models import RegionBoard, RegionUsage
from evrptw_hierarchy.generation.region_generator import RegionGenerator
from evrptw_hierarchy.io.persistence import ensure_dir, load_pickle, save_pickle
from evrptw_hierarchy.validation.reports import summarize_region

SERVICE_TERRITORY_POOL_FORMAT = "evrptw_service_territory_pool_v1"


def _fresh_usage(board: RegionBoard) -> RegionUsage:
    return RegionUsage(
        region_id=board.region_id,
        sampled_days=0,
        customer_activation_counts=np.zeros(len(board.customers), dtype=np.int32),
        cluster_activation_counts=np.zeros(len(board.cluster_centers), dtype=np.int32),
    )


def _board_from_payload(payload: Any) -> RegionBoard:
    if isinstance(payload, RegionBoard):
        return payload
    if isinstance(payload, dict):
        fields = RegionBoard.__dataclass_fields__
        kwargs = {key: payload[key] for key in fields if key in payload}
        return RegionBoard(**kwargs)
    raise TypeError(f"Unsupported territory payload type: {type(payload).__name__}")


def _generate_one(args: tuple[str, str, int, int, int, int]) -> dict[str, Any]:
    config_path, shard_dir_raw, region_index, latent_customers, cs_candidates, base_seed = args
    shard_dir = Path(shard_dir_raw)
    path = shard_dir / f"region_{region_index:03d}_board.pkl"
    if path.exists():
        board = _board_from_payload(load_pickle(path))
        row = summarize_region(board, _fresh_usage(board))
        return {"region_id": board.region_id, "path": str(path), "status": "existing", "region_row": row}

    cfg = load_yaml(config_path)
    vehicle = vehicle_from_config(cfg)
    rng = np.random.default_rng(int(base_seed) + int(region_index) * 104_729)
    generator = RegionGenerator(cfg, vehicle, rng)
    board = generator.generate(int(region_index), int(latent_customers), int(cs_candidates))
    save_pickle(path, board)
    row = summarize_region(board, _fresh_usage(board))
    return {"region_id": board.region_id, "path": str(path), "status": "generated", "region_row": row}


def generate_service_territory_pool(
    *,
    config_path: str | Path,
    save_path: str | Path,
    num_territories: int,
    latent_customer_pool_size: int,
    cs_candidate_pool_size: int,
    seed: int | None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Generate a reusable service-territory pool incrementally.

    Existing ``region_*_board.pkl`` files are reused, so interrupted pool
    generation can be resumed safely. Each worker writes one territory pickle and
    returns only summary metadata, avoiding a large parent-process memory spike.
    """
    root = ensure_dir(save_path)
    bundle_path = root / "service_territory_pool.pkl"
    if bundle_path.exists():
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if int(manifest.get("num_service_territories", -1)) >= int(num_territories):
                manifest["pool_path"] = str(root)
                return manifest
    shard_dir = ensure_dir(root / "_territory_shards")
    config_path = str(Path(config_path).resolve())
    seed_value = 0 if seed is None else int(seed)
    workers = int(num_workers or min(8, max(1, os.cpu_count() or 1)))
    workers = max(1, min(workers, int(num_territories)))

    jobs = [
        (config_path, str(shard_dir), idx, int(latent_customer_pool_size), int(cs_candidate_pool_size), seed_value)
        for idx in range(int(num_territories))
    ]
    rows: list[dict[str, Any]] = []
    generated = 0
    existing = 0

    if workers == 1:
        for job in jobs:
            result = _generate_one(job)
            rows.append(result)
            generated += int(result["status"] == "generated")
            existing += int(result["status"] == "existing")
            print(f"territory_pool progress {len(rows)}/{int(num_territories)} status={result['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                rows.append(result)
                generated += int(result["status"] == "generated")
                existing += int(result["status"] == "existing")
                if len(rows) % max(1, int(num_territories) // 20) == 0 or len(rows) == int(num_territories):
                    print(f"territory_pool progress {len(rows)}/{int(num_territories)} generated={generated} existing={existing}", flush=True)

    rows.sort(key=lambda row: row["region_id"])
    region_rows = [row["region_row"] for row in rows]
    header = {
        "format": SERVICE_TERRITORY_POOL_FORMAT,
        "dataset_family": "EVRPTW-D",
        "dataset_version": "AC-v1",
        "calibration_profile": "Amazon-Calibrated",
        "num_service_territories": int(num_territories),
        "latent_customer_pool_size": int(latent_customer_pool_size),
        "cs_candidate_pool_size": int(cs_candidate_pool_size),
        "seed": seed,
    }
    with bundle_path.open("wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for row in rows:
            payload = load_pickle(row["path"])
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "AC-v1",
        "calibration_profile": "Amazon-Calibrated",
        "pool_type": "service_territory_pool",
        "pool_path": str(root),
        "bundle_path": str(bundle_path),
        "format": SERVICE_TERRITORY_POOL_FORMAT,
        "num_service_territories": int(num_territories),
        "num_regions": int(num_territories),
        "latent_customer_pool_size": int(latent_customer_pool_size),
        "cs_candidate_pool_size": int(cs_candidate_pool_size),
        "mother_num_customers": int(latent_customer_pool_size),
        "mother_num_charging_stations": int(cs_candidate_pool_size),
        "seed": seed,
        "num_workers": workers,
        "generated_count": generated,
        "existing_count": existing,
        "service_territory_ids": [row["region_id"] for row in rows],
        "region_ids": [row["region_id"] for row in rows],
        "region_rows": region_rows,
        "terminology": {
            "service_territory_graph": "Stable city/region/delivery-station service territory.",
            "legacy_internal_fields": ["mother_board_id"],
        },
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return manifest
