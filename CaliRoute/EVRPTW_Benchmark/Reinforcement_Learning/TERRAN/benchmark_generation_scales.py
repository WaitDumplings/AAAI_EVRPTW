from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from .async_instances import AsyncInstancePool
from .trainer import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _generator_root() -> Path:
    return _repo_root() / "EVRPTW_Dataset_Generator"


def _make_fixed_board_config(base_config_path: str | Path, output_dir: Path) -> Path:
    path = Path(base_config_path)
    if not path.is_absolute():
        path = _generator_root() / path
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Benchmark active-day sampling against a fixed service-territory pool.
    # Disable freshness replacement so the timing is not contaminated by
    # occasional region regeneration after high customer exposure.
    freshness = dict(cfg.get("freshness", {}))
    freshness["use_exposure_rule"] = False
    freshness["use_jaccard_rule"] = False
    cfg["freshness"] = freshness

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="fixed_board_benchmark_",
        dir=output_dir,
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return Path(tmp.name)


def _parse_scales(items: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in items:
        left, right = item.split(":", 1)
        out.append((int(left), int(right)))
    return out


def _run_scale(
    *,
    config_path: Path,
    num_customers: int,
    num_charging_stations: int,
    num_instances: int,
    warmup_instances: int,
    num_workers: int,
    queue_size: int,
    mother_num_customers: int,
    mother_num_charging_stations: int,
    region_reuse_limit: int,
    seed: int,
    report_every: int,
    territory_pool_path: str | Path | None,
    region_pool_path: str | Path | None,
) -> dict[str, Any]:
    pool = AsyncInstancePool(
        config_path=config_path,
        num_regions=max(1, int(num_workers)),
        mother_num_customers=int(mother_num_customers),
        mother_num_charging_stations=int(mother_num_charging_stations),
        num_customers=int(num_customers),
        num_charging_stations=int(num_charging_stations),
        region_reuse_limit=int(region_reuse_limit),
        seed=int(seed),
        max_attempts_per_instance=None,
        num_workers=int(num_workers),
        queue_size=int(queue_size),
        regions_per_worker=1,
        multiprocessing_context="spawn",
        territory_pool_path=territory_pool_path,
        region_pool_path=region_pool_path,
    )

    start_s = time.perf_counter()
    pool.start()
    start_time_s = time.perf_counter() - start_s
    try:
        warmup_s0 = time.perf_counter()
        for _ in range(int(warmup_instances)):
            _ = pool.sample()
        warmup_s = time.perf_counter() - warmup_s0

        measured_s0 = time.perf_counter()
        last_report_s = measured_s0
        for idx in range(1, int(num_instances) + 1):
            instance = pool.sample()
            if len(instance.customers) != int(num_customers):
                raise RuntimeError(f"Bad instance customer count: expected {num_customers}, got {len(instance.customers)}")
            if report_every > 0 and idx % report_every == 0:
                now = time.perf_counter()
                elapsed = now - measured_s0
                window = now - last_report_s
                print(
                    f"scale=Cus{num_customers}_CS{num_charging_stations} "
                    f"generated={idx}/{num_instances} "
                    f"elapsed_s={elapsed:.3f} "
                    f"window_s={window:.3f}",
                    flush=True,
                )
                last_report_s = now
        measured_s = time.perf_counter() - measured_s0
        wait_s = float(pool.total_wait_time_s)
    finally:
        pool.close(terminate=True)

    return {
        "num_customers": int(num_customers),
        "num_charging_stations": int(num_charging_stations),
        "num_instances": int(num_instances),
        "warmup_instances": int(warmup_instances),
        "num_workers": int(num_workers),
        "queue_size": int(queue_size),
        "mother_num_customers": int(mother_num_customers),
        "mother_num_charging_stations": int(mother_num_charging_stations),
        "pool_start_s": float(start_time_s),
        "warmup_s": float(warmup_s),
        "warmup_per_instance_s": float(warmup_s / max(int(warmup_instances), 1)),
        "measured_generation_s": float(measured_s),
        "avg_generation_per_instance_s": float(measured_s / max(int(num_instances), 1)),
        "instances_per_second": float(int(num_instances) / max(measured_s, 1e-12)),
        "queue_wait_s": wait_s,
        "avg_queue_wait_per_instance_s": float(wait_s / max(int(num_instances) + int(warmup_instances), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-territory online instance generation across customer scales.")
    parser.add_argument("--config", default="cus100_terran.yaml")
    parser.add_argument("--num-instances", type=int, default=10_000)
    parser.add_argument("--warmup-instances", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--mother-num-customers", type=int, default=10_000)
    parser.add_argument("--mother-num-charging-stations", type=int, default=120)
    parser.add_argument("--region-reuse-limit", type=int, default=1_000_000)
    parser.add_argument("--territory-pool-path", "--region-pool-path", dest="territory_pool_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["5:3", "15:3", "50:10", "100:20", "1000:120"],
        help="Customer:CS pairs, e.g. 100:20.",
    )
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/logs/fixed_territory_generation_scale_benchmark.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    output = Path(args.output)
    if not output.is_absolute():
        output = _repo_root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_cfg = _make_fixed_board_config(data_cfg.get("generator_config", "configs/amazon_hierarchy.yaml"), output.parent)

    rows = []
    for offset, (num_customers, num_cs) in enumerate(_parse_scales(args.scales)):
        row = _run_scale(
            config_path=fixed_cfg,
            num_customers=num_customers,
            num_charging_stations=num_cs,
            num_instances=int(args.num_instances),
            warmup_instances=int(args.warmup_instances),
            num_workers=int(args.num_workers),
            queue_size=int(args.queue_size),
            mother_num_customers=int(args.mother_num_customers),
            mother_num_charging_stations=int(args.mother_num_charging_stations),
            region_reuse_limit=int(args.region_reuse_limit),
            seed=int(args.seed) + offset * 10_000,
            report_every=int(args.report_every),
            territory_pool_path=args.territory_pool_path,
            region_pool_path=None,
        )
        rows.append(row)
        write_header = not output.exists()
        with output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)

    print(f"output={output}")
    print(f"fixed_config={fixed_cfg}")


if __name__ == "__main__":
    main()
