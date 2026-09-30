from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any

from .async_instances import AsyncInstancePool
from .data_pool import OnlineInstancePool
from .trainer import load_config


def pool_kwargs(cfg: dict[str, Any], seed: int, args: argparse.Namespace) -> dict[str, Any]:
    data = cfg["data"]
    return dict(
        config_path=data.get("generator_config", "configs/amazon_hierarchy.yaml"),
        num_regions=int(args.mother_board_pool_size or data.get("mother_board_pool_size", 8)),
        mother_num_customers=int(args.mother_num_customers or data.get("mother_num_customers", 5000)),
        mother_num_charging_stations=int(args.mother_num_charging_stations or data.get("mother_num_charging_stations", 120)),
        num_customers=int(args.num_customers or data.get("num_customers", 100)),
        num_charging_stations=int(args.num_charging_stations or data.get("num_charging_stations", 20)),
        region_reuse_limit=int(data.get("region_reuse_limit", 200)),
        seed=seed,
        max_attempts_per_instance=data.get("max_attempts_per_instance"),
        territory_pool_path=args.territory_pool_path or data.get("territory_pool_path"),
        region_pool_path=data.get("region_pool_path"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sequential vs async online instance generation.")
    parser.add_argument("--config", default="cus100_terran.yaml")
    parser.add_argument("--num-instances", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--queue-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--num-customers", type=int, default=None)
    parser.add_argument("--num-charging-stations", type=int, default=None)
    parser.add_argument("--mother-board-pool-size", "--service-territory-pool-size", dest="mother_board_pool_size", type=int, default=None)
    parser.add_argument("--mother-num-customers", type=int, default=None)
    parser.add_argument("--mother-num-charging-stations", type=int, default=None)
    parser.add_argument("--territory-pool-path", type=str, default=None)
    parser.add_argument("--output", default="EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/logs/instance_prefetch_benchmark.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = int(args.num_instances)

    seq_start = time.perf_counter()
    seq_pool = OnlineInstancePool(**pool_kwargs(cfg, args.seed, args))
    seq_init_s = time.perf_counter() - seq_start
    seq_sample_start = time.perf_counter()
    seq_instances = [seq_pool.sample() for _ in range(n)]
    seq_sample_s = time.perf_counter() - seq_sample_start
    del seq_instances

    async_kwargs = pool_kwargs(cfg, args.seed + 10_000, args)
    async_start = time.perf_counter()
    async_pool = AsyncInstancePool(
        **async_kwargs,
        num_workers=int(args.num_workers),
        queue_size=max(int(args.num_workers) * 2, n * int(args.queue_batches)),
    )
    async_pool.start()
    async_start_s = time.perf_counter() - async_start
    try:
        async_warm_s = async_pool.warmup(n)
        async_pop_start = time.perf_counter()
        async_instances = [async_pool.sample() for _ in range(n)]
        async_pop_s = time.perf_counter() - async_pop_start
        del async_instances
        async_refill_s = async_pool.warmup(n)
        async_pop2_start = time.perf_counter()
        async_instances = [async_pool.sample() for _ in range(n)]
        async_pop2_s = time.perf_counter() - async_pop2_start
        del async_instances
    finally:
        async_pool.close(terminate=True)

    row = {
        "config": args.config,
        "num_instances": n,
        "num_workers": int(args.num_workers),
        "num_customers": int(async_kwargs["num_customers"]),
        "num_charging_stations": int(async_kwargs["num_charging_stations"]),
        "service_territory_pool_size": int(async_kwargs["num_regions"]),
        "seq_init_s": seq_init_s,
        "seq_sample_s": seq_sample_s,
        "seq_sample_per_instance_s": seq_sample_s / max(n, 1),
        "async_start_s": async_start_s,
        "async_warm_s": async_warm_s,
        "async_warm_per_instance_s": async_warm_s / max(n, 1),
        "async_pop_s": async_pop_s,
        "async_pop_per_instance_s": async_pop_s / max(n, 1),
        "async_refill_s": async_refill_s,
        "async_refill_per_instance_s": async_refill_s / max(n, 1),
        "async_pop2_s": async_pop2_s,
        "async_pop2_per_instance_s": async_pop2_s / max(n, 1),
        "raw_generation_speedup_seq_sample_over_async_warm": seq_sample_s / max(async_warm_s, 1e-12),
        "steady_generation_speedup_seq_sample_over_async_refill": seq_sample_s / max(async_refill_s, 1e-12),
        "reset_wait_speedup_seq_sample_over_async_pop": seq_sample_s / max(async_pop_s, 1e-12),
        "reset_wait_speedup_seq_sample_over_async_pop2": seq_sample_s / max(async_pop2_s, 1e-12),
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[3] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    for key, value in row.items():
        print(f"{key}={value}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
