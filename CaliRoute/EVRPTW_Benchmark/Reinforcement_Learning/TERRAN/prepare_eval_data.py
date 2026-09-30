from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed TERRAN eval instances.")
    parser.add_argument("--config-path", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--save-path", type=Path, default=REPO_ROOT / "results/generated_data/terran_eval")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--num-customers", type=int, default=15)
    parser.add_argument("--num-charging-stations", type=int, default=3)
    parser.add_argument("--num-regions", type=int, default=8)
    parser.add_argument("--mother-num-customers", type=int, default=5000)
    parser.add_argument("--mother-num-charging-stations", type=int, default=120)
    parser.add_argument("--region-reuse-limit", type=int, default=200)
    parser.add_argument("--territory-pool-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = HierarchyDatasetGenerator.from_config_path(args.config_path, seed=args.seed)
    if args.territory_pool_path is not None:
        generator.load_region_pool(args.territory_pool_path, num_regions=args.num_regions, shuffle=True, replacement_policy="cycle")
    summary = generator.generate(
        save_path=args.save_path,
        num_instances=args.num_instances,
        num_customers=args.num_customers,
        num_charging_stations=args.num_charging_stations,
        num_regions=args.num_regions,
        mother_num_customers=args.mother_num_customers,
        mother_num_charging_stations=args.mother_num_charging_stations,
        region_reuse_limit=args.region_reuse_limit,
        save_plots=not args.no_plots,
        save_regions=args.territory_pool_path is None,
        dataset_metadata={
            "dataset_family": "EVRPTW-D",
            "dataset_version": "AC-v1",
            "calibration_profile": "Amazon-Calibrated",
            "suite_name": f"AC_Custom_{args.num_customers}",
            "service_territory_pool_path": str(args.territory_pool_path or ""),
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
