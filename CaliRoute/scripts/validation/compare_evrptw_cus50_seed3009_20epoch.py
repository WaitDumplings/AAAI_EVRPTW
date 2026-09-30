from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
NEW_ROOT = ROOT / "results" / "logs" / "Cus_50_CS_10"
OLD_ROOTS = (
    ROOT / "results_legacy" / "logs" / "Cus_50_CS_10",
)
OUT_DIR = ROOT / "results" / "validation" / "evrptw_cus50_seed3009_20epoch"
METHODS = ("ppo", "slppo", "dapg", "awbc")
OLD_RUN_NAMES = {
    "ppo": "O2O_CUS50_OFFLINE_JUDGE_PPO_ROUTE_POS_SEED3009_E1500",
    "slppo": "O2O_CUS50_SL_PPO_ROUTE_POS_SEED3009_E1500",
    "dapg": "O2O_CUS50_OFFLINE_JUDGE_DAPG_ROUTE_POS_SEED3009_E1500",
    "awbc": "O2O_CUS50_AWBC_ROUTE_POS_SEED3009_E1500",
}


def _read_last_eval(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    epoch20 = [r for r in rows if str(r.get("epoch", "")).strip() == "20"]
    return epoch20[-1] if epoch20 else rows[-1]


def _find_new(method: str, root: Path) -> Path | None:
    path = root / f"CALIROUTE_VALIDATE_EVRPTW_CUS50_{method.upper()}_SEED3009_E20" / "seed_3009" / "eval_log.csv"
    return path if path.exists() else None


def _find_old(method: str, roots: tuple[Path, ...]) -> Path | None:
    run_name = OLD_RUN_NAMES.get(method)
    if run_name:
        for root in roots:
            path = root / run_name / "seed_3009" / "eval_log.csv"
            if path.exists():
                return path
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.glob("*/seed_3009/eval_log.csv")
        if _old_name_matches(path.parts[-3].lower(), method)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _old_name_matches(name: str, method: str) -> bool:
    if "cus50" not in name or "seed3009" not in name:
        return False
    if method == "ppo":
        return "ppo" in name and "sl_ppo" not in name and "awbc" not in name and "dapg" not in name
    if method == "slppo":
        return "sl_ppo" in name or "slppo" in name
    return method in name


def _metric(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return ""
    aliases = {
        "min_obj": ("min_obj", "eval_avg_min_objective_distance_km", "eval_avg_objective_distance_km"),
        "min_veh": ("min_veh", "eval_avg_min_vehicle_count", "eval_avg_vehicle_count"),
        "med_obj": ("med_obj", "eval_avg_median_objective_distance_km"),
        "med_veh": ("med_veh", "eval_avg_median_vehicle_count"),
        "fr": ("fr", "eval_feasible_rate"),
        "epoch": ("epoch",),
    }
    value = ""
    for candidate in aliases.get(key, (key,)):
        value = row.get(candidate, "")
        if value not in (None, ""):
            break
    return "" if value is None else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Cus50 seed-3009 validation logs with optional historical runs.")
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT, help="Directory containing the new Cus50 run folders.")
    parser.add_argument(
        "--old-root", type=Path, action="append", default=None,
        help="Directory containing historical Cus50 run folders; repeat to search multiple directories. "
        "Defaults to CaliRoute/results_legacy/logs/Cus_50_CS_10.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory to write comparison.csv.")
    args = parser.parse_args()
    args.new_root = args.new_root.expanduser().resolve()
    args.old_roots = tuple(path.expanduser().resolve() for path in (args.old_root or OLD_ROOTS))
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.new_root.is_dir():
        parser.error(f"New log directory does not exist: {args.new_root}. Run validation first or pass --new-root.")
    for root in args.old_roots:
        if not root.is_dir():
            print(
                f"Warning: historical log directory does not exist: {root}. "
                "Pass --old-root to use your archived logs; unavailable comparison fields stay empty.",
                file=sys.stderr,
            )
    return args


def main() -> None:
    args = parse_args()
    out_path = args.out_dir / "comparison.csv"
    fields = [
        "method",
        "new_eval_path",
        "old_eval_path",
        "new_epoch",
        "new_min_obj",
        "new_min_veh",
        "new_med_obj",
        "new_med_veh",
        "new_fr",
        "old_epoch",
        "old_min_obj",
        "old_min_veh",
        "old_med_obj",
        "old_med_veh",
        "old_fr",
        "delta_min_obj_new_minus_old",
    ]
    rows = []
    for method in METHODS:
        new_path = _find_new(method, args.new_root)
        old_path = _find_old(method, args.old_roots)
        new_row = _read_last_eval(new_path) if new_path else None
        old_row = _read_last_eval(old_path) if old_path else None
        if new_row is None:
            print(f"Warning: no new evaluation rows found for {method} under {args.new_root}.", file=sys.stderr)
        if old_row is None:
            print(f"Warning: no historical evaluation rows found for {method}; historical metrics stay empty.", file=sys.stderr)
        delta = ""
        try:
            if new_row and old_row:
                delta = f"{float(_metric(new_row, 'min_obj')) - float(_metric(old_row, 'min_obj')):.6f}"
        except (TypeError, ValueError):
            delta = ""
        rows.append(
            {
                "method": method,
                "new_eval_path": str(new_path or ""),
                "old_eval_path": str(old_path or ""),
                "new_epoch": _metric(new_row, "epoch"),
                "new_min_obj": _metric(new_row, "min_obj"),
                "new_min_veh": _metric(new_row, "min_veh"),
                "new_med_obj": _metric(new_row, "med_obj"),
                "new_med_veh": _metric(new_row, "med_veh"),
                "new_fr": _metric(new_row, "fr"),
                "old_epoch": _metric(old_row, "epoch"),
                "old_min_obj": _metric(old_row, "min_obj"),
                "old_min_veh": _metric(old_row, "min_veh"),
                "old_med_obj": _metric(old_row, "med_obj"),
                "old_med_veh": _metric(old_row, "med_veh"),
                "old_fr": _metric(old_row, "fr"),
                "delta_min_obj_new_minus_old": delta,
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
