from __future__ import annotations

from pathlib import Path

CALIROUTE_ROOT = Path(__file__).resolve().parents[3]
DATASET_PACKAGE_ROOT = CALIROUTE_ROOT.parent / "AAAI_Dataset"


def resolve_run_paths(
    *,
    problem: str,
    split: str,
    scale: str,
    dataset_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve defaults from the checkout; explicit paths use the caller's cwd.

    dataset_root is problem-specific and contains split/CusN directories.
    Frozen test inputs live in test_release; all default outputs live in results.
    """
    problem = problem.lower()
    if problem not in {"cvrp", "vrptw", "evrptw"}:
        raise ValueError(f"Unsupported problem: {problem}")
    if split not in {"train", "val", "test", "eval"}:
        raise ValueError(f"Unsupported split: {split}")
    scale_text = str(scale).lower().removeprefix("cus")
    customers = int(scale_text)
    if customers <= 0:
        raise ValueError("Customer count must be positive")
    scale = f"Cus{customers}"
    if dataset_path:
        source = Path(dataset_path)
    else:
        release = "test_release" if split == "test" else "dataset"
        data_root = Path(dataset_root) if dataset_root else DATASET_PACKAGE_ROOT / release / problem
        source = data_root / split / scale
    destination = (
        Path(output_path)
        if output_path
        else CALIROUTE_ROOT / "results" / "gurobi" / problem / split / scale
    )
    return source.expanduser().resolve(), destination.expanduser().resolve()
