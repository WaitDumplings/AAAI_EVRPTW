from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

from evrptw_core.schema import EVRPTWInstance, EVRPTWSolution

INSTANCE_BUNDLE_FORMAT = "evrptw_instance_bundle_v1"


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 2 pickles to load in NumPy 1.x environments."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "umath", "numeric", "fromnumeric", "shape_base", "_multiarray_umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=["*"])
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


_install_numpy_pickle_compat()


def _load_pickle_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected pickle dict at {path}, got {type(data)!r}")
    return data


def _iter_instance_dicts_from_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as f:
        first = pickle.load(f)
        if not isinstance(first, dict):
            raise TypeError(f"Expected pickle dict at {path}, got {type(first)!r}")
        if first.get("format") == INSTANCE_BUNDLE_FORMAT:
            count = first.get("num_instances")
            read = 0
            while count is None or read < int(count):
                try:
                    payload = pickle.load(f)
                except EOFError:
                    break
                if not isinstance(payload, dict):
                    raise TypeError(f"Bad instance payload in {path}: {type(payload)!r}")
                yield payload
                read += 1
            return
        if "instances" in first and isinstance(first["instances"], list):
            for payload in first["instances"]:
                if not isinstance(payload, dict):
                    raise TypeError(f"Bad instance payload in {path}: {type(payload)!r}")
                yield payload
            return
        yield first


def instance_bundle_path(dataset_path: str | Path, num_customers: int | None = None, num_charging_stations: int | None = None) -> Path | None:
    root = Path(dataset_path)
    if root.is_file():
        return root
    direct = root / "instances.pkl"
    if direct.exists():
        return direct
    if num_customers is not None and num_charging_stations is not None:
        nested = root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}" / "instances.pkl"
        if nested.exists():
            return nested
    return None


def iter_instance_dicts(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
) -> Iterator[dict[str, Any]]:
    root = Path(dataset_path)
    bundle = instance_bundle_path(root, num_customers, num_charging_stations)
    if bundle is not None:
        yield from _iter_instance_dicts_from_file(bundle)
        return
    if root.is_file():
        yield from _iter_instance_dicts_from_file(root)
        return
    if num_customers is not None and num_charging_stations is not None:
        nested = root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}"
        search_root = nested if nested.exists() else root
    else:
        search_root = root / "instances" if (root / "instances").exists() else root
    bundle_paths = sorted(search_root.glob("**/instances.pkl"))
    instance_paths = sorted(search_root.glob("**/instance_*.pkl"))
    for path in [*bundle_paths, *instance_paths]:
        yield from _iter_instance_dicts_from_file(path)


def iter_instances(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
) -> Iterator[EVRPTWInstance]:
    for payload in iter_instance_dicts(dataset_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        instance = EVRPTWInstance.from_dict(payload)
        if num_customers is not None and instance.num_customers != int(num_customers):
            continue
        if num_charging_stations is not None and instance.num_charging_stations != int(num_charging_stations):
            continue
        yield instance


def load_instances(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
    limit: int | None = None,
) -> list[EVRPTWInstance]:
    out: list[EVRPTWInstance] = []
    for instance in iter_instances(dataset_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        out.append(instance)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def load_instance(path: str | Path) -> EVRPTWInstance:
    iterator = iter_instances(path)
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(f"No EVRPTW instance found in {path}") from exc


def load_solution(path: str | Path) -> EVRPTWSolution:
    return EVRPTWSolution.from_dict(_load_pickle_dict(path))


def save_solution(path: str | Path, solution: EVRPTWSolution) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(solution.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return out
