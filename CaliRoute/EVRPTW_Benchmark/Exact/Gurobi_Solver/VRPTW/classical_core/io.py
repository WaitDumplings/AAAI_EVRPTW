from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

from .schema import ClassicalVRPInstance, ClassicalVRPSolution

INSTANCE_BUNDLE_FORMAT = "classical_vrp_instance_bundle_v1"


def _install_numpy_pickle_compat() -> None:
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
                yield payload
            return
        yield first


def iter_instances(dataset_path: str | Path) -> Iterator[ClassicalVRPInstance]:
    root = Path(dataset_path)
    if root.is_file():
        paths = [root]
    else:
        direct = root / "instances.pkl"
        paths = [direct] if direct.exists() else sorted(root.glob("**/instances.pkl"))
    for path in paths:
        yield from (ClassicalVRPInstance.from_dict(payload) for payload in _iter_instance_dicts_from_file(path))


def save_solution(path: str | Path, solution: ClassicalVRPSolution) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(solution, f, protocol=pickle.HIGHEST_PROTOCOL)
