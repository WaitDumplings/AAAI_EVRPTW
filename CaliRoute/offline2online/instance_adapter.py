from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Any, Iterator

import numpy as np

from .integrations.evrptw_db import configure_evrptw_db

EVRPTW_DB_ROOT = configure_evrptw_db()

from evrptw_core.schema import EVRPTWInstance


REPO_ROOT = Path(__file__).resolve().parents[1]
EVRPTW_BUNDLE_FORMAT = "evrptw_instance_bundle_v1"
CLASSICAL_BUNDLE_FORMAT = "classical_vrp_instance_bundle_v1"
CLASSICAL_PROBLEM_TYPES = {"cvrp", "vrptw", "cvrptw"}
DEFAULT_CLASSICAL_SPEED_KMH = 3600.0
DEFAULT_NON_BINDING_HORIZON_S = 1_000_000
DEFAULT_BATTERY_CAPACITY_KWH = 1.0


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


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    if str(path).startswith("EVRPTW_DB_ROOT/"):
        return EVRPTW_DB_ROOT / str(path).split("/", 1)[1]
    return REPO_ROOT / out


def normalize_problem_type(value: Any | None, *, default: str | None = "evrptw") -> str | None:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "none", "auto"}:
        return default
    text = text.replace("-", "_").replace(" ", "_")
    if "evrptw" in text or "evrp_tw" in text or "geo_ac" in text:
        return "evrptw"
    if "cvrptw" in text:
        return "cvrptw"
    if "vrptw" in text:
        return "vrptw"
    if "cvrp" in text:
        return "cvrp"
    if text in {"evrptw_d", "evrp_tw_d", "evrp"}:
        return "evrptw"
    return default


def problem_type_from_config(cfg: dict[str, Any] | None, *, default: str = "evrptw") -> str:
    cfg = cfg or {}
    data_cfg = cfg.get("data", {}) or {}
    offline_cfg = cfg.get("offline", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    candidates = [
        data_cfg.get("problem_type"),
        data_cfg.get("problem_class"),
        data_cfg.get("problem"),
        cfg.get("problem_type"),
        cfg.get("problem_class"),
        cfg.get("dataset_name"),
        data_cfg.get("train_dataset_path"),
        data_cfg.get("instance_dataset_path"),
        data_cfg.get("fixed_train_path"),
        eval_cfg.get("eval_path"),
        offline_cfg.get("expert_dataset_path"),
    ]
    for candidate in candidates:
        problem_type = normalize_problem_type(candidate, default=None)
        if problem_type is not None:
            return problem_type
    return default


def num_charging_stations_for_problem(
    data_cfg: dict[str, Any] | None,
    problem_type: str,
    *,
    evrptw_default: int = 3,
) -> int:
    data_cfg = data_cfg or {}
    configured = data_cfg.get("num_charging_stations")
    if problem_type in CLASSICAL_PROBLEM_TYPES:
        if configured not in (None, "", 0, "0"):
            raise ValueError(
                f"problem_type={problem_type!r} is adapted as a no-charging-station instance; "
                "set data.num_charging_stations to 0 or omit it."
            )
        return 0
    return int(configured if configured not in (None, "") else evrptw_default)


def _instance_bundle_path(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
) -> Path | None:
    root = resolve_repo_path(dataset_path)
    if root.is_file():
        return root
    direct = root / "instances.pkl"
    if direct.exists():
        return direct
    if num_customers is not None and num_charging_stations is not None:
        nested = root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}" / "instances.pkl"
        if nested.exists():
            return nested
    if num_customers is not None:
        for name in (f"Cus{int(num_customers)}", f"Cus_{int(num_customers)}"):
            nested = root / name / "instances.pkl"
            if nested.exists():
                return nested
    return None


def _iter_payloads_from_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as f:
        first = pickle.load(f)
        if not isinstance(first, dict):
            raise TypeError(f"Expected pickle dict at {path}, got {type(first)!r}")
        if first.get("format") in {EVRPTW_BUNDLE_FORMAT, CLASSICAL_BUNDLE_FORMAT}:
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


def iter_instance_payloads(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
) -> Iterator[dict[str, Any]]:
    root = resolve_repo_path(dataset_path)
    bundle = _instance_bundle_path(
        root,
        num_customers=num_customers,
        num_charging_stations=num_charging_stations,
    )
    if bundle is not None:
        yield from _iter_payloads_from_file(bundle)
        return
    if root.is_file():
        yield from _iter_payloads_from_file(root)
        return
    if num_customers is not None and num_charging_stations is not None:
        nested = root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}"
        search_root = nested if nested.exists() else root
    else:
        search_root = root / "instances" if (root / "instances").exists() else root
    for path in [*sorted(search_root.glob("**/instances.pkl")), *sorted(search_root.glob("**/instance_*.pkl"))]:
        yield from _iter_payloads_from_file(path)


def _problem_type_from_payload(payload: dict[str, Any], fallback: str | None) -> str:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    candidates = [
        fallback,
        payload.get("problem_type"),
        payload.get("problem_class"),
        metadata.get("problem_type"),
        metadata.get("problem_class"),
    ]
    for candidate in candidates:
        problem_type = normalize_problem_type(candidate, default=None)
        if problem_type is not None:
            return problem_type
    return "evrptw"


def _array(data: Any, *, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(data, dtype=dtype)


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def _coords_from_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    depot = _first_present(payload, ("depot", "depot_loc", "depot_location"))
    customers = _first_present(payload, ("customers", "customer_locs", "customer_locations"))
    coords = _first_present(payload, ("coords", "locations", "node_locations"))
    if customers is None and coords is not None:
        coords_arr = _array(coords)
        if coords_arr.ndim != 2 or coords_arr.shape[0] < 2:
            raise ValueError("coords/locations must have shape (1 + num_customers, 2)")
        if depot is None:
            depot = coords_arr[0]
        customers = coords_arr[1:]
    if depot is None or customers is None:
        raise KeyError("classical adapter requires depot and customers/coords")
    depot_arr = _array(depot).reshape(2)
    customers_arr = _array(customers)
    if customers_arr.ndim != 2 or customers_arr.shape[1] != 2:
        raise ValueError(f"customers must have shape (num_customers, 2), got {customers_arr.shape}")
    return depot_arr, customers_arr


def _customer_vector(payload: dict[str, Any], names: tuple[str, ...], n: int, *, dtype: Any, default: float | int | None = None) -> np.ndarray:
    value = _first_present(payload, names)
    if value is None:
        if default is None:
            raise KeyError(f"missing required customer vector: {names[0]}")
        return np.full(n, default, dtype=dtype)
    arr = _array(value, dtype=dtype).reshape(-1)
    if arr.shape[0] == n + 1:
        arr = arr[1:]
    if arr.shape[0] != n:
        raise ValueError(f"{names[0]} must have length {n} or {n + 1}, got {arr.shape[0]}")
    return arr


def _customer_matrix(payload: dict[str, Any], names: tuple[str, ...], n: int, *, default: tuple[float, float] | None = None) -> np.ndarray:
    value = _first_present(payload, names)
    if value is None:
        if default is None:
            raise KeyError(f"missing required customer matrix: {names[0]}")
        return np.tile(np.asarray(default, dtype=np.float32).reshape(1, 2), (n, 1))
    arr = _array(value).reshape(-1, 2)
    if arr.shape[0] == n + 1:
        arr = arr[1:]
    if arr.shape[0] != n:
        raise ValueError(f"{names[0]} must have {n} or {n + 1} rows, got {arr.shape[0]}")
    return arr


def _distance_matrix(payload: dict[str, Any], depot: np.ndarray, customers: np.ndarray) -> np.ndarray:
    n = int(customers.shape[0])
    value = _first_present(payload, ("distance_matrix_km", "distance_matrix", "distances", "distance"))
    if value is None:
        coords = np.vstack([depot.reshape(1, 2), customers])
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    dist = _array(value)
    expected = n + 1
    if dist.shape == (expected, expected):
        return dist
    if dist.ndim == 2 and dist.shape[0] >= expected and dist.shape[1] >= expected:
        indices = np.arange(expected, dtype=np.int64)
        return dist[np.ix_(indices, indices)].astype(np.float32)
    raise ValueError(f"distance matrix must have shape ({expected}, {expected}) or larger, got {dist.shape}")


def _infer_speed_from_time_matrix(payload: dict[str, Any], distance_matrix_km: np.ndarray) -> float | None:
    time_matrix = _first_present(payload, ("travel_time_matrix_s", "time_matrix_s", "travel_time_matrix"))
    if time_matrix is None:
        return None
    times = _array(time_matrix, dtype=np.float64)
    if times.shape != distance_matrix_km.shape:
        return None
    dist = np.asarray(distance_matrix_km, dtype=np.float64)
    mask = (times > 1e-9) & (dist > 1e-9) & np.isfinite(times) & np.isfinite(dist)
    if not np.any(mask):
        return None
    speed_kmh = np.median(dist[mask] / times[mask] * 3600.0)
    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        return None
    return float(speed_kmh)


def _working_window(payload: dict[str, Any], tw_s: np.ndarray | None, *, use_time_windows: bool) -> tuple[int, int]:
    if use_time_windows:
        start = payload.get("working_start_s")
        end = payload.get("working_end_s")
        if (start is None or end is None) and tw_s is not None and tw_s.size:
            finite = tw_s[np.isfinite(tw_s)]
            if finite.size:
                if start is None:
                    start = float(np.min(tw_s[:, 0]))
                if end is None:
                    end = float(np.max(tw_s[:, 1]))
        start_i = int(float(start if start is not None else 0))
        end_i = int(float(end if end is not None else start_i + DEFAULT_NON_BINDING_HORIZON_S))
        if end_i <= start_i:
            end_i = start_i + DEFAULT_NON_BINDING_HORIZON_S
        return start_i, end_i
    return 0, DEFAULT_NON_BINDING_HORIZON_S


def _classical_to_evrptw_dict(payload: dict[str, Any], problem_type: str) -> dict[str, Any]:
    depot, customers = _coords_from_payload(payload)
    n = int(customers.shape[0])
    distance_matrix_km = _distance_matrix(payload, depot, customers)
    demands_cm3 = _customer_vector(
        payload,
        ("demands_cm3", "demands", "demand", "customer_demands", "customer_demand"),
        n,
        dtype=np.float32,
    )
    package_counts = _customer_vector(
        payload,
        ("package_counts", "packages", "package_count"),
        n,
        dtype=np.int32,
        default=0,
    )

    use_time_windows = problem_type in {"vrptw", "cvrptw"}
    if use_time_windows:
        tw_s = _customer_matrix(payload, ("tw_s", "time_windows", "customer_time_windows", "tw"), n)
        service_time_s = _customer_vector(
            payload,
            ("service_time_s", "service_times_s", "service_times", "service_time"),
            n,
            dtype=np.float32,
            default=0.0,
        )
    else:
        tw_s = None
        service_time_s = np.zeros(n, dtype=np.float32)
    working_start_s, working_end_s = _working_window(payload, tw_s, use_time_windows=use_time_windows)
    if tw_s is None:
        tw_s = np.tile(np.array([working_start_s, working_end_s], dtype=np.float32).reshape(1, 2), (n, 1))

    vehicle = dict(payload.get("vehicle", {}) or {})
    capacity = (
        vehicle.get("cargo_capacity_cm3")
        or payload.get("cargo_capacity_cm3")
        or payload.get("vehicle_capacity")
        or payload.get("capacity")
        or payload.get("loading_capacity")
        or np.inf
    )
    speed_profile = dict(payload.get("speed_profile", {}) or {})
    inferred_speed = _infer_speed_from_time_matrix(payload, distance_matrix_km)
    effective_speed = (
        speed_profile.get("effective_speed_kmh")
        or payload.get("effective_speed_kmh")
        or inferred_speed
        or DEFAULT_CLASSICAL_SPEED_KMH
    )
    design_speed = speed_profile.get("design_speed_kmh") or payload.get("design_speed_kmh") or effective_speed
    speed_profile.setdefault("design_speed_kmh", float(design_speed))
    speed_profile.setdefault("effective_speed_kmh", float(effective_speed))
    vehicle["cargo_capacity_cm3"] = float(capacity)
    vehicle["battery_capacity_kwh"] = float(vehicle.get("battery_capacity_kwh", DEFAULT_BATTERY_CAPACITY_KWH))
    vehicle["consumption_kwh_per_km"] = 0.0
    vehicle["full_charge_time_s"] = 0.0
    vehicle["design_speed_kmh"] = float(design_speed)

    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "adapted_to_evrptw": True,
            "adapter_problem_type": problem_type,
            "source_problem_class": payload.get("problem_class", metadata.get("problem_class")),
            "charging_constraint": False,
            "time_window_constraint": bool(use_time_windows),
        }
    )
    return {
        "instance_id": str(payload.get("instance_id", metadata.get("instance_id", ""))),
        "region_id": str(payload.get("region_id", metadata.get("region_id", ""))),
        "mother_board_id": str(payload.get("mother_board_id", "")),
        "operating_day_id": str(payload.get("operating_day_id", "")),
        "day_type": str(payload.get("day_type", "classical")),
        "working_start_s": int(working_start_s),
        "working_end_s": int(working_end_s),
        "depot": depot.astype(np.float32),
        "customers": customers.astype(np.float32),
        "charging_stations": np.zeros((0, 2), dtype=np.float32),
        "distance_matrix_km": distance_matrix_km.astype(np.float32),
        "demands_cm3": demands_cm3.astype(np.float32),
        "package_counts": package_counts.astype(np.int32),
        "service_time_s": service_time_s.astype(np.float32),
        "tw_s": tw_s.astype(np.float32),
        "cs_time_to_depot_s": np.zeros((0,), dtype=np.float32),
        "vehicle": vehicle,
        "speed_profile": speed_profile,
        "cs_activation": {},
        "metadata": metadata,
    }


def adapt_instance_payload(payload: dict[str, Any], *, problem_type: str | None = None) -> EVRPTWInstance:
    resolved_problem = _problem_type_from_payload(payload, problem_type)
    if resolved_problem == "evrptw":
        return EVRPTWInstance.from_dict(payload)
    adapted = _classical_to_evrptw_dict(payload, resolved_problem)
    if not adapted["instance_id"]:
        adapted["instance_id"] = "adapted_instance"
    return EVRPTWInstance.from_dict(adapted)


def iter_adapted_instances(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
    problem_type: str | None = None,
    limit: int | None = None,
) -> Iterator[EVRPTWInstance]:
    yielded = 0
    for idx, payload in enumerate(
        iter_instance_payloads(
            dataset_path,
            num_customers=num_customers,
            num_charging_stations=num_charging_stations,
        )
    ):
        if "instance_id" not in payload or payload.get("instance_id") in (None, ""):
            payload = dict(payload)
            payload["instance_id"] = f"adapted_{idx:06d}"
        instance = adapt_instance_payload(payload, problem_type=problem_type)
        if num_customers is not None and instance.num_customers != int(num_customers):
            continue
        if num_charging_stations is not None and instance.num_charging_stations != int(num_charging_stations):
            continue
        yield instance
        yielded += 1
        if limit is not None and yielded >= int(limit):
            break


def load_adapted_instances(
    dataset_path: str | Path,
    *,
    num_customers: int | None = None,
    num_charging_stations: int | None = None,
    problem_type: str | None = None,
    limit: int | None = None,
) -> list[EVRPTWInstance]:
    return list(
        iter_adapted_instances(
            dataset_path,
            num_customers=num_customers,
            num_charging_stations=num_charging_stations,
            problem_type=problem_type,
            limit=limit,
        )
    )


@dataclass
class AdaptedFixedDatasetInstancePool:
    dataset_path: str | Path
    num_customers: int | None = None
    num_charging_stations: int | None = None
    seed: int | None = None
    sample_mode: str = "shuffle_cycle"
    problem_type: str | None = None

    def __post_init__(self) -> None:
        path = resolve_repo_path(self.dataset_path)
        self.dataset_path = path
        resolved_problem = normalize_problem_type(self.problem_type, default=None)
        self.problem_type = resolved_problem or "auto"
        self.instances = load_adapted_instances(
            path,
            num_customers=self.num_customers,
            num_charging_stations=self.num_charging_stations,
            problem_type=resolved_problem,
        )
        if not self.instances:
            raise FileNotFoundError(
                f"No adapted instances found under {path} "
                f"for Cus{self.num_customers}/CS{self.num_charging_stations}/problem={resolved_problem}"
            )
        self.sample_mode = str(self.sample_mode or "shuffle_cycle").lower()
        if self.sample_mode not in {"shuffle_cycle", "cycle", "random"}:
            raise ValueError("fixed dataset sample_mode must be one of: shuffle_cycle, cycle, random")
        self.rng = np.random.default_rng(self.seed)
        self.order = np.arange(len(self.instances), dtype=np.int64)
        if self.sample_mode == "shuffle_cycle":
            self.rng.shuffle(self.order)
        self.cursor = 0
        self.sample_count = 0
        self.region_pool_status = f"fixed_dataset:{path}:problem={self.problem_type}"
        self._reward_scale_cache: dict[str, float] = {}

    def sample(self) -> EVRPTWInstance:
        if self.sample_mode == "random":
            idx = int(self.rng.integers(0, len(self.instances)))
        else:
            if self.cursor >= len(self.order):
                self.cursor = 0
                if self.sample_mode == "shuffle_cycle":
                    self.rng.shuffle(self.order)
            idx = int(self.order[self.cursor])
            self.cursor += 1
        self.sample_count += 1
        return self.instances[idx]

    def reward_distance_scale_km(self, mode: str = "single_customer_repair_median") -> float:
        mode = str(mode)
        if mode in self._reward_scale_cache:
            return self._reward_scale_cache[mode]
        values: list[float] = []
        if mode == "max_edge":
            for instance in self.instances:
                dist = np.asarray(instance.distance_matrix_km, dtype=np.float64)
                finite = dist[np.isfinite(dist)]
                if finite.size:
                    values.append(float(finite.max()))
        elif mode in {
            "single_customer_repair_sum",
            "single_customer_repair_mean",
            "single_customer_repair_median",
        }:
            per_instance: list[float] = []
            all_customer_repairs: list[float] = []
            for instance in self.instances:
                n = int(instance.num_customers)
                dist = np.asarray(instance.distance_matrix_km, dtype=np.float64)
                repairs = dist[0, 1 : n + 1] + dist[1 : n + 1, 0]
                repairs = repairs[np.isfinite(repairs)]
                if not repairs.size:
                    continue
                if mode == "single_customer_repair_sum":
                    per_instance.append(float(repairs.sum()))
                else:
                    all_customer_repairs.extend(float(x) for x in repairs)
            values = per_instance if mode == "single_customer_repair_sum" else all_customer_repairs
        else:
            raise ValueError(f"Unsupported dataset reward scale mode: {mode}")
        if not values:
            scale = 1.0
        elif mode.endswith("_mean") or mode.endswith("_sum"):
            scale = float(np.mean(values))
        elif mode.endswith("_median"):
            scale = float(np.median(values))
        else:
            scale = float(np.max(values))
        scale = max(scale, 1e-9)
        self._reward_scale_cache[mode] = scale
        return scale

    def usage_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "region_id": "fixed_dataset",
                "sampled_days": self.sample_count,
                "customer_exposure_rate": "",
                "recent_mean_jaccard_distance": "",
                "cluster_exposure_entropy": "",
                "region_pool_status": self.region_pool_status,
                "dataset_size": len(self.instances),
                "sample_mode": self.sample_mode,
                "problem_type": self.problem_type,
            }
        ]

    def close(self, terminate: bool = False) -> None:
        del terminate
