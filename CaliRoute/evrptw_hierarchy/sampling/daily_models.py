from __future__ import annotations

import numpy as np


def sample_trunc_normal(rng: np.random.Generator, mean: float, std: float, low: float, high: float) -> float:
    if std <= 1e-12 or high <= low:
        return float(np.clip(mean, low, high))
    for _ in range(100):
        value = float(rng.normal(mean, std))
        if low <= value <= high:
            return value
    return float(np.clip(mean, low, high))


def sample_day_profile(config: dict, rng: np.random.Generator) -> dict:
    day_cfg = config.get("day", {})
    day_type = day_cfg.get("day_type", "mixed")
    if day_type == "mixed":
        p_weekday = float(day_cfg.get("weekday_weight", 5.0)) / max(
            float(day_cfg.get("weekday_weight", 5.0)) + float(day_cfg.get("weekend_weight", 2.0)), 1e-12
        )
        day_type = "weekday" if rng.random() < p_weekday else "weekend"
    profile = day_cfg.get(day_type, day_cfg.get("weekday", {}))
    start_h = sample_trunc_normal(
        rng,
        float(profile.get("working_start_mean_h", 7.8)),
        float(profile.get("working_start_std_h", 0.79)),
        float(profile.get("working_start_min_h", 6.3)),
        float(profile.get("working_start_max_h", 10.5)),
    )
    horizon_min = int(round(float(day_cfg.get("working_horizon_hours", 12.0)) * 60.0))
    working_start = int(round(start_h * 60.0))
    working_end = min(1439, working_start + horizon_min)
    congestion = config.get("congestion", {})
    factor = sample_trunc_normal(
        rng,
        float(congestion.get("mean_factor", 0.375)),
        float(congestion.get("std_factor", 0.06)),
        float(congestion.get("min_factor", 0.25)),
        float(congestion.get("max_factor", 0.60)),
    )
    return {
        "day_type": day_type,
        "working_start_min": working_start,
        "working_end_min": working_end,
        "congestion_factor": factor,
    }


def sample_demands_cm3(config: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cfg = config.get("demand", {})
    pkg_cfg = cfg.get("package_count_negative_binomial", {})
    mean_extra = float(pkg_cfg.get("mean_extra_packages", 0.62194))
    dispersion = float(pkg_cfg.get("dispersion", 0.327))
    p = dispersion / max(dispersion + mean_extra, 1e-12)
    package_counts = 1 + rng.negative_binomial(dispersion, p, size=n)
    package_counts = np.clip(package_counts, 1, int(cfg.get("max_packages_per_stop", 72))).astype(np.int32)

    vol_cfg = cfg.get("per_package_volume_lognormal", {})
    median_cm3 = float(vol_cfg.get("median_cm3", 7000.0))
    sigma = float(vol_cfg.get("sigma", 1.0))
    max_pkg = float(vol_cfg.get("max_package_volume_cm3", 300000.0))
    volumes = np.empty(n, dtype=np.float32)
    for i, k in enumerate(package_counts):
        draws = rng.lognormal(np.log(max(median_cm3, 1e-9)), sigma, size=int(k))
        volumes[i] = float(np.clip(draws, 1.0, max_pkg).sum())
    return volumes, package_counts


def sample_service_time_min(config: dict, demand_cm3: np.ndarray, package_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cfg = config.get("service_time", {})
    seconds = (
        float(cfg.get("base_seconds", 28.1626))
        + float(cfg.get("beta_package_seconds", 46.9063)) * package_counts.astype(float)
        + float(cfg.get("beta_volume_seconds_per_cm3", 0.000358429)) * demand_cm3.astype(float)
    )
    noise = rng.lognormal(mean=0.0, sigma=float(cfg.get("lognormal_noise_sigma", 0.75)), size=len(seconds))
    centered_noise = noise / max(float(np.mean(noise)), 1e-12)
    seconds = seconds * centered_noise
    seconds = np.clip(seconds, float(cfg.get("min_seconds", 5.0)), float(cfg.get("max_seconds", 8007.0)))
    return (seconds / 60.0).astype(np.float32)


def sample_time_windows(
    config: dict,
    day_type: str,
    working_start_min: int,
    working_end_min: int,
    depot_to_customer_min: np.ndarray,
    customer_to_depot_min: np.ndarray,
    service_time_min: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    cfg = config.get("time_window", {})
    n = int(len(service_time_min))
    tw = np.column_stack([
        np.full(n, working_start_min, dtype=np.float32),
        np.full(n, working_end_min, dtype=np.float32),
    ])
    beta_cfg = cfg.get("presence_rate_beta", {}).get(day_type, cfg.get("presence_rate_beta", {}).get("weekday", {}))
    rate = rng.beta(float(beta_cfg.get("alpha", 6.1)), float(beta_cfg.get("beta", 60.7)))
    rate *= float(cfg.get("presence_rate_multiplier", 1.0))
    num_tw = int(np.clip(round(rate * n), int(cfg.get("min_tw_customers", 0)), n))
    if num_tw <= 0:
        return tw, {"num_tw": 0, "presence_rate": 0.0}

    feasible_start = working_start_min + depot_to_customer_min
    feasible_end = working_end_min - customer_to_depot_min - service_time_min
    feasible = np.where(feasible_start + 1e-6 <= feasible_end)[0]
    if feasible.size == 0:
        return tw, {"num_tw": 0, "presence_rate": 0.0, "warning": "no_feasible_tw_customers"}

    service_rank = np.argsort(np.argsort(service_time_min)).astype(float) / max(n - 1, 1)
    weights = 1.0 + float(cfg.get("service_time_selection_weight", 1.5)) * service_rank[feasible]
    weights = weights / weights.sum()
    chosen = rng.choice(feasible, size=min(num_tw, feasible.size), replace=False, p=weights)

    scenario_share = cfg.get("realistic_strain_share", {}).get(day_type, 0.8)
    num_strain = int(round(float(scenario_share) * len(chosen)))
    labels = np.array(["loose"] * len(chosen), dtype=object)
    if len(chosen):
        labels[:num_strain] = "strain"
        rng.shuffle(labels)

    profiles = cfg.get("width_center_profiles", {}).get(day_type, cfg.get("width_center_profiles", {}).get("weekday", {}))
    for idx, label in zip(chosen, labels):
        prof = profiles.get(str(label), {})
        width_h = sample_trunc_normal(
            rng,
            float(prof.get("width_mean_h", 7.2)),
            float(prof.get("width_std_h", 1.0)),
            float(prof.get("width_min_h", 0.5)),
            float(prof.get("width_max_h", 12.0)),
        )
        center_rel_h = sample_trunc_normal(
            rng,
            float(prof.get("center_mean_h", 4.0)),
            float(prof.get("center_std_h", 1.2)),
            float(prof.get("center_min_h", -1.0)),
            float(prof.get("center_max_h", 10.0)),
        )
        width = width_h * 60.0
        center = working_start_min + center_rel_h * 60.0
        start = center - 0.5 * width
        end = center + 0.5 * width
        span_start = float(feasible_start[idx])
        span_end = float(feasible_end[idx])
        if end < span_start or start > span_end:
            center = rng.uniform(span_start, span_end)
            start = center - 0.5 * min(width, span_end - span_start)
            end = center + 0.5 * min(width, span_end - span_start)
        tw[idx, 0] = max(span_start, start, working_start_min)
        # TW stores the latest service-start time; service duration and return
        # travel are handled by feasibility checks.
        tw[idx, 1] = min(span_end, end, working_end_min)
        if tw[idx, 1] <= tw[idx, 0]:
            tw[idx] = [working_start_min, working_end_min]
    metadata = {
        "num_tw": int(np.sum((tw[:, 0] > working_start_min) | (tw[:, 1] < working_end_min))),
        "presence_rate": float(np.mean((tw[:, 0] > working_start_min) | (tw[:, 1] < working_end_min))),
        "num_strain_target": int(num_strain),
    }
    return tw.astype(np.float32), metadata
