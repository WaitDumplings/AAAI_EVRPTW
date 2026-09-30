from __future__ import annotations

import numpy as np

from evrptw_core.schema import EVRPTWInstance, ValidationResult


def validate_instance_structure(instance: EVRPTWInstance) -> ValidationResult:
    """Validate schema-level consistency without solving the EVRP-TW instance."""

    errors: list[str] = []
    warnings: list[str] = []
    n = instance.num_customers
    m = instance.num_charging_stations
    terminals = instance.num_terminals

    if instance.depot.shape != (2,):
        errors.append(f"depot must have shape (2,), got {instance.depot.shape}")
    if instance.customers.shape != (n, 2):
        errors.append(f"customers must have shape (n, 2), got {instance.customers.shape}")
    if instance.charging_stations.shape != (m, 2):
        errors.append(f"charging_stations must have shape (m, 2), got {instance.charging_stations.shape}")
    if instance.distance_matrix_km.shape != (terminals, terminals):
        errors.append(
            f"distance_matrix_km must have shape ({terminals}, {terminals}), "
            f"got {instance.distance_matrix_km.shape}"
        )
    for name, arr in [
        ("demands_cm3", instance.demands_cm3),
        ("package_counts", instance.package_counts),
        ("service_time_s", instance.service_time_s),
    ]:
        if arr.shape[0] != n:
            errors.append(f"{name} length must equal num_customers={n}, got {arr.shape[0]}")
    if instance.tw_s.shape != (n, 2):
        errors.append(f"tw_s must have shape ({n}, 2), got {instance.tw_s.shape}")
    if instance.cs_time_to_depot_s.shape[0] != m:
        errors.append(f"cs_time_to_depot_s length must equal num_charging_stations={m}, got {instance.cs_time_to_depot_s.shape[0]}")

    if instance.working_end_s <= instance.working_start_s:
        errors.append("working_end_s must be greater than working_start_s")
    if np.any(instance.demands_cm3 < 0):
        errors.append("demands_cm3 contains negative values")
    if np.any(instance.service_time_s < 0):
        errors.append("service_time_s contains negative values")
    if instance.tw_s.size and np.any(instance.tw_s[:, 1] < instance.tw_s[:, 0]):
        errors.append("tw_s contains end before start")
    if np.any(~np.isfinite(instance.distance_matrix_km)):
        warnings.append("distance_matrix_km contains non-finite values")
    if instance.distance_matrix_km.shape == (terminals, terminals):
        diagonal = np.diag(instance.distance_matrix_km)
        if np.max(np.abs(diagonal)) > 1e-5:
            warnings.append("distance_matrix_km diagonal is not zero")

    return ValidationResult(
        success=not errors,
        errors=errors,
        warnings=warnings,
        metrics={
            "num_customers": n,
            "num_charging_stations": m,
            "num_terminals": terminals,
            "working_horizon_s": int(instance.working_end_s - instance.working_start_s),
        },
    )
