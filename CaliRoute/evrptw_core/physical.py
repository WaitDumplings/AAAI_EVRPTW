"""One physical edge-cost contract for environments and independent validators."""
from __future__ import annotations

import numpy as np


def _explicit_matrix(instance, name, distance):
    value = getattr(instance, name, None)
    if value is None:
        value = (getattr(instance, 'raw', None) or {}).get(name)
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != distance.shape:
        raise ValueError(f'{name} shape must match distance_matrix_km')
    if np.isnan(matrix).any() or (matrix < 0).any():
        raise ValueError(f'{name} must contain nonnegative costs or +inf')
    if not np.array_equal(np.isfinite(matrix), np.isfinite(distance)):
        raise ValueError(f'{name} reachability must match distance_matrix_km')
    return matrix


def resolve_physical_edge_matrices(instance, *, prefer_explicit_edge_matrices=False):
    """Return km/s/kWh matrices without changing units or route semantics.

    ``False`` preserves historical D/v and D*c behavior. ``True`` selects an
    explicit T/E matrix independently when available. Positive nonuniform costs
    are valid; we do not pretend that nominal speed/consumption then determine
    every edge. Explicit matrices are required to share the road reachability.
    """
    if not isinstance(prefer_explicit_edge_matrices, bool):
        raise ValueError('prefer_explicit_edge_matrices must be a boolean')
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    terminals = 1 + int(instance.num_customers) + int(getattr(instance, 'num_charging_stations', 0))
    if distance.ndim != 2 or distance.shape != (terminals, terminals):
        raise ValueError('distance_matrix_km shape must match the instance terminals')
    if np.isnan(distance).any() or (distance < 0).any():
        raise ValueError('distance_matrix_km must contain nonnegative distances or +inf')
    travel = _explicit_matrix(instance, 'travel_time_matrix_s', distance) if prefer_explicit_edge_matrices else None
    energy = _explicit_matrix(instance, 'energy_matrix_kwh', distance) if prefer_explicit_edge_matrices else None
    travel_source = 'provided_travel_time_matrix_s' if travel is not None else 'distance_over_effective_speed'
    energy_source = 'provided_energy_matrix_kwh' if energy is not None else 'distance_times_consumption'
    if travel is None:
        speed = float(instance.speed_profile.get('effective_speed_kmh')
                      or instance.vehicle.get('design_speed_kmh') or 40.)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('effective_speed_kmh must be finite and positive')
        travel = distance / max(speed / 3600., 1e-12)
    if energy is None:
        consumption = float(instance.vehicle.get('consumption_kwh_per_km', .404))
        if not np.isfinite(consumption) or consumption < 0:
            raise ValueError('consumption_kwh_per_km must be finite and nonnegative')
        # Avoid inf*0 for inactive battery resources; unreachable roads remain
        # unreachable for all cost matrices and are excluded by the action mask.
        energy = np.full_like(distance, np.inf)
        np.multiply(distance, consumption, out=energy, where=np.isfinite(distance))
    return {'distance_km': distance, 'travel_time_s': travel, 'energy_kwh': energy,
            'travel_time_source': travel_source, 'energy_source': energy_source}
