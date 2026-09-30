from __future__ import annotations

import numpy as np

from offline2online.offline_data import same_route_matrix_from_routes


def test_same_route_matrix_is_permutation_invariant() -> None:
    routes = [
        [0, 3, 2, 1, 0],
        [0, 7, 5, 0],
        [0, 4, 6, 0],
    ]
    shuffled_route_order = [
        [0, 4, 6, 0],
        [0, 7, 5, 0],
        [0, 3, 2, 1, 0],
    ]
    shuffled_internal_order = [
        [0, 1, 3, 2, 0],
        [0, 5, 7, 0],
        [0, 6, 4, 0],
    ]
    expected = same_route_matrix_from_routes(routes, num_customers=7)
    assert np.array_equal(expected, same_route_matrix_from_routes(shuffled_route_order, num_customers=7))
    assert np.array_equal(expected, same_route_matrix_from_routes(shuffled_internal_order, num_customers=7))


def test_same_route_matrix_ignores_depot_and_charging_stations() -> None:
    routes = [
        [0, 1, 8, 2, 0],
        [0, 3, 9, 4, 0],
    ]
    labels = same_route_matrix_from_routes(routes, num_customers=4)
    assert labels.shape == (4, 4)
    assert labels[0, 1] == 1.0
    assert labels[2, 3] == 1.0
    assert labels[0, 2] == 0.0
    assert np.array_equal(labels, labels.T)
