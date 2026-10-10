from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl4co.envs.common.utils import Generator
from tensordict.tensordict import TensorDict


class FixedRCVRPGenerator(Generator):
    """Sample RCVRP training batches from a converted npz dataset."""

    def __init__(
        self,
        data_file: str,
        num_loc: int | None = None,
        sample_with_replacement: bool = True,
        demand_is_normalized: bool = False,
        vehicle_capacity: float = 1.0,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_file = str(data_file)
        self.sample_with_replacement = sample_with_replacement
        self.demand_is_normalized = demand_is_normalized
        self.vehicle_capacity = vehicle_capacity
        self.min_loc = min_loc
        self.max_loc = max_loc

        self._data = np.load(Path(data_file), allow_pickle=False)
        self.dataset_size = int(self._data["locs"].shape[0])
        self.num_loc = int(self._data["locs"].shape[1])
        if num_loc is not None and int(num_loc) != self.num_loc:
            # Prefer the dataset metadata over a stale experiment override.
            self.num_loc = int(self._data["locs"].shape[1])

        if self.demand_is_normalized:
            demand = self._data["demand"]
        else:
            demand = self._data["demand"] / self._data["capacity"][:, None]
        self.capacity = 1.0
        self.max_demand = float(np.max(demand)) if demand.size else 1.0

    def _sample_indices(self, batch_size: list[int]) -> np.ndarray:
        size = int(batch_size[0])
        if self.sample_with_replacement:
            return np.random.randint(0, self.dataset_size, size=size)
        if size > self.dataset_size:
            raise ValueError(
                f"Requested batch size {size} exceeds dataset size {self.dataset_size} "
                "with sample_with_replacement=False."
            )
        if size == self.dataset_size:
            return np.arange(self.dataset_size)
        return np.random.choice(self.dataset_size, size=size, replace=False)

    def _generate(self, batch_size) -> TensorDict:
        idx = self._sample_indices(batch_size)
        demand = self._data["demand"][idx].astype(np.float32)
        if not self.demand_is_normalized:
            capacity = self._data["capacity"][idx].astype(np.float32)
            demand = demand / capacity[:, None]

        raw_demand = self._data["demand"][idx].astype(np.float32)
        raw_capacity = self._data["capacity"][idx].astype(np.float32)

        td = TensorDict(
            {
                "depot": torch.from_numpy(self._data["depot"][idx].astype(np.float32)),
                "locs": torch.from_numpy(self._data["locs"][idx].astype(np.float32)),
                "demand": torch.from_numpy(demand.astype(np.float32)),
                "capacity": torch.ones((len(idx), 1), dtype=torch.float32),
                "distance_matrix": torch.from_numpy(
                    self._data["distance_matrix"][idx].astype(np.float32)
                ),
                "instance_idx": torch.from_numpy(idx.astype(np.int64)),
                "raw_demand": torch.from_numpy(raw_demand),
                "raw_capacity": torch.from_numpy(raw_capacity[:, None]),
            },
            batch_size=[len(idx)],
        )
        return td

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._data = np.load(Path(self.data_file), allow_pickle=False)
