from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl4co.envs.common.utils import Generator
from tensordict.tensordict import TensorDict


class FixedRMTVRPGenerator(Generator):
    """Sample RCVRPTW training batches from a converted npz dataset."""

    def __init__(
        self,
        data_file: str,
        num_loc: int | None = None,
        sample_with_replacement: bool = True,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_file = str(data_file)
        self.sample_with_replacement = sample_with_replacement
        self.min_loc = min_loc
        self.max_loc = max_loc

        self._data = np.load(Path(data_file), allow_pickle=False)
        self.dataset_size = int(self._data["locs"].shape[0])
        self.num_loc = int(self._data["locs"].shape[1] - 1)
        if num_loc is not None and int(num_loc) != self.num_loc:
            # Prefer the dataset metadata over a stale experiment override.
            self.num_loc = int(self._data["locs"].shape[1] - 1)

        self.capacity = 1.0
        self.max_demand = float(np.max(self._data["demand_linehaul"]))

    def _sample_indices(self, batch_size: list[int]) -> np.ndarray:
        size = int(batch_size[0])
        if self.sample_with_replacement:
            return np.random.randint(0, self.dataset_size, size=size)
        if size > self.dataset_size:
            raise ValueError(
                f"Requested batch size {size} exceeds dataset size {self.dataset_size} "
                "with sample_with_replacement=False."
            )
        return np.random.choice(self.dataset_size, size=size, replace=False)

    def _array(
        self,
        key: str,
        idx: np.ndarray,
        default: np.ndarray | None = None,
        dtype=np.float32,
    ) -> torch.Tensor:
        if key in self._data:
            arr = self._data[key][idx]
        elif default is not None:
            arr = default
        else:
            raise KeyError(f"Missing key {key!r} in {self.data_file}")
        return torch.from_numpy(arr.astype(dtype, copy=False))

    def _generate(self, batch_size) -> TensorDict:
        idx = self._sample_indices(batch_size)
        batch = len(idx)
        zeros_customers = np.zeros((batch, self.num_loc), dtype=np.float32)
        ones = np.ones((batch, 1), dtype=np.float32)

        td = TensorDict(
            {
                "locs": self._array("locs", idx),
                "demand_linehaul": self._array("demand_linehaul", idx),
                "demand_backhaul": self._array(
                    "demand_backhaul", idx, default=zeros_customers
                ),
                "backhaul_class": self._array("backhaul_class", idx, default=ones),
                "distance_limit": self._array(
                    "distance_limit",
                    idx,
                    default=np.full((batch, 1), np.inf, dtype=np.float32),
                ),
                "time_windows": self._array("time_windows", idx),
                "service_time": self._array("service_time", idx),
                "vehicle_capacity": self._array("vehicle_capacity", idx, default=ones),
                "capacity_original": self._array("capacity_original", idx, default=ones),
                "open_route": self._array(
                    "open_route",
                    idx,
                    default=np.zeros((batch, 1), dtype=bool),
                    dtype=bool,
                ),
                "speed": self._array("speed", idx, default=ones),
                "distance_matrix": self._array("distance_matrix", idx),
                "duration_matrix": self._array("duration_matrix", idx),
            },
            batch_size=[batch],
        )
        return td

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._data = np.load(Path(self.data_file), allow_pickle=False)
