from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

from evrptw_core.schema import EVRPTWInstance

from .data_pool import OnlineInstancePool


@dataclass(frozen=True)
class _WorkerConfig:
    config_path: str | Path
    num_regions: int
    mother_num_customers: int
    mother_num_charging_stations: int
    num_customers: int
    num_charging_stations: int
    region_reuse_limit: int
    seed: int | None
    max_attempts_per_instance: int | None
    territory_pool_path: str | Path | None
    region_pool_path: str | Path | None
    region_pool_shuffle: bool
    region_pool_replacement_policy: str


def _worker_loop(worker_id: int, cfg: _WorkerConfig, result_queue, stop_event) -> None:  # noqa: ANN001
    seed = None if cfg.seed is None else int(cfg.seed) + 1_000_003 * (worker_id + 1)
    try:
        pool = OnlineInstancePool(
            config_path=cfg.config_path,
            num_regions=max(1, int(cfg.num_regions)),
            mother_num_customers=int(cfg.mother_num_customers),
            mother_num_charging_stations=int(cfg.mother_num_charging_stations),
            num_customers=int(cfg.num_customers),
            num_charging_stations=int(cfg.num_charging_stations),
            region_reuse_limit=int(cfg.region_reuse_limit),
            seed=seed,
            max_attempts_per_instance=cfg.max_attempts_per_instance,
            territory_pool_path=cfg.territory_pool_path,
            region_pool_path=cfg.region_pool_path,
            region_pool_shuffle=cfg.region_pool_shuffle,
            region_pool_replacement_policy=cfg.region_pool_replacement_policy,
        )
        while not stop_event.is_set():
            instance = pool.sample()
            if instance is None:
                raise RuntimeError("OnlineInstancePool.sample returned None")
            payload = ("ok", worker_id, instance)
            while not stop_event.is_set():
                try:
                    result_queue.put(payload, timeout=0.25)
                    break
                except queue.Full:
                    continue
    except BaseException:
        payload = ("error", worker_id, traceback.format_exc())
        try:
            result_queue.put(payload, timeout=1.0)
        except Exception:
            pass


class AsyncInstancePool:
    """Persistent multiprocessing prefetcher for online training instances.

    Each worker owns a small independent ``OnlineInstancePool`` and continuously
    fills a bounded queue with ready-to-reset ``EVRPTWInstance`` objects. This
    keeps the main training loop synchronous and deterministic at the env API
    level while allowing CPU instance activation to overlap with rollout/PPO.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        num_regions: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
        num_customers: int,
        num_charging_stations: int,
        region_reuse_limit: int,
        seed: int | None,
        max_attempts_per_instance: int | None,
        num_workers: int = 4,
        queue_size: int = 128,
        regions_per_worker: int | None = None,
        multiprocessing_context: str = "spawn",
        get_timeout_s: float = 300.0,
        territory_pool_path: str | Path | None = None,
        region_pool_path: str | Path | None = None,
        region_pool_shuffle: bool = True,
        region_pool_replacement_policy: str = "cycle",
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.queue_size = max(self.num_workers, int(queue_size))
        self.get_timeout_s = float(get_timeout_s)
        worker_regions = int(regions_per_worker) if regions_per_worker is not None else max(1, int(num_regions + self.num_workers - 1) // self.num_workers)
        self.worker_cfg = _WorkerConfig(
            config_path=config_path,
            num_regions=worker_regions,
            mother_num_customers=int(mother_num_customers),
            mother_num_charging_stations=int(mother_num_charging_stations),
            num_customers=int(num_customers),
            num_charging_stations=int(num_charging_stations),
            region_reuse_limit=int(region_reuse_limit),
            seed=seed,
            max_attempts_per_instance=max_attempts_per_instance,
            territory_pool_path=territory_pool_path,
            region_pool_path=region_pool_path,
            region_pool_shuffle=bool(region_pool_shuffle),
            region_pool_replacement_policy=str(region_pool_replacement_policy),
        )
        self.mp_ctx = mp.get_context(multiprocessing_context)
        self.result_queue = self.mp_ctx.Queue(maxsize=self.queue_size)
        self.stop_event = self.mp_ctx.Event()
        self.processes: list[mp.Process] = []
        self._buffer: deque[EVRPTWInstance] = deque()
        self.sample_count = 0
        self.started = False
        self.worker_errors: list[str] = []
        self.total_wait_time_s = 0.0

    def start(self) -> None:
        if self.started:
            return
        self.stop_event.clear()
        for worker_id in range(self.num_workers):
            proc = self.mp_ctx.Process(
                target=_worker_loop,
                args=(worker_id, self.worker_cfg, self.result_queue, self.stop_event),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
        self.started = True

    def _get_from_queue(self) -> EVRPTWInstance:
        self.start()
        start = time.perf_counter()
        try:
            item = self.result_queue.get(timeout=self.get_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for async instance after {self.get_timeout_s:.1f}s") from exc
        self.total_wait_time_s += time.perf_counter() - start
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError(f"Bad async instance queue payload: {type(item).__name__} {item!r}")
        status, worker_id, payload = item
        if status == "error":
            self.worker_errors.append(str(payload))
            raise RuntimeError(f"Async instance worker {worker_id} failed:\n{payload}")
        if status != "ok":
            raise RuntimeError(f"Unknown async instance worker status from worker {worker_id}: {status!r}")
        if payload is None:
            raise RuntimeError(f"Async instance worker {worker_id} returned None")
        return payload

    def sample(self) -> EVRPTWInstance:
        if self._buffer:
            instance = self._buffer.popleft()
        else:
            instance = self._get_from_queue()
        self.sample_count += 1
        return instance

    def warmup(self, n: int) -> float:
        start = time.perf_counter()
        for _ in range(int(n)):
            self._buffer.append(self._get_from_queue())
        return time.perf_counter() - start

    def close(self, terminate: bool = False) -> None:
        if not self.started:
            return
        self.stop_event.set()
        for proc in self.processes:
            proc.join(timeout=1.0)
        if terminate:
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self.processes:
                proc.join(timeout=1.0)
        self.processes.clear()
        self.started = False

    def usage_summary(self) -> list[dict[str, Any]]:
        return []

    def __enter__(self) -> "AsyncInstancePool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close(terminate=exc_type is not None)


__all__ = ["AsyncInstancePool"]
