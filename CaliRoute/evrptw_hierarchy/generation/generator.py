from __future__ import annotations

from dataclasses import MISSING, dataclass, field
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from evrptw_hierarchy.configs.config import load_yaml, vehicle_from_config
from evrptw_hierarchy.core.models import RegionBoard, RegionUsage, VehicleConfig
from evrptw_hierarchy.generation.region_generator import RegionGenerator
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle
from evrptw_hierarchy.io.persistence import ensure_dir, load_pickle, save_pickle
from evrptw_hierarchy.sampling.active_day import ActiveDaySampler
from evrptw_hierarchy.validation.reports import summarize_instance, summarize_region, write_reports
from evrptw_hierarchy.visualization.plots import write_region_svg


def _fresh_usage(board: RegionBoard) -> RegionUsage:
    return RegionUsage(
        region_id=board.region_id,
        sampled_days=0,
        customer_activation_counts=np.zeros(len(board.customers), dtype=np.int32),
        cluster_activation_counts=np.zeros(len(board.cluster_centers), dtype=np.int32),
    )


def _region_board_from_payload(payload: Any) -> RegionBoard:
    if isinstance(payload, RegionBoard):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported region-board payload type: {type(payload).__name__}")
    fields = RegionBoard.__dataclass_fields__
    kwargs = {key: payload[key] for key in fields if key in payload}
    missing = [
        key
        for key, field_def in fields.items()
        if key not in kwargs and field_def.default is MISSING and field_def.default_factory is MISSING
    ]
    if missing:
        raise ValueError(f"Region-board payload is missing required fields: {missing}")
    return RegionBoard(**kwargs)


@dataclass
class HierarchyDatasetGenerator:
    config: dict[str, Any]
    seed: int | None = None
    vehicle: VehicleConfig = field(init=False)
    rng: np.random.Generator = field(init=False)
    boards: list[RegionBoard] = field(default_factory=list, init=False)
    usages: list[RegionUsage] = field(default_factory=list, init=False)
    oracles: dict[str, DistanceOracle] = field(default_factory=dict, init=False)
    next_region_serial: int = field(default=0, init=False)
    precomputed_boards: list[RegionBoard] = field(default_factory=list, init=False)
    next_precomputed_index: int = field(default=0, init=False)
    precomputed_replacement_policy: str = field(default="cycle", init=False)

    def __post_init__(self) -> None:
        cfg_seed = self.config.get("seed")
        actual_seed = self.seed if self.seed is not None else cfg_seed
        self.rng = np.random.default_rng(None if actual_seed in (None, "") else int(actual_seed))
        self.vehicle = vehicle_from_config(self.config)

    @classmethod
    def from_config_path(cls, config_path: str | Path, seed: int | None = None) -> "HierarchyDatasetGenerator":
        return cls(load_yaml(config_path), seed=seed)

    def _next_precomputed_board(self, slot_index: int) -> RegionBoard | None:
        if not self.precomputed_boards:
            return None
        if self.next_precomputed_index >= len(self.precomputed_boards):
            if self.precomputed_replacement_policy != "cycle":
                return None
            self.next_precomputed_index = 0
        active_ids = {board.region_id for idx, board in enumerate(self.boards) if idx != slot_index}
        for _ in range(len(self.precomputed_boards)):
            board = self.precomputed_boards[self.next_precomputed_index]
            self.next_precomputed_index = (self.next_precomputed_index + 1) % len(self.precomputed_boards)
            if board.region_id not in active_ids:
                return board
        return self.precomputed_boards[self.next_precomputed_index]

    def _create_region(self, slot_index: int, mother_num_customers: int, mother_num_charging_stations: int) -> tuple[RegionBoard, RegionUsage]:
        precomputed = self._next_precomputed_board(slot_index)
        if precomputed is not None:
            board = precomputed
            usage = _fresh_usage(board)
            if slot_index < len(self.boards):
                old_id = self.boards[slot_index].region_id
                self.oracles.pop(old_id, None)
                self.boards[slot_index] = board
                self.usages[slot_index] = usage
            else:
                self.boards.append(board)
                self.usages.append(usage)
            return board, usage

        generator = RegionGenerator(self.config, self.vehicle, self.rng)
        serial = self.next_region_serial
        self.next_region_serial += 1
        board = generator.generate(serial, mother_num_customers, mother_num_charging_stations)
        usage = _fresh_usage(board)
        if slot_index < len(self.boards):
            old_id = self.boards[slot_index].region_id
            self.oracles.pop(old_id, None)
            self.boards[slot_index] = board
            self.usages[slot_index] = usage
        else:
            self.boards.append(board)
            self.usages.append(usage)
        return board, usage

    def _oracle_for(self, board: RegionBoard) -> DistanceOracle:
        if board.region_id not in self.oracles:
            sp_cfg = self.config.get("shortest_path", {})
            depot_nodes = np.asarray([board.depot_node_id], dtype=np.int32)
            if board.depot_candidate_node_ids is not None and len(board.depot_candidate_node_ids):
                depot_nodes = np.unique(
                    np.concatenate([depot_nodes, np.asarray(board.depot_candidate_node_ids, dtype=np.int32)])
                ).astype(np.int32)
            if str(board.metadata.get("customer_connection_mode", "")).startswith("lazy_"):
                terminal_node_ids = np.concatenate([
                    depot_nodes,
                    board.cs_node_ids.astype(np.int32),
                ])
            else:
                terminal_node_ids = np.concatenate([
                    depot_nodes,
                    board.customer_node_ids.astype(np.int32),
                    board.cs_node_ids.astype(np.int32),
                ])
            mode = str(sp_cfg.get("oracle_mode", "auto"))
            estimated_mb = terminal_node_ids.size * terminal_node_ids.size * 4.0 / (1024.0 * 1024.0)
            use_terminal = mode == "terminal_matrix" or (
                mode == "auto"
                and terminal_node_ids.size <= int(sp_cfg.get("terminal_matrix_max_terminals", 8000))
                and estimated_mb <= float(sp_cfg.get("terminal_matrix_max_mb", 512.0))
            )
            self.oracles[board.region_id] = DistanceOracle(
                len(board.road_nodes),
                board.road_edges,
                board.road_edge_lengths_km,
                terminal_node_ids=terminal_node_ids,
                use_terminal_matrix=use_terminal,
            )
        return self.oracles[board.region_id]

    def _is_stale(self, usage: RegionUsage, region_reuse_limit: int) -> bool:
        fresh = self.config.get("freshness", {})
        if usage.sampled_days >= int(region_reuse_limit):
            return True
        if bool(fresh.get("use_exposure_rule", True)):
            if usage.customer_exposure_rate >= float(fresh.get("customer_exposure_rate_threshold", 0.85)):
                return True
        if bool(fresh.get("use_jaccard_rule", True)):
            min_recent = int(fresh.get("min_recent_days", 10))
            if len(usage.recent_active_customer_sets) >= min_recent:
                if usage.recent_mean_jaccard_distance <= float(fresh.get("recent_jaccard_diversity_threshold", 0.65)):
                    return True
        return False

    def _select_region_slot(self, region_reuse_limit: int) -> int:
        eligible = [idx for idx, usage in enumerate(self.usages) if not self._is_stale(usage, region_reuse_limit)]
        if not eligible:
            return int(np.argmin([usage.sampled_days for usage in self.usages]))
        return min(eligible, key=lambda idx: self.usages[idx].sampled_days)

    def _save_regions(self, save_path: Path) -> None:
        region_dir = ensure_dir(save_path / "regions")
        for board in self.boards:
            save_pickle(region_dir / f"{board.region_id}_board.pkl", board)

    def save_region_pool(
        self,
        save_path: str | Path,
        num_regions: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
    ) -> dict[str, Any]:
        """Generate and persist a reusable service-territory graph pool."""
        root = ensure_dir(save_path)
        self.prepare_region_pool(
            num_regions=int(num_regions),
            mother_num_customers=int(mother_num_customers),
            mother_num_charging_stations=int(mother_num_charging_stations),
        )
        self._save_regions(root)
        usages = [_fresh_usage(board) for board in self.boards]
        region_rows = [summarize_region(board, usage) for board, usage in zip(self.boards, usages)]
        manifest = {
            "dataset_family": "EVRPTW-D",
            "dataset_version": "AC-v1",
            "calibration_profile": "Amazon-Calibrated",
            "pool_type": "service_territory_pool",
            "pool_path": str(root),
            "num_service_territories": int(len(self.boards)),
            "num_regions": int(len(self.boards)),
            "latent_customer_pool_size": int(mother_num_customers),
            "cs_candidate_pool_size": int(mother_num_charging_stations),
            "mother_num_customers": int(mother_num_customers),
            "mother_num_charging_stations": int(mother_num_charging_stations),
            "seed": self.seed,
            "service_territory_ids": [board.region_id for board in self.boards],
            "region_ids": [board.region_id for board in self.boards],
            "region_rows": region_rows,
            "terminology": {
                "service_territory_graph": "Stable city/region/delivery-station service territory.",
                "legacy_internal_fields": ["mother_board_id"],
            },
        }
        with (root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def load_region_pool(
        self,
        pool_path: str | Path,
        num_regions: int | None = None,
        shuffle: bool = True,
        replacement_policy: str = "cycle",
    ) -> None:
        """Load a reusable service-territory pool and activate up to ``num_regions`` boards.

        ``num_regions`` controls the active service-territory pool used by online
        training. The full precomputed pool remains available for stale-territory
        replacement, so a larger offline pool can support many training epochs
        without regenerating territory geometry.
        """
        root = Path(pool_path)
        boards: list[RegionBoard]
        bundle_path = root if root.is_file() else root / "service_territory_pool.pkl"
        if bundle_path.exists():
            boards = []
            with bundle_path.open("rb") as f:
                first = pickle.load(f)
                if isinstance(first, dict) and first.get("format") == "evrptw_service_territory_pool_v1":
                    count = int(first.get("num_service_territories", 0))
                    for _ in range(count):
                        try:
                            boards.append(_region_board_from_payload(pickle.load(f)))
                        except EOFError:
                            break
                elif isinstance(first, dict) and "service_territories" in first:
                    boards = [_region_board_from_payload(payload) for payload in first["service_territories"]]
                else:
                    boards = [_region_board_from_payload(first)]
            if not boards:
                raise FileNotFoundError(f"No service territory payloads found in {bundle_path}")
        else:
            region_dir = root / "regions" if (root / "regions").exists() else root
            board_paths = sorted(region_dir.glob("*_board.pkl"))
            if not board_paths:
                raise FileNotFoundError(f"No service territory bundle or region board pickles found under {root}")
            boards = [_region_board_from_payload(load_pickle(path)) for path in board_paths]
        if shuffle:
            order = self.rng.permutation(len(boards))
            boards = [boards[int(idx)] for idx in order]
        active_count = len(boards) if num_regions is None else min(int(num_regions), len(boards))
        if active_count <= 0:
            raise ValueError("num_regions must select at least one precomputed region board.")
        self.precomputed_boards = boards
        self.precomputed_replacement_policy = str(replacement_policy)
        self.boards = boards[:active_count]
        self.usages = [_fresh_usage(board) for board in self.boards]
        self.oracles.clear()
        self.next_precomputed_index = active_count
        self.next_region_serial = max(self.next_region_serial, len(boards))

    def prepare_region_pool(
        self,
        num_regions: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
    ) -> None:
        """Ensure the in-memory service-territory pool contains ``num_regions`` boards."""
        while len(self.boards) < int(num_regions):
            slot = len(self.boards)
            self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))

    def sample_active_instance(
        self,
        num_customers: int,
        num_charging_stations: int,
        region_reuse_limit: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
        instance_index: int = 0,
        max_attempts_per_instance: int | None = None,
    ):
        """Sample one feasible active-day instance from the current region pool.

        This is the online training API used by RL environments. It keeps the
        same stale-region replacement rules as offline dataset generation but
        does not write instances or region boards to disk.
        """
        if not self.boards:
            raise RuntimeError("Region pool is empty; call prepare_region_pool first.")
        sampler = ActiveDaySampler(self.config, self.vehicle, self.rng)
        max_attempts = int(max_attempts_per_instance or self.config.get("generation", {}).get("max_attempts_per_instance", 30))
        outer_attempts = int(self.config.get("generation", {}).get("max_region_attempts_per_instance", 5))
        last_error = "not_started"
        for _ in range(outer_attempts):
            slot = self._select_region_slot(int(region_reuse_limit))
            if self._is_stale(self.usages[slot], int(region_reuse_limit)):
                self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
            board = self.boards[slot]
            usage = self.usages[slot]
            oracle = self._oracle_for(board)
            try:
                instance = sampler.build_instance(
                    board=board,
                    usage_index=usage.sampled_days + 1,
                    instance_index=int(instance_index),
                    num_customers=int(num_customers),
                    num_charging_stations=int(num_charging_stations),
                    max_attempts=max_attempts,
                    oracle=oracle,
                )
                usage.record_day(
                    instance.active_customer_ids,
                    board.cluster_labels,
                    int(self.config.get("freshness", {}).get("recent_window", 30)),
                )
                return instance
            except Exception as exc:
                last_error = str(exc)
                self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
        raise RuntimeError(f"Failed to sample online active instance: {last_error}")

    def generate(
        self,
        save_path: str | Path,
        num_instances: int,
        num_customers: int,
        num_charging_stations: int,
        num_regions: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
        region_reuse_limit: int,
        max_attempts_per_instance: int | None = None,
        save_plots: bool = True,
        save_regions: bool = True,
        dataset_metadata: dict[str, Any] | None = None,
        clear_oracle_after_instance: bool = False,
        bundle_instances: bool = True,
    ) -> dict[str, Any]:
        root = Path(save_path)
        ensure_dir(root)
        instance_dir = root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}"
        instances_path = root / "instances.pkl"
        plots_dir = ensure_dir(root / "analysis_outputs" / "plots")

        while len(self.boards) < int(num_regions):
            slot = len(self.boards)
            self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
        if save_regions:
            self._save_regions(root)

        sampler = ActiveDaySampler(self.config, self.vehicle, self.rng)
        max_attempts = int(max_attempts_per_instance or self.config.get("generation", {}).get("max_attempts_per_instance", 30))
        failed_attempt_rows: list[dict[str, Any]] = []
        instance_rows: list[dict[str, Any]] = []
        plot_limit = int(self.config.get("visualization", {}).get("max_instance_plots", 10))

        bundle_file = None
        if bundle_instances:
            header = {
                "format": "evrptw_instance_bundle_v1",
                "dataset_metadata": dataset_metadata or {},
                "num_instances": int(num_instances),
                "num_customers": int(num_customers),
                "num_charging_stations": int(num_charging_stations),
            }
            bundle_file = instances_path.open("wb")
            pickle.dump(header, bundle_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            ensure_dir(instance_dir)

        try:
            for instance_index in range(int(num_instances)):
                outer_attempts = int(self.config.get("generation", {}).get("max_region_attempts_per_instance", 5))
                last_error = "not_started"
                instance = None
                board = None
                slot = 0
                for outer_attempt in range(outer_attempts):
                    slot = self._select_region_slot(int(region_reuse_limit))
                    if self._is_stale(self.usages[slot], int(region_reuse_limit)):
                        self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
                        if save_regions:
                            self._save_regions(root)
                    board = self.boards[slot]
                    usage = self.usages[slot]
                    oracle = self._oracle_for(board)
                    try:
                        instance = sampler.build_instance(
                            board=board,
                            usage_index=usage.sampled_days + 1,
                            instance_index=instance_index,
                            num_customers=int(num_customers),
                            num_charging_stations=int(num_charging_stations),
                            max_attempts=max_attempts,
                            oracle=oracle,
                        )
                        usage.record_day(
                            instance.active_customer_ids,
                            board.cluster_labels,
                            int(self.config.get("freshness", {}).get("recent_window", 30)),
                        )
                        break
                    except Exception as exc:  # Keep generation robust; failed candidates are reported, not saved.
                        last_error = str(exc)
                        failed_attempt_rows.append({
                            "instance_index": instance_index,
                            "outer_attempt": outer_attempt + 1,
                            "region_id": board.region_id,
                            "error": last_error,
                        })
                        self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
                        if save_regions:
                            self._save_regions(root)
                if instance is None or board is None:
                    raise RuntimeError(f"Failed to generate instance_{instance_index:06d}: {last_error}")

                if bundle_instances:
                    assert bundle_file is not None
                    pickle.dump(instance.to_pickle_dict(), bundle_file, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    save_pickle(instance_dir / f"{instance.instance_id}.pkl", instance)
                row = summarize_instance(instance)
                instance_rows.append(row)
                if save_plots and instance_index < plot_limit:
                    write_region_svg(self.boards[slot], plots_dir / f"{instance.instance_id}_active_day.svg", instance=instance)
                if clear_oracle_after_instance:
                    self.oracles.pop(board.region_id, None)
        finally:
            if bundle_file is not None:
                bundle_file.close()

        if save_regions:
            self._save_regions(root)
        region_rows = [summarize_region(board, usage) for board, usage in zip(self.boards, self.usages)]
        write_reports(root, region_rows, instance_rows, failed_attempt_rows)
        if dataset_metadata is not None:
            metadata_dir = ensure_dir(root / "metadata")
            with (metadata_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(dataset_metadata, f, indent=2)
        if save_plots:
            for idx, board in enumerate(self.boards[:plot_limit]):
                write_region_svg(board, plots_dir / f"{board.region_id}_service_territory.svg")

        return {
            "save_path": str(root),
            "instances_path": str(instances_path if bundle_instances else instance_dir),
            "instances_dir": str(root if bundle_instances else instance_dir),
            "num_instances": int(num_instances),
            "num_regions_in_pool": int(len(self.boards)),
            "num_service_territories_in_pool": int(len(self.boards)),
            "generated_instance_rows": instance_rows,
            "region_rows": region_rows,
            "failed_attempt_rows": failed_attempt_rows,
        }
