"""Shared result-based resume checks, without importing Gurobi or loading data."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


# Numeric fallbacks are Gurobi LOADED, INTERRUPTED, and INPROGRESS statuses.
_RETRY_STATUSES = {"", "ERROR", "INVALID_INSTANCE", "INTERRUPTED", "LOADED", "INPROGRESS", "1", "11", "14"}


def completed_instance_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Use the latest summary row per ID; a time-limited solve is completed."""
    statuses: dict[str, str] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id:
            statuses[instance_id] = (
                str(row.get("status_name") or "").strip().upper()
                or str(row.get("status") or "").strip().upper()
            )
    return {instance_id for instance_id, status in statuses.items() if status not in _RETRY_STATUSES}


def read_completed_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return set()
        if "instance_id" not in reader.fieldnames:
            raise ValueError(f"Missing instance_id column in {summary_path}")
        return completed_instance_ids(reader)
