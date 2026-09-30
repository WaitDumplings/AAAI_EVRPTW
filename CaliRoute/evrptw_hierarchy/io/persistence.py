from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_pickle(path: str | Path, obj: Any) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    payload = obj.to_pickle_dict() if hasattr(obj, "to_pickle_dict") else obj
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out
