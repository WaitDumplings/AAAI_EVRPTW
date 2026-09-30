from __future__ import annotations

from pathlib import Path

import numpy as np

from evrptw_hierarchy.core.models import ActiveInstance, RegionBoard
from evrptw_hierarchy.io.persistence import ensure_dir


def _scale_points(points: np.ndarray, width: int, height: int, pad: int) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    out = (arr - lo) * scale + pad
    out[:, 1] = height - out[:, 1]
    return out


def write_region_svg(board: RegionBoard, path: str | Path, instance: ActiveInstance | None = None) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    width, height, pad = 1200, 900, 30
    base_points = [board.road_nodes]
    if instance is not None:
        base_points.append(instance.customers)
        base_points.append(instance.charging_stations)
    all_points = np.vstack(base_points)
    scaled_all = _scale_points(all_points, width, height, pad)
    node_scaled = scaled_all[: len(board.road_nodes)]

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g stroke="#d0d7de" stroke-width="1" opacity="0.65">',
    ]
    for u, v in board.road_edges:
        p, q = node_scaled[int(u)], node_scaled[int(v)]
        lines.append(f'<line x1="{p[0]:.2f}" y1="{p[1]:.2f}" x2="{q[0]:.2f}" y2="{q[1]:.2f}"/>')
    lines.append('</g>')

    if len(board.customers):
        cust_scaled = node_scaled[board.customer_node_ids]
        lines.append('<g fill="#7d8590" opacity="0.38">')
        for p in cust_scaled:
            lines.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="1.35"/>')
        lines.append('</g>')

    if len(board.cluster_gateway_node_ids):
        gw_scaled = node_scaled[board.cluster_gateway_node_ids]
        lines.append('<g fill="#6f42c1" opacity="0.95">')
        for p in gw_scaled:
            lines.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="4.0"/>')
        lines.append('</g>')

    cs_scaled = node_scaled[board.cs_node_ids]
    lines.append('<g fill="#0969da" stroke="#ffffff" stroke-width="0.8">')
    for p in cs_scaled:
        lines.append(f'<rect x="{p[0] - 2.4:.2f}" y="{p[1] - 2.4:.2f}" width="4.8" height="4.8"/>')
    lines.append('</g>')

    depot = node_scaled[int(board.depot_node_id)]
    lines.append(f'<circle cx="{depot[0]:.2f}" cy="{depot[1]:.2f}" r="7" fill="#cf222e" stroke="#ffffff" stroke-width="1.5"/>')

    if instance is not None:
        offset = len(board.road_nodes)
        active_cust_scaled = scaled_all[offset : offset + len(instance.customers)]
        offset += len(instance.customers)
        active_cs_scaled = scaled_all[offset : offset + len(instance.charging_stations)]
        lines.append('<g fill="#fb8500" opacity="0.88">')
        for p in active_cust_scaled:
            lines.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="2.1"/>')
        lines.append('</g>')
        lines.append('<g fill="#0096c7" stroke="#001d3d" stroke-width="0.8">')
        for p in active_cs_scaled:
            lines.append(f'<rect x="{p[0] - 4:.2f}" y="{p[1] - 4:.2f}" width="8" height="8"/>')
        lines.append('</g>')

    title = f"{board.region_id}"
    if instance is not None:
        title += f" / {instance.instance_id}"
    lines.append(f'<text x="24" y="32" font-family="Arial" font-size="18" fill="#24292f">{title}</text>')
    lines.append('</svg>')
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
