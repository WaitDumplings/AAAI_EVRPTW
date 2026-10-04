"""Tensor-only directed road injection for attention-based routing models.

Contract: directed road matrices [B,N,N] (or [N,N]) and optional physical
quantities in consistent units -> additive attention bias [B,H,N,N]. A caller
retains ownership of hard feasibility masks and graph/depot tokens. No dataset,
environment or trainer imports are required. Cache the output only while both
the instance and model weights remain unchanged.
"""
from __future__ import annotations

import torch
from torch import nn


class RoadDistanceInjection(nn.Module):
    """Bounded, zero-initialized residual with scale-aware directed features.

    Outgoing distance ratios and depot detour savings express local route
    structure; signed reverse-edge differences retain road directionality.
    Time and energy use their own horizon/capacity instead of sharing a scale.
    Defaults also support a distance-only host model. Hard masks are NOT inferred.
    """

    feature_dim = 10

    def __init__(self, n_heads: int, hidden_dim: int = 32, max_bias: float = 2.0):
        super().__init__()
        if int(n_heads) < 1 or int(hidden_dim) < 1 or float(max_bias) <= 0:
            raise ValueError("heads, hidden_dim and max_bias must be positive")
        self.n_heads = int(n_heads)
        self.max_bias = float(max_bias)
        self.net = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_heads))
        self.head_gate = nn.Parameter(torch.zeros(n_heads))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.diagnostics_enabled = False
        self._diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _matrix(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.shape[-1] != value.shape[-2]:
            raise ValueError("road matrices must have shape [B,N,N] or [N,N]")
        return value.float()

    def features(self, distance: torch.Tensor, travel_time: torch.Tensor | None = None,
                 energy: torch.Tensor | None = None, time_windows: torch.Tensor | None = None,
                 service_time: torch.Tensor | None = None,
                 battery_capacity: torch.Tensor | None = None) -> torch.Tensor:
        distance = self._matrix(distance)
        B, N, _ = distance.shape
        # Ignore diagonal, zero and unreachable entries when setting the scale.
        valid = torch.isfinite(distance) & (distance > 0)
        clean = torch.where(valid, distance, torch.zeros_like(distance))
        scale = (clean.sum((-1, -2), keepdim=True) / valid.sum((-1, -2), keepdim=True).clamp_min(1)).clamp_min(1e-6)
        row_scale = (clean.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)).clamp_min(1e-6)
        dist = torch.nan_to_num(distance, nan=0., posinf=1e6, neginf=0.)
        travel = torch.zeros_like(dist) if travel_time is None else self._matrix(travel_time).to(dist)
        use_energy = torch.zeros_like(dist) if energy is None else self._matrix(energy).to(dist)
        if time_windows is None:
            tw = dist.new_zeros(B, N, 2)
            tw[..., 1] = 1.0
        else:
            tw = time_windows.to(dist)
            if tw.ndim == 2:
                tw = tw.unsqueeze(0)
        if tw.shape != (B, N, 2):
            raise ValueError("time_windows must have shape [B,N,2] or [N,2]")
        horizon = torch.nan_to_num(tw[..., 1], nan=1., posinf=1., neginf=1.).amax(-1).clamp_min(1e-6)[:, None, None]
        service = dist.new_zeros(B, N) if service_time is None else service_time.to(dist).reshape(B, N)
        capacity = dist.new_ones(B, 1, 1) if battery_capacity is None else torch.as_tensor(battery_capacity, device=dist.device, dtype=dist.dtype).reshape(-1, 1, 1).clamp_min(1e-6)
        arrival = tw[..., 0].unsqueeze(-1) + service.unsqueeze(-1) + travel
        # Node zero is the depot; hosts using a different depot order must reorder.
        detour_saving = dist[:, :, :1] + dist[:, :1, :] - dist
        features = torch.stack((
            dist / scale,
            dist / row_scale,
            (dist - dist.transpose(-1, -2)) / scale,
            travel / horizon,
            use_energy / capacity,
            (tw[..., 1].unsqueeze(-2) - arrival) / horizon,
            torch.relu(tw[..., 0].unsqueeze(-2) - arrival) / horizon,
            ((tw[..., 1] - tw[..., 0]).unsqueeze(-2) / horizon).expand(B, N, N),
            detour_saving / scale,
            (tw[..., 0].unsqueeze(-2) - tw[..., 0].unsqueeze(-1)) / horizon,
        ), dim=-1)
        features = torch.nan_to_num(features, nan=0., posinf=1e6, neginf=-1e6)
        # Smooth signed compression preserves differences without allowing
        # sentinel/unreachable values to dominate the small residual network.
        return (features.sign() * torch.log1p(features.abs())).clamp(-5., 5.)

    def forward(self, distance: torch.Tensor, travel_time: torch.Tensor | None = None,
                energy: torch.Tensor | None = None, time_windows: torch.Tensor | None = None,
                service_time: torch.Tensor | None = None,
                battery_capacity: torch.Tensor | None = None,
                base_bias: torch.Tensor | None = None) -> torch.Tensor:
        features = self.features(distance, travel_time, energy, time_windows, service_time, battery_capacity)
        gate = 2.0 * torch.sigmoid(self.head_gate)
        residual = self.max_bias * torch.tanh(self.net(features) * gate / self.max_bias)
        residual = residual.permute(0, 3, 1, 2)
        if self.diagnostics_enabled:
            with torch.no_grad():
                self._diagnostics = {
                    'feature_rms': features.square().mean().sqrt().detach(),
                    'feature_abs_max': features.abs().amax().detach(),
                    'gate_mean': gate.mean().detach(), 'gate_std': gate.std(unbiased=False).detach(),
                    'residual_rms': residual.float().square().mean().sqrt().detach(),
                    'residual_saturation': (residual.abs() > 0.95 * self.max_bias).float().mean().detach(),
                }
                if base_bias is not None:
                    base = torch.where(torch.isfinite(base_bias), base_bias, torch.zeros_like(base_bias))
                    self._diagnostics['residual_to_base'] = self._diagnostics['residual_rms'] / base.float().square().mean().sqrt().clamp_min(1e-6)
        return residual

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Detached device tensors; the host chooses when to synchronize/log."""
        return dict(self._diagnostics)
