"""Low-frequency detached plugin and gradient diagnostics for fine-tuning."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .slppo_diagnostics import tensors_to_floats


def begin_monitor_epoch(agent, epoch, train_cfg):
    interval = max(1, int(train_cfg.get('monitor_interval', 20)))
    sampled = int(epoch) == 1 or int(epoch) % interval == 0
    agent._monitor_this_epoch = sampled
    agent._monitor_optimizer_metrics = {}
    agent._monitor_optimizer_sampled = False
    for module in agent.modules():
        if hasattr(module, 'diagnostics_enabled'):
            module.diagnostics_enabled = sampled
    return sampled


def module_update_snapshot(agent):
    if not getattr(agent, '_monitor_this_epoch', False) or getattr(agent, '_monitor_optimizer_sampled', False):
        return {}
    agent._monitor_optimizer_sampled = True
    grouped = {}
    for name, param in agent.named_parameters():
        if not param.requires_grad:
            continue
        group = ('static_fusion' if 'static_fusion' in name else
                 'edge_relations' if any(part in name for part in ('edge_relation_encoder', 'edge_relation_adapter')) else
                 'resource_decoder' if 'resource_decoder' in name else
                 'input_context' if 'physical_input_adapter' in name else
                 'rdi' if any(key in name for key in ('rdi_adapter', 'residual_edge_bias', 'dist_bias_scale', 'type_pair_bias')) else
                 'agda' if 'dynamic_graph_kv_encoder' in name else
                 'critic' if name.startswith('critic.') else None)
        if group is not None:
            grouped.setdefault(group, []).append((param, param.detach().clone()))
    metrics = {}
    for name, values in grouped.items():
        metrics[f'module_{name}_parameter_norm'] = torch.stack([before.float().square().sum() for _, before in values]).sum().sqrt()
        grads = [p.grad.detach().float().square().sum() for p, _ in values if p.grad is not None]
        metrics[f'module_{name}_grad_norm_unclipped'] = torch.stack(grads).sum().sqrt() if grads else metrics[f'module_{name}_parameter_norm'].new_zeros(())
    agent._monitor_optimizer_metrics.update(metrics)
    return grouped


def finish_module_update(agent, snapshot, skipped=False):
    if snapshot:
        parameter = next(agent.parameters())
        agent._monitor_optimizer_metrics["module_first_attempt_skipped"] = parameter.new_tensor(float(skipped))
    for name, values in snapshot.items():
        update = torch.stack([(p.detach().float() - before.float()).square().sum() for p, before in values]).sum().sqrt()
        agent._monitor_optimizer_metrics[f'module_{name}_update_norm'] = update
        agent._monitor_optimizer_metrics[f'module_{name}_update_to_parameter_ratio'] = update / agent._monitor_optimizer_metrics[f'module_{name}_parameter_norm'].clamp_min(1e-12)


def plugin_diagnostics(agent):
    output = dict(getattr(agent, '_monitor_optimizer_metrics', {}))
    if getattr(agent, '_monitor_this_epoch', False):
        for name, module in agent.named_modules():
            if hasattr(module, 'diagnostics_enabled') and callable(getattr(module, 'diagnostics', None)):
                prefix = ('static_fusion' if 'static_fusion' in name else
                          'resource_decoder' if 'resource_decoder' in name else
                          'input_context' if 'physical_input_adapter' in name else
                          'rdi' if 'rdi_adapter' in name else 'agda' if 'agda_adapter' in name else name.replace('.', '_'))
                output.update({f'{prefix}_{key}': value for key, value in module.diagnostics().items()})
    return tensors_to_floats(output)


def average_diagnostics(records):
    if not records:
        return {}
    keys = set().union(*(row.keys() for row in records))
    return {key: float(np.mean([row[key] for row in records if isinstance(row.get(key), (float, int, np.number))]))
            for key in keys if any(isinstance(row.get(key), (float, int, np.number)) for row in records)}


def append_monitor_row(path, row):
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + '\n')
