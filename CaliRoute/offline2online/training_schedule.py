"""Reproducible schedules for larger-batch fine-tuning; no optimizer state reset."""
from __future__ import annotations

import math
from typing import Any


def schedule_for_epoch(cfg: dict[str, Any], epoch: int) -> dict[str, float]:
    train = cfg.get('training', cfg)
    total = int(train.get('epochs', 1))
    if not 1 <= int(epoch) <= total:
        raise ValueError('epoch must be within the configured training horizon')
    peak = float(train.get('learning_rate', 1e-4))
    initial_entropy = float(train.get('entropy_initial_coef', train.get('ent_coef', .01)))
    final_entropy = float(train.get('entropy_final_coef', initial_entropy))
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not all(math.isfinite(v) and v >= 0 for v in (initial_entropy, final_entropy)):
        raise ValueError('entropy coefficients must be finite and nonnegative')
    mode = str(train.get('lr_schedule', 'constant'))
    if mode == 'constant':
        lr = peak
    elif mode == 'warmup_cosine':
        warmup = int(train.get('lr_warmup_epochs', 20))
        floor = float(train.get('lr_min', peak * .1))
        if not 0 <= warmup < total or not math.isfinite(floor) or not 0 <= floor <= peak:
            raise ValueError('require 0 <= warmup < epochs and 0 <= lr_min <= learning_rate')
        if epoch <= warmup:
            lr = peak * epoch / max(warmup, 1)
        else:
            progress = (epoch - max(warmup, 1)) / max(total - max(warmup, 1), 1)
            lr = floor + (peak - floor) * .5 * (1 + math.cos(math.pi * progress))
    else:
        raise ValueError('lr_schedule must be constant or warmup_cosine')
    fraction = (epoch - 1) / max(total - 1, 1)
    entropy = final_entropy + (initial_entropy - final_entropy) * .5 * (1 + math.cos(math.pi * fraction))
    return {'learning_rate': lr, 'ent_coef': entropy}


def apply_epoch_schedule(cfg, optimizer, epoch):
    train = cfg.setdefault('training', {})
    train.setdefault('entropy_initial_coef', float(train.get('ent_coef', .01)))
    values = schedule_for_epoch(cfg, epoch)
    for group in optimizer.param_groups:
        group['lr'] = values['learning_rate']
    train['ent_coef'] = values['ent_coef']
    return values
