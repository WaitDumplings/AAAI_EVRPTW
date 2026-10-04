import math
from types import SimpleNamespace

import pytest

from offline2online.training_schedule import apply_epoch_schedule, schedule_for_epoch


def test_long_schedule_warmup_decay_and_resume_are_epoch_deterministic():
    cfg = {'training': {'epochs': 1000, 'learning_rate': 5e-5 * math.sqrt(2), 'lr_schedule': 'warmup_cosine',
                        'lr_warmup_epochs': 20, 'lr_min': 1e-5, 'ent_coef': .01, 'entropy_final_coef': .002}}
    opt = SimpleNamespace(param_groups=[{'lr': .1}])
    at_1 = apply_epoch_schedule(cfg, opt, 1)
    assert at_1['learning_rate'] == pytest.approx(5e-5 * math.sqrt(2) / 20)
    assert at_1['ent_coef'] == .01
    at_20 = apply_epoch_schedule(cfg, opt, 20)
    assert at_20['learning_rate'] == pytest.approx(5e-5 * math.sqrt(2))
    at_500 = apply_epoch_schedule(cfg, opt, 500)
    at_end = apply_epoch_schedule(cfg, opt, 1000)
    assert at_end['learning_rate'] == 1e-5
    assert at_end['ent_coef'] == .002
    assert apply_epoch_schedule(cfg, opt, 500) == at_500
    assert opt.param_groups[0]['lr'] == at_500['learning_rate']


def test_legacy_default_schedule_does_not_anneal():
    cfg = {'training': {'epochs': 3, 'learning_rate': 1e-4, 'ent_coef': .01}}
    assert schedule_for_epoch(cfg, 1) == schedule_for_epoch(cfg, 3)


@pytest.mark.parametrize('update', [{'lr_warmup_epochs': 100}, {'lr_min': 1}, {'learning_rate': float('nan')}])
def test_invalid_schedule_fails_before_training(update):
    cfg = {'epochs': 100, 'lr_schedule': 'warmup_cosine', 'learning_rate': 1e-4, **update}
    with pytest.raises(ValueError):
        schedule_for_epoch(cfg, 1)
