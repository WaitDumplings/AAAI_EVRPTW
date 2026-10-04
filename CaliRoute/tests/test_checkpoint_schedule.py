"""Recovery checkpoints are independent of expensive validation and archives."""

import pytest
import torch

from offline2online.checkpoint_schedule import epoch_checkpoint_plan


@pytest.mark.parametrize(
    "epoch, expected",
    [
        (1, (False, False, False)),
        (5, (False, False, True)),
        (20, (True, False, True)),
        (50, (False, True, True)),
        (1000, (True, True, True)),
    ],
)
def test_rolling_checkpoints_do_not_change_validation_or_archives(epoch, expected):
    plan = epoch_checkpoint_plan(
        epoch, 1000, eval_interval=20, checkpoint_interval=50,
        latest_checkpoint_interval=5,
    )
    assert (plan.evaluate, plan.archive, plan.latest) == expected
    assert plan.capture_resume_state == any(expected)


def test_legacy_default_creates_no_latest_and_keeps_final_archive():
    for epoch in (1, 5, 20, 1000):
        plan = epoch_checkpoint_plan(epoch, 1000, eval_interval=0, checkpoint_interval=0)
        assert not plan.latest
        assert not plan.evaluate
        assert plan.archive == (epoch == 1000)
        assert plan.capture_resume_state == (epoch == 1000)


def test_final_epoch_gets_recovery_checkpoint_even_off_interval():
    plan = epoch_checkpoint_plan(7, 7, eval_interval=20, checkpoint_interval=50, latest_checkpoint_interval=5)
    assert plan.evaluate and plan.archive and plan.latest and plan.capture_resume_state


def test_uncompleted_epoch_cannot_be_checkpointed():
    with pytest.raises(ValueError, match='completed epoch'):
        epoch_checkpoint_plan(0, 1000, eval_interval=20, checkpoint_interval=50, latest_checkpoint_interval=5)


def test_interrupted_latest_write_preserves_previous_complete_checkpoint(tmp_path, monkeypatch):
    from offline2online.trainer import save_checkpoint, _load_training_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / 'checkpoint_latest.pt'
    model._training_resume_state = {
        'version': 1, 'world_size': 2,
        'ranks': [{'rank': 0, 'optimizer_steps': 80}, {'rank': 1, 'optimizer_steps': 80}],
        'completed_epoch': 5, 'next_training_epoch': 6,
        'snapshot_stage': 'training_complete_before_evaluation', 'evaluation_pending': False,
    }
    save_checkpoint(path, model, optimizer, {}, epoch=5, seed=3009)
    original_bytes = path.read_bytes()
    model._training_resume_state = {
        **model._training_resume_state, 'completed_epoch': 20,
        'next_training_epoch': 21, 'evaluation_pending': True,
    }
    original_save = torch.save

    def partial_write_then_fail(value, destination):
        destination.write_bytes(b'incomplete replacement')
        raise OSError('simulated interrupted save')

    monkeypatch.setattr(torch, 'save', partial_write_then_fail)
    with pytest.raises(OSError, match='interrupted save'):
        save_checkpoint(path, model, optimizer, {}, epoch=20, seed=3009)
    assert path.read_bytes() == original_bytes
    assert torch.load(path, weights_only=False)['epoch'] == 5

    monkeypatch.setattr(torch, 'save', original_save)
    save_checkpoint(path, model, optimizer, {}, epoch=20, seed=3009)
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.02)
    info = _load_training_checkpoint(restored, restored_optimizer, path, 'cpu')
    assert info['epoch'] == 20 and info['optimizer_loaded']
    assert restored_optimizer.param_groups[0]['lr'] == .01
    assert restored._pending_training_resume_state['world_size'] == 2
    assert len(restored._pending_training_resume_state['ranks']) == 2
    assert restored._pending_training_resume_state['next_training_epoch'] == 21
    assert restored._pending_training_resume_state['evaluation_pending'] is True
    assert not path.with_name('.checkpoint_latest.pt.tmp').exists()
