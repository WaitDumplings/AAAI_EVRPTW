"""Authoritative scoring can audit historical decoding without changing it."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from offline2online import trainer
from test_vrptw_eval_validation import instance, row


@pytest.mark.parametrize('env_explicit,override,expected_explicit', [
    (False, None, False), (True, None, True), (False, True, True), (True, False, False)])
def test_independent_scoring_override_keeps_environment_physics(tmp_path, monkeypatch,
        env_explicit, override, expected_explicit):
    data = instance()
    data.travel_time_matrix_s = data.distance_matrix_km.copy()
    data.travel_time_matrix_s[0, 3] = 3.  # Violates due=1002 only in authoritative seconds.
    monkeypatch.setattr(trainer, '_eval_instance_batches', lambda *a, **k: iter([[data]]))
    env_options = []
    def environment(**kwargs):
        env_options.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(trainer, 'make_terran_env', environment)
    exported = dict(row(), min_objective_distance_km=5., median_objective_distance_km=5.,
        vehicle_count=2, min_vehicle_count=2, median_vehicle_count=2,
        traj_feasible_rate=1., feasible_traj_count=1, runtime_s=.1)
    monkeypatch.setattr(trainer, '_rollout_eval_batch_min_median', lambda *a, **k: [dict(exported)])
    cfg = dict(data=dict(problem_type='vrptw', num_customers=3),
        env=dict(prefer_explicit_edge_matrices=env_explicit),
        evaluation=dict(eval_path=str(tmp_path), eval_output_dir=str(tmp_path/'eval'), eval_save_routes=True))
    if override is not None:
        cfg['evaluation']['prefer_explicit_edge_matrices'] = override
    metrics = trainer.evaluate_fixed_dataset(torch.nn.Linear(1, 1), cfg, seed=31, epoch=1, device='cpu')
    saved = json.loads((tmp_path/'eval/epoch_0001.jsonl').read_text())
    assert saved['route_validation']['valid'] is not expected_explicit
    assert metrics['eval_feasible_rate'] == float(not expected_explicit)
    assert saved['route_validation']['travel_time_source'] == (
        'provided_travel_time_matrix_s' if expected_explicit else 'distance_over_effective_speed')
    assert env_options[0]['prefer_explicit_edge_matrices'] is env_explicit
    assert cfg['env']['prefer_explicit_edge_matrices'] is env_explicit
