#!/usr/bin/env python3
"""Render portable PNG/SVG monitoring curves from a running comparison."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from run_plugin_comparison import recorded_session_times


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def render(experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = json.loads((experiment / 'manifest.json').read_text())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    charts = [
        (f"Validation best-of-{manifest['protocol'].get('eval_n_traj', 50)} distance (km)", 'eval_avg_min_objective_distance_km', 'eval'),
        ('Validation distance vs recorded active-session time', 'eval_avg_min_objective_distance_km', 'time'),
        ('Validation feasibility', 'eval_feasible_rate', 'eval'),
        ('PPO approximate KL', 'global_approx_kl', 'train'),
        ('Policy entropy', 'global_entropy', 'train'),
        ('Critic explained variance', 'value_explained_variance', 'monitor'),
        ('Learning rate', 'learning_rate', 'train'),
        ('Gradient norm before clipping', 'grad_norm', 'train'),
        ('Effective replay weight', 'policy_replay_weight', 'train'),
    ]
    for arm, info in manifest['arms'].items():
        train = read_rows(Path(info['log_dir']) / 'train_log.csv')
        evaluation = read_rows(Path(info['log_dir']) / 'eval_log.csv')
        monitor = experiment / arm / 'monitoring/monitor_rank_0.jsonl'
        monitored = []
        if monitor.exists():
            for line in monitor.read_text().splitlines():
                try:
                    row = json.loads(line)
                    # Monitor fields may be grouped; flatten one diagnostic level.
                    for key, value in list(row.items()):
                        if isinstance(value, dict):
                            row.update(value)
                    monitored.append(row)
                except json.JSONDecodeError:
                    pass
        by_epoch = {}
        for row in recorded_session_times(train):
            if row['recorded_active_session_seconds'] is not None:
                by_epoch[int(row['epoch'])] = row['recorded_active_session_seconds'] / 60
        for axis, (title, key, kind) in zip(axes.flat, charts):
            rows = monitored if kind == 'monitor' else evaluation if kind in ('eval', 'time') else train
            values = []
            for row in rows:
                try:
                    x = by_epoch.get(int(row['epoch'])) if kind == 'time' else int(row['epoch'])
                    y = float(row[key])
                    if x is not None:
                        values.append((x, y))
                except (ValueError, KeyError, TypeError):
                    pass
            if values:
                axis.plot(*zip(*values), label=arm, linewidth=1.4)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel('Recorded active-session time (min)\nExcludes downtime, discarded work, unrecorded tails'
                            if kind == 'time' else 'Epoch', fontsize=9)
            axis.grid(alpha=.2)
    axes[1, 0].axhline(.02, color='grey', linestyle='--', linewidth=.8)
    for axis in axes.flat:
        if axis.lines:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
    title = 'CVRP plug-in comparison — one-seed screening, validation only'
    report_path = experiment / 'comparison.json'
    if report_path.exists():
        progress = json.loads(report_path.read_text()).get('progress')
        if progress:
            completed = ', '.join(f"{arm}: {item['completed_training_epochs']}" for arm, item in progress['arms'].items())
            title += f"\n{progress['run_state']} | target {progress['target_epochs']} epochs | completed {completed}"
    fig.suptitle(title)
    for extension in ('png', 'svg'):
        fig.savefig(experiment / f'training_curves.{extension}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    render(args.experiment.resolve())
