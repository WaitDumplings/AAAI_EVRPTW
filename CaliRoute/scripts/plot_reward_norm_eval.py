#!/usr/bin/env python3
"""Plot completed reward/norm validation observations, without smoothing.

Exports a PNG preview, vector SVG/PDF, raw CSV and a hashed source snapshot.
Running logs may advance later; each chart records when its snapshot was read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

STYLES = {
    'baseline': ('Baseline', '#475569', 'o'),
    'reward': ('Reward', '#D97706', 's'),
    'normalization': ('Norm', '#059669', '^'),
    'combined': ('Reward + Norm', '#7C3AED', 'D'),
}


def read_observations(experiment):
    manifest = json.loads((experiment / 'manifest.json').read_text())
    series, sources, excluded = {}, {}, []
    protocol = manifest['protocol']
    for arm, spec in manifest['arms'].items():
        path = Path(spec['log_dir']) / 'eval_log.csv'
        raw = path.read_bytes()
        sources[arm] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        observed = {}
        for row in csv.DictReader(io.StringIO(raw.decode())):
            if row.get('eval_status') != 'ok':
                excluded.append(dict(arm=arm, epoch=row.get('epoch'), reason='incomplete or unsuccessful validation'))
                continue
            try:
                epoch = int(row['epoch'])
                distance = float(row['eval_avg_objective_distance_km'])
                feasible = float(row['eval_feasible_rate'])
                count = int(row['eval_num_instances'])
                n_traj = int(row['eval_n_traj'])
            except (KeyError, ValueError, TypeError):
                excluded.append(dict(arm=arm, epoch=row.get('epoch'), reason='incomplete metrics'))
                continue
            if not math.isfinite(distance) or distance < 0 or not 0 <= feasible <= 1:
                raise ValueError(f'Invalid validation metric: {arm} epoch {epoch}')
            if count != protocol['validation_instances'] or n_traj != protocol['eval_n_traj']:
                raise ValueError(f'Inconsistent validation protocol: {arm} epoch {epoch}')
            if epoch in observed:
                raise ValueError(f'Duplicate validation epoch: {arm} epoch {epoch}')
            observed[epoch] = dict(arm=arm, epoch=epoch, distance_km=distance,
                feasible_rate=feasible, num_instances=count, n_traj=n_traj)
        if not observed:
            raise ValueError(f'No complete validation observations for {arm}')
        series[arm] = observed
    return manifest, series, sources, excluded


def render(experiment, output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    experiment = Path(experiment).resolve()
    output = Path(output_dir).resolve() if output_dir else experiment / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    manifest, series, sources, excluded = read_observations(experiment)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    common = sorted(set.intersection(*(set(values) for values in series.values())))
    protocol = manifest['protocol']
    input_screen = set(series).issubset({'legacy', 'depot', 'context', 'combined'}) and bool(set(series) & {'legacy', 'depot', 'context'})
    styles = ({'legacy': ('Existing input', '#475569', 'o'),
               'depot': ('Depot + fixed unit', '#D97706', 's'),
               'context': ('Physical context', '#059669', '^'),
               'combined': ('Depot + context', '#7C3AED', 'D')} if input_screen else STYLES)
    title = 'VRPTW100 | Physical input normalization' if input_screen else 'VRPTW100 | Reward and normalization'
    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.20, top=.76)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.text(.09, .935, title, fontsize=21, weight='bold', color='#172033')
    fig.text(.09, .888,
        f"Seed {protocol['seed']}  |  {protocol['validation_instances']:,} validation instances  |  "
        f"Best of {protocol['eval_n_traj']} trajectories  |  Lower is better", fontsize=11, color='#526174')
    for arm, observed in series.items():
        name, color, marker = styles.get(arm, (arm, '#334155', 'o'))
        epochs = sorted(observed)
        distances = [observed[e]['distance_km'] for e in epochs]
        ax.plot(epochs, distances, label=name, color=color, marker=marker,
            markersize=5, linewidth=2.2, markeredgecolor='white', markeredgewidth=.6)
    ax.set_xlabel('Fine-tuning epoch (starts from the shared epoch-300 model)', fontsize=11, labelpad=10)
    ax.set_ylabel('Mean validation distance (km)', fontsize=12, labelpad=10)
    ax.xaxis.set_major_locator(MultipleLocator(int(protocol.get('eval_interval', 20))))
    ax.set_xlim(-2, int(protocol['epochs'])+2)
    ax.grid(axis='y', color='#E2E8F0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, colors='#475569', length=0, pad=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#CBD5E1')
    ax.legend(loc='lower left', bbox_to_anchor=(0,1.02), ncol=4, frameon=False,
        fontsize=11, handlelength=2.7, columnspacing=2.2, borderaxespad=0)
    if common:
        latest_common = common[-1]
        ax.axvline(latest_common, color='#94A3B8', linestyle=(0,(3,4)), linewidth=1, zorder=0)
        ax.text(latest_common-2, .985, f'Latest common eval: {latest_common}', transform=ax.get_xaxis_transform(),
            ha='right', va='top', fontsize=9, color='#64748B')
    feasibility = min(row['feasible_rate'] for obs in series.values() for row in obs.values())
    endpoint_note = '  |  '.join(f"{styles.get(a,(a,))[0]} through {max(obs)}" for a,obs in series.items())
    fig.text(.09,.105,endpoint_note,fontsize=10,color='#334155')
    fig.text(.09,.068,
        f'Raw validation points; no smoothing or extrapolation. Minimum feasibility: {feasibility:.0%}.',
        fontsize=9,color='#64748B')
    fig.text(.09,.035,f'Snapshot: {timestamp}  |  Validation only; local seed, not a test-set result.',fontsize=9,color='#64748B')
    stem = output / 'validation_curves'
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(stem.with_suffix('.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)
    fields = ('arm','epoch','distance_km','feasible_rate','num_instances','n_traj')
    with stem.with_suffix('.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for arm,obs in series.items():
            writer.writerows(obs[e] for e in sorted(obs))
    metadata = dict(experiment=str(experiment),snapshot_utc=timestamp,protocol=protocol,
        sources=sources,excluded_rows=excluded,common_epochs=common,
        latest_epochs={arm:max(obs) for arm,obs in series.items()},
        interpretation='Raw validation distance only; compare arms at matching observed epochs.')
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(stem.with_suffix('.png'))
    return stem.with_suffix('.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    render(args.experiment,args.output_dir)
