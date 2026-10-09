#!/usr/bin/env python3
"""Print a read-only training snapshot, optionally refreshing in the terminal.

Usage (from CaliRoute):
    python scripts/watch_comparison.py
    python scripts/watch_comparison.py results/optimization/<run> --watch 30
    python scripts/watch_comparison.py /path/to/comparison.json
    python scripts/watch_comparison.py <run> --reference-km 224.46 --reference-epochs 1500

EVRPTW100 snapshots automatically show the user-reported historical validation
best: 224.46 km from a 1500-epoch run (evaluation protocol not yet verified).

No third-party packages, training imports, GPU queries or file writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'optimization'
DISTANCE = 'eval_avg_objective_distance_km'


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fmt(value, digits=2):
    value = number(value)
    return '--' if value is None else f'{value:.{digits}f}'


def percent(value):
    value = number(value)
    return '--' if value is None else f'{100 * value:.1f}%'


def table(headers, rows):
    rows = [[str(v) for v in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for index, row in enumerate(rows):
        print('  '.join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print('  '.join('-' * width for width in widths))


def source_path(value):
    if value:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            comparison = path / 'comparison.json'
            return comparison if comparison.exists() else path / 'status.json'
        return path
    files = list(RESULTS.glob('*/comparison.json'))
    if not files:
        raise FileNotFoundError(f'No comparison.json under {RESULTS}; pass a run directory or JSON path.')
    return max(files, key=lambda path: path.stat().st_mtime)


def evaluation(value):
    if not isinstance(value, dict) or value.get('eval_status') not in (None, '', 'ok'):
        return '--'
    distance = number(value.get(DISTANCE))
    return '--' if distance is None else f'{distance:.2f}@e{fmt(value.get("epoch"), 0)}'


def historical_reference(arms, protocol, reference_km=None, reference_epochs=None):
    if reference_km is None:
        # Never apply the EVRPTW100 reference to VRPTW/CVRP or unknown tasks.
        if str(protocol.get('task', '')).lower() != 'evrptw100':
            return
        reference_km = 224.46
        if reference_epochs is None:
            reference_epochs = 1500
    duration = f'; {reference_epochs}-epoch run' if reference_epochs is not None else ''
    print(f'\nHistorical validation reference: {reference_km:.2f} km{duration}')
    print('User-reported best during that run, not necessarily its final epoch; evaluation protocol UNVERIFIED.')
    rows = []
    for name, arm in arms.items():
        latest = arm.get('latest_validation') or {}
        best = arm.get('best_checkpoint') or {}
        values = []
        for result in (latest, best):
            distance = number(result.get(DISTANCE))
            if (result.get('eval_status') not in (None, '', 'ok')
                    or distance is None or distance < 0):
                values.extend(['--', '--'])
            else:
                delta = distance - reference_km
                values.extend([f'{delta:+.2f}', f'{100 * delta / reference_km:+.2f}%'])
        rows.append([name, *values, percent(best.get('eval_feasible_rate'))])
    table(['Arm', 'Latest delta km', 'Latest gap', 'Best delta km', 'Best gap', 'Best FR'], rows)
    print('Gap = (current km / reference km - 1) x 100%; negative is lower distance.')
    print('Check validation split, feasibility, charging rules and best-of-N before claiming improvement.')


def combined_snapshot(first, second):
    """Join separately supervised arms only when their comparison budgets agree."""
    if not all(isinstance(d, dict) and isinstance(d.get('arms'), dict) and d['arms']
               for d in (first, second)):
        raise ValueError('Both snapshots must have a nonempty arms mapping.')
    keys = ('task', 'seed', 'epochs', 'global_batch', 'n_traj',
            'ppo_update_epochs', 'eval_interval', 'eval_n_traj', 'validation_instances')
    a, b = first.get('protocol') or {}, second.get('protocol') or {}
    for key in keys:
        if a.get(key) is None or a.get(key) != b.get(key):
            raise ValueError(f'Cannot compare snapshots: protocol {key} differs or is missing.')
    if set(first['arms']) & set(second['arms']):
        raise ValueError('Cannot compare snapshots with overlapping arm names.')
    result = dict(first)
    result['arms'] = {**first['arms'], **second['arms']}
    result['initial_validation_by_arm'] = {
        **(first.get('initial_validation_by_arm') or {}),
        **(second.get('initial_validation_by_arm') or {})}
    matched = {}
    for source in (first, second):
        for epoch, rows in (source.get('matched_validation_epochs') or {}).items():
            if isinstance(rows, dict):
                matched.setdefault(epoch, {}).update({name: row for name, row in rows.items()
                                                     if name in source['arms']})
    result['matched_validation_epochs'] = matched
    states = [d.get('state', '--') for d in (first, second)]
    result['state'] = states[0] if states[0] == states[1] else ' / '.join(states)
    # Report the older snapshot timestamp so the combined view cannot look fresher.
    stamps = [d.get('updated_at_utc') for d in (first, second)]
    result['updated_at_utc'] = min(stamps, key=lambda x: datetime.fromisoformat(x.replace('Z', '+00:00'))) if all(stamps) else None
    return result


def render(path, reference_km=None, reference_epochs=None, compare_with=None):
    data = json.loads(path.read_text(encoding='utf-8'))
    if compare_with is not None:
        data = combined_snapshot(data, json.loads(compare_with.read_text(encoding='utf-8')))
    if not isinstance(data, dict) or not isinstance(data.get('arms'), dict):
        raise ValueError('Expected a comparison/status object with an arms mapping.')
    arms = data['arms']
    protocol = data.get('protocol') or {}
    stamp = data.get('updated_at_utc')
    print(f'Run: {path.parent.name} | State: {data.get("state", "--")}')
    print(f'Source: {path}')
    if compare_with is not None:
        print(f'Compare with: {compare_with} (snapshot age uses the older source)')
    age = '--'
    if stamp:
        try:
            updated = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
            if updated.tzinfo is not None:
                age = f'{max(0, (datetime.now(timezone.utc) - updated).total_seconds()):.0f}s'
        except (ValueError, TypeError):
            pass
    print(f'Updated: {stamp or "--"} | Snapshot age: {age}')
    if protocol:
        print(f'PPO passes: {fmt(protocol.get("ppo_update_epochs"), 0)} | '
              f'Batch: {fmt(protocol.get("global_batch"), 0)} x '
              f'{fmt(protocol.get("n_traj"), 0)} trajectories | '
              f'Eval every {fmt(protocol.get("eval_interval"), 0)} epochs, '
              f'best-of-{fmt(protocol.get("eval_n_traj"), 0)}')
    print()
    rows = []
    for name, arm in arms.items():
        # Only formal-run fields; preflight_progress is intentionally separate.
        train = arm.get('latest_train_row') or {}
        latest = arm.get('latest_validation') or {}
        best = arm.get('best_checkpoint') or {}
        progress = f'{fmt(arm.get("completed_training_epochs"), 0)}/{fmt(arm.get("target_epochs", protocol.get("epochs")), 0)}'
        state = arm.get('state', '--')
        if arm.get('stage') == 'preflight':
            state += '/preflight'
        kl = number(train.get('global_approx_kl'))
        if kl is None:
            kl = number(train.get('approx_kl'))
        lr = number(train.get('learning_rate'))
        valid_eval = latest.get('eval_status') in (None, '', 'ok')
        rows.append([name, arm.get('gpu', '--'), state, progress,
                     evaluation(latest), percent(latest.get('eval_feasible_rate')) if valid_eval else '--',
                     evaluation(best), fmt(kl, 5), percent(train.get('clip_fraction')),
                     f'{lr:.1e}' if lr is not None else '--'])
    table(['Arm', 'GPU', 'State', 'Epoch', 'Latest km@epoch', 'Eval FR',
           'Best ckpt km@epoch', 'Train KL', 'Clip%', 'LR'], rows)

    if protocol.get('ppo_warmup_epochs'):
        warmup = int(protocol['ppo_warmup_epochs'])
        print(f'\nSchedule: epochs 1-{warmup} pure PPO; epoch {warmup+1} onward SL-PPO; same optimizer.')
        table(['Arm', 'Training phase', 'Phase epoch'], [
            [name, (arm.get('latest_train_row') or {}).get('training_phase', '--'),
             fmt((arm.get('latest_train_row') or {}).get('phase_epoch'), 0)]
            for name, arm in arms.items()])

    # New diagnostics are optional. Keep old snapshots and the main table
    # unchanged; a missing measurement must not be presented as zero drift.
    if any(any(key in (arm.get('latest_train_row') or {})
               for key in ('ppo_passes_executed', 'post_update_kl'))
           for arm in arms.values()):
        print('\nPPO update diagnostics (latest training row)')
        rows = []
        for name, arm in arms.items():
            train = arm.get('latest_train_row') or {}
            fresh_kl = number(train.get('post_update_kl'))
            # The trainer records zero monitoring time on unsampled epochs;
            # display -- there too, rather than implying a zero-cost KL check.
            kl_time = number(train.get('post_update_kl_time_s')) if fresh_kl is not None else None
            rows.append([name, fmt(train.get('epoch'), 0),
                         fmt(train.get('ppo_passes_executed'), 0),
                         fmt(fresh_kl, 5), fmt(kl_time, 3)])
        table(['Arm', 'Epoch', 'Executed passes', 'Fresh KL', 'KL time (s)'], rows)
        print('Fresh KL uses the updated policy on sampled rollout actions; Train KL is the update aggregate.')
        print('-- means no measurement in this row; fresh KL need not be measured every epoch.')

    matched = data.get('matched_validation_epochs') or {}
    expected_count = number(protocol.get('validation_instances'))
    complete = {}
    for epoch, values in matched.items():
        epoch_number = number(epoch)
        if epoch_number is None or not epoch_number.is_integer() or epoch_number < 0:
            continue
        if not isinstance(values, dict) or not arms or not all(name in values for name in arms):
            continue
        good = True
        for name in arms:
            row = values[name]
            if not isinstance(row, dict):
                good = False
                break
            distance, feasible, count = (number(row.get(key)) for key in
                                         (DISTANCE, 'eval_feasible_rate', 'eval_num_instances'))
            if (distance is None or distance < 0 or feasible is None or not 0 <= feasible <= 1
                    or count is None or count <= 0 or not count.is_integer()
                    or (expected_count is not None and count != expected_count)
                    or row.get('eval_status') not in (None, '', 'ok')):
                good = False
                break
        if good and len({number(values[name]['eval_num_instances']) for name in arms}) == 1:
            complete[int(epoch_number)] = values
    print()
    if complete:
        epoch = max(complete)
        print(f'Latest common complete validation: epoch {epoch} (km; lower is better)')
        initial = data.get('initial_validation_by_arm') or {}
        rows = []
        for name in arms:
            start = (initial.get(name) or {}).get('distance_km')
            if number(start) is None:
                start = complete.get(0, {}).get(name, {}).get(DISTANCE)
            row = complete[epoch][name]
            rows.append([name, fmt(start), fmt(row[DISTANCE]),
                         percent(row['eval_feasible_rate']), fmt(row['eval_num_instances'], 0)])
        table(['Arm', 'Initial km', f'Epoch {epoch} km', 'Feasible', 'Instances'], rows)
    else:
        print('No common complete validation is recorded in this JSON yet.')
    historical_reference(arms, protocol, reference_km, reference_epochs)
    print('\nBest ckpt follows trainer selection; epoch 0 may be excluded. KL is the logged training aggregate.')
    print('JSON snapshot only: running/GPU fields do not verify live processes or GPU utilization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', help='Run directory or comparison/status JSON; default: newest comparison.json')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='Refresh interval, e.g. 30; Ctrl-C to exit')
    parser.add_argument('--compare-with', help='Second run directory or JSON; combine separate original/optimized runs')
    parser.add_argument('--reference-km', type=float,
                        help='Historical validation best in km; default: 224.46 only for task evrptw100')
    parser.add_argument('--reference-epochs', type=int,
                        help='Total epochs of the historical run, not the epoch of its best checkpoint')
    args = parser.parse_args()
    if args.reference_km is not None and (not math.isfinite(args.reference_km) or args.reference_km <= 0):
        parser.error('--reference-km must be a finite positive number')
    if args.reference_epochs is not None and args.reference_epochs <= 0:
        parser.error('--reference-epochs must be a positive integer')
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 1):
        parser.error('--watch must be a finite number >= 1 second')
    # Pin the auto-selected experiment so another run cannot silently replace it.
    selected = args.path
    while True:
        if args.watch is not None and sys.stdout.isatty():
            print('\033[2J\033[H', end='')
        try:
            path = source_path(selected)
            if selected is None:
                selected = str(path.parent)
            peer = source_path(args.compare_with) if args.compare_with else None
            render(path, args.reference_km, args.reference_epochs, peer)
        except (OSError, ValueError) as exc:
            print(f'Cannot read snapshot: {exc}', file=sys.stderr, flush=True)
            if args.watch is None:
                return 1
        sys.stdout.flush()
        if args.watch is None:
            return 0
        time.sleep(args.watch)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped watching.')
