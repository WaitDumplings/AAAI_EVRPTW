#!/usr/bin/env python3
"""Print a read-only training snapshot, optionally refreshing in the terminal.

Usage (from CaliRoute):
    python scripts/watch_comparison.py
    python scripts/watch_comparison.py results/optimization/<run> --watch 30
    python scripts/watch_comparison.py /path/to/comparison.json

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


def render(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('arms'), dict):
        raise ValueError('Expected a comparison/status object with an arms mapping.')
    arms = data['arms']
    protocol = data.get('protocol') or {}
    stamp = data.get('updated_at_utc')
    print(f'Run: {path.parent.name} | State: {data.get("state", "--")}')
    print(f'Source: {path}')
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
    print('\nBest ckpt follows trainer selection; epoch 0 may be excluded. KL is the logged training aggregate.')
    print('JSON snapshot only: running/GPU fields do not verify live processes or GPU utilization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', help='Run directory or comparison/status JSON; default: newest comparison.json')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='Refresh interval, e.g. 30; Ctrl-C to exit')
    args = parser.parse_args()
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
            render(path)
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
