# Legacy Gurobi benchmarks

CVRP, VRPTW, and EVRPTW solvers imported from
[WaitDumplings/gurobi_mul](https://github.com/WaitDumplings/gurobi_mul), revision
`c891bedcb2fa125891bc55f464470d8fafbd3e0f` (2026-06-11). These are the earlier
AAAI batch solvers, before the ICLR solver optimizations. `SOURCE.json` records
the original Python file hashes.

The MILP formulations, callbacks, optimize-call time limits, and EVRPTW
warm starts and vehicle-count tie-break are preserved. Integration changes cover
package imports, dataset/output paths, test-split selection, launch scripts, and
consistent result-based resume checks.
Each solver retains its original schema under its own package; CVRP and VRPTW
have different `classical_core` schemas.

## Setup

Run from `AAAI_EVRPTW/CaliRoute`:

```bash
pip install -r EVRPTW_Benchmark/Exact/Gurobi_Solver/requirements.txt
```

Actual solves require a working Gurobi license and a compatible `gurobipy`
version. The range runner's `--dry-run` needs neither Gurobi nor dataset loading.
Gurobi remains an optional dependency for CaliRoute training.

## Data and outputs

Default input paths are resolved relative to the checkout, independent of the
current working directory:

- train/val: `AAAI_Dataset/dataset/<problem>/<split>/CusN`
- test: `AAAI_Dataset/test_release/<problem>/test/CusN`

Default outputs are under
`CaliRoute/results/gurobi/<problem>/<split>/CusN/`:

```text
gurobi_summary.csv
gurobi_time_trace.csv
solutions/
```

The dataset and frozen test files are inputs. Results stay outside the dataset
and are ignored by Git. No historical result files are bundled with this code.

`--dataset_path` selects a bundle file or split/scale directory directly.
`--dataset_root` selects a problem-specific directory containing `train/val/test`
and `CusN` beneath each split. `--output_path` overrides the results directory.
Explicit relative paths are relative to the caller's working directory.

## Full frozen test run (two arguments)

From `AAAI_EVRPTW/CaliRoute`, activate the Python environment with Gurobi installed,
then choose only the problem and customer scale:

```bash
bash scripts/run_gurobi_test.sh vrptw 15
bash scripts/run_gurobi_test.sh evrptw 50
bash scripts/run_gurobi_test.sh cvrp 100
```

Each command immediately returns after starting an independent background batch.
Accepted problems are `vrptw`, `evrptw`, and `cvrp`; scales are `15`, `50`, and `100`.
The script also works from another directory when invoked by its absolute path.
`python scripts/run_gurobi_test.py evrptw 50` is an equivalent entry point.

Fixed settings:

- Input: `AAAI_Dataset/test_release/<problem>/test/CusN/instances.pkl`.
  Each of the nine bundles has 1,000 instances. Metadata is checked before launch;
  the whole bundle is read without filtering numeric instance-ID suffixes.
- 30 worker processes, each solving one instance at a time with one Gurobi thread.
  A separate coordinator writes results as workers finish.
- 7,200 seconds (2 hours) of Gurobi optimization per instance, with zero target MIP
  gap. Instances proven optimal finish earlier. Model construction and saving add
  overhead; the whole 1,000-instance batch takes longer than 2 hours.
- Incumbent checkpoints at 60, 300, 900, 3,600, and 7,200 seconds.
- EVRPTW copies per physical charging station depend on scale: Cus15 uses 1;
  Cus50 and Cus100 each use 2. The launcher sets this automatically.
  Its optional vehicle-count tie-break is disabled for this entry point because
  the legacy second optimization would receive another 7,200-second budget.

Results go to `results/gurobi/<problem>/test/CusN/`. Each launch prints its PID,
log file, a `tail -f` command, and a command to stop that batch's process group.
Logs and per-launch PID files are in that output directory's `logs/`; `launcher.pid`
records the latest coordinator PID. PID files remain after completion as records.
An inherited lock prevents concurrent launches of the same problem/scale through
this script and releases when the batch exits. Different problem/scale pairs may
run concurrently, each using its own 30 workers.

Resume is always enabled for this two-argument launcher. Before starting the
background runner, it checks that problem/scale's `gurobi_summary.csv` and prints
the summary path and number of recorded completed IDs. The runner then matches
actual input instance IDs against those results before submitting any worker
jobs. It logs the exact matching skipped/pending counts, for example:

```text
Resume: enabled; skipped=237 pending=763; summary=.../gurobi_summary.csv
```

All three problems use the same rules:

- Finished results such as `OPTIMAL`, `TIME_LIMIT`, and `INFEASIBLE` are skipped.
  Reaching the 2-hour limit counts as finished even if no feasible solution was found.
- Missing results, blank statuses, `ERROR`, `INVALID_INSTANCE`, `INTERRUPTED`,
  `LOADED`, and `INPROGRESS` are retried. Checkpoint files alone do not count as a
  finished result; unfinished instances start a new solve.
- Status matching ignores case and surrounding whitespace. If duplicate IDs occur
  in the summary, the last row is used. Unrelated IDs do not affect the actual
  skipped/pending counts for the selected bundle.
- If no instances remain, the runner logs `No pending instances; nothing to solve.`
  and exits without creating a worker pool or performing a license preflight.

Completed summary rows are retained as new results are saved. The recorded-ID
count shown by the shell is historical; exact counts for the current input bundle
appear in the log. Launch settings and the exact solver command are also recorded
there, along with any worker startup or license failures. The general range
launchers below use these same resume rules when `--skip_completed` is enabled.

## Run a batch

Inspect paths and arguments first:

```bash
python -m EVRPTW_Benchmark.Exact.Gurobi_Solver.EVRPTW.run_range \
  --split val --scale Cus15 --start_index 0 --end_index 10 \
  --workers 2 --threads 1 --cs_copies 2 --dry-run

python -m EVRPTW_Benchmark.Exact.Gurobi_Solver.CVRP.run_range \
  --split val --scale Cus50 --start_index 0 --end_index 10 \
  --workers 2 --threads 1 --dry-run

python -m EVRPTW_Benchmark.Exact.Gurobi_Solver.VRPTW.run_range \
  --split val --scale Cus50 --start_index 0 --end_index 10 \
  --workers 2 --threads 1 --dry-run
```

Remove `--dry-run` to solve. Indices are half-open: `[start_index, end_index)`.
Use `--split test` for final evaluation on the frozen release. Both module
execution and direct execution of `.../<problem>/run_range.py` are supported.

The range runners resume by skipping finished results in their output summary;
`--no_skip_completed` requests recomputation. Give concurrent independent jobs
different output directories. Existing legacy CSV upsert behavior is preserved.

The legacy default time limit is 7200 seconds per optimize call. In EVRPTW, the
optional vehicle-count tie-break can start a second capped solve after the
distance objective is proven optimal. For a shorter EVRPTW run, also shorten
the checkpoint schedule, e.g. `--time_limit_s 60 --checkpoints_s 60`.

## Shell entry points

```bash
bash EVRPTW_Benchmark/Exact/Gurobi_Solver/EVRPTW/run_gurobi_range.sh \
  --cus 15 --split val --start 0 --end 10 --workers 2 --threads 1 \
  --cs-copies 2 --dry-run
```

Replace `EVRPTW` with `CVRP` or `VRPTW` and omit `--cs-copies` for classical
problems. The wrappers use the active `python` (`PYTHON_BIN` or `--python` can
override it); `--conda-env NAME` optionally selects a Conda environment.
`--detach` runs in the background; logs default to
`results/gurobi/<problem>/logs/`. A dry run creates no logs or result directories.

Legacy shell defaults are retained: 24 workers, 1 thread per worker, Cus15,
indices `[0,100)`, and 7200 seconds. The split defaults to val for CVRP and train
for VRPTW/EVRPTW. Set the split explicitly when comparing methods.
EVRPTW charging-station copies default to 4 in the legacy `run_gurobi_range.sh`, 2 in
`run_range.py`, and 3 in the low-level runner/solver, matching the old entry
points. Set this parameter explicitly for reproducible comparisons.

## Low-level runner and EVRPTW refinement

Each problem also exposes `run_gurobi.py`, taking explicit `--dataset_path` and
`--save_path`. EVRPTW retains `--expert_summary_path` for warm-start refinement:

```bash
python -m EVRPTW_Benchmark.Exact.Gurobi_Solver.EVRPTW.run_gurobi \
  --dataset_path ../AAAI_Dataset/dataset/evrptw/train/Cus15 \
  --expert_summary_path ../AAAI_Dataset/dataset/evrptw/train/Cus15/gurobi_summary.csv \
  --save_path results/gurobi/evrptw/refine/Cus15 \
  --scales Cus15 --start_index 0 --end_index 10 --workers 2 --threads 1 \
  --cs_copies 2 --time_limit_s 1800 --checkpoints_s 1800
```

The old `--evrptw_root` override is removed: this benchmark uses its bundled
legacy schema. The separate 12-hour single-instance experiment is not included.
