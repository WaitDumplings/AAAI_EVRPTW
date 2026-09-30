# Legacy Gurobi benchmarks

CVRP, VRPTW, and EVRPTW solvers imported from
[WaitDumplings/gurobi_mul](https://github.com/WaitDumplings/gurobi_mul), revision
`c891bedcb2fa125891bc55f464470d8fafbd3e0f` (2026-06-11). These are the earlier
AAAI batch solvers, before the ICLR solver optimizations. `SOURCE.json` records
the original Python file hashes.

The MILP formulations, callbacks, time limits, resume behavior, and EVRPTW
warm starts and vehicle-count tie-break are preserved. Integration changes cover
package imports, dataset/output paths, test-split selection, and launch scripts.
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

The range runners resume by skipping IDs already in their output summary;
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
EVRPTW charging-station copies default to 4 in the shell launcher, 2 in
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
