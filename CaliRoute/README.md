# CaliRoute

CaliRoute is the training code for real-road CVRP, VRPTW, and EVRPTW
experiments. The repository is organized around one main research line:

```text
PPO backbone -> SL-PPO
```

`SL-PPO` is the proposed method. `PPO`, `DAPG`, and `AWBC` are comparison
methods that share the same backbone and environment interface.

## Data Layout

The code and dataset live under the same `AAAI_EVRPTW` workspace:

```text
AAAI_EVRPTW/
  CaliRoute/
  AAAI_Dataset/               # downloaded separately; excluded from Git
    dataset/                 # train/val splits and reference solutions
    test_release/            # frozen final test splits
```

Dataset download link: **to be added**. Extract the download into
`AAAI_EVRPTW/AAAI_Dataset/`.

By default `train.py` resolves data from `../AAAI_Dataset/dataset` relative to
the CaliRoute code root. You can override this with `--data-root`.
The frozen test set is stored separately at `../AAAI_Dataset/test_release`,
with splits such as `evrptw/test/Cus50/`. It is reserved for final evaluation.

Expected split layout:

```text
AAAI_Dataset/dataset/
  cvrp/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
  vrptw/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
  evrptw/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
```

Each split directory should contain:

```text
instances.pkl
metadata.json
expert_solutions.csv
gurobi_summary.csv
gurobi_time_trace.csv
public_metadata.json
```

## Quick Start

From the `AAAI_EVRPTW` repository root, install dependencies and run SL-PPO
on EVRPTW Cus50:

```bash
cd CaliRoute
pip install -r requirements.txt
python train.py \
  --problem evrptw \
  --customers 50 \
  --charging-stations 10 \
  --offline-method slppo \
  --device cuda:0
```

The command automatically uses:

```text
../AAAI_Dataset/dataset/evrptw/train/Cus50
../AAAI_Dataset/dataset/evrptw/val/Cus50
../AAAI_Dataset/dataset/evrptw/train/Cus50/expert_solutions.csv
../AAAI_Dataset/dataset/evrptw/val/Cus50/gurobi_summary.csv
```

Inspect the generated config without launching training:

```bash
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo --dry-run --print-config
```

## Methods

All methods use the same PPO rollout/update backbone:

```bash
python train.py --problem cvrp   --customers 50 --offline-method ppo
python train.py --problem vrptw  --customers 50 --offline-method dapg
python train.py --problem evrptw --customers 50 --charging-stations 10 --offline-method awbc
python train.py --problem evrptw --customers 50 --charging-stations 10 --offline-method slppo
```

Default update counts:

```text
PPO/DAPG/AWBC: ppo_update_epochs = 3
SL-PPO:        ppo_update_epochs = 4
```

SL-PPO uses a solution-level objective with group-relative and reference
advantages. Its candidate pool is configurable:

```bash
--pool weighted   # default priority sampler
--pool best       # always choose the currently highest-priority instances
--pool off        # no priority sampler
```

The bundled multi-method launch scripts use `weighted` by default. Set
`SLPPO_POOL=best` before running a script to switch the SL-PPO branch to
best-pool.

The expert-candidate term is exposed with descriptive public names:

```bash
--sl-expert-candidate-weight 0.60
```

## Paper Launch Scripts

The scripts under `scripts/paper_2080ti_runs/` assume this layout:

```text
AAAI_EVRPTW/
  CaliRoute/
  AAAI_Dataset/
    dataset/
    test_release/
```

Activate the Python environment, then run a full pipeline with `all` from the
CaliRoute root:

```bash
bash scripts/paper_2080ti_runs/run_evrptw_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_vrptw_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_vrptw_cus50_all_methods.sh all
bash scripts/paper_2080ti_runs/run_cvrp_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_cvrp_cus50_all_methods.sh all
```

The scripts require a run group such as `all`, or a GPU id and method name.
For example, run PPO on GPU 0:

```bash
bash scripts/paper_2080ti_runs/run_evrptw_cus15_all_methods.sh 0 ppo
```

`all` starts PPO and then launches DAPG, SL-PPO, and AWBC after the PPO
initialization checkpoint is available (epoch 100 by default). Single-method
DAPG, SL-PPO, and AWBC launches wait for that checkpoint. Scripts detach by
default; append `--foreground` to keep the launcher in the current terminal.

Default GPU mapping is:

```text
PPO   -> GPU0
DAPG  -> GPU1
SL-PPO -> GPU2
AWBC  -> GPU3
```

Override only the GPU ids if needed:

```bash
GPU_PPO=0 GPU_DAPG=1 GPU_SLPPO=2 GPU_AWBC=3 \
  bash scripts/paper_2080ti_runs/run_vrptw_cus50_all_methods.sh all
```

Cus15 and Cus50 use the same fixed 2080Ti-safe rollout/update geometry from
`scripts/paper_2080ti_runs/common_2080ti_config.sh`:

```text
num_envs=64, n_traj=50, ppo_step_chunk_size=16,
num_minibatches=4, eval_batch_size=128
```

SL-PPO uses `ppo_update_epochs=4`; PPO, DAPG, and AWBC use
`ppo_update_epochs=3`. These values are intentionally shared across methods so
the comparison does not change rollout or minibatch geometry.

## Ablations

Backbone and decoder ablations are controlled from the same entry point:

```bash
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo \
  --dde on \
  --qkv-delta none \
  --action-key on \
  --action-bias on \
  --distance-injection encoder
```

Supported switches:

```text
--dde on/off
--qkv-delta none/k/v/kv
--action-key on/off
--action-bias on/off
--distance-injection encoder/none
```

Encoder distance injection is implemented as road-distance attention bias.
Embedding-level distance injection is intentionally not exposed until it is
implemented as a separate model path.

## Generated Outputs

Training outputs are written under `results/` and are intentionally ignored by
git. The sibling `AAAI_Dataset/` directory is also excluded from the
`AAAI_EVRPTW` Git repository and is distributed through a separate download.

## Project Structure

```text
train.py, main.py              Public training entry points.
caliroute/                     Method presets, config builder, CLI, ablation flags.
offline2online/                Stable training engine retained for reproducibility.
configs/templates/             Reference generated configs.
scripts/paper_2080ti_runs/     Paper experiment launch scripts.
scripts/validation/            Short regression checks against old runs.
docs/                          Engineering notes and review documents.
```

Generated logs and checkpoints are written under `results/`, which is ignored by
git.

## Gurobi benchmarks

`EVRPTW_Benchmark/Exact/Gurobi_Solver/{CVRP,VRPTW,EVRPTW}/` contains the original
AAAI Gurobi batch solvers. They use the sibling `AAAI_Dataset` and write results
under `results/gurobi/`. The solvers preserve the earlier `gurobi_mul`
formulations; see the [benchmark guide](EVRPTW_Benchmark/Exact/Gurobi_Solver/README.md)
for setup, dry runs, test-set selection, and EVRPTW warm starts.

For a full frozen test batch, choose only the problem and scale:

```bash
bash scripts/run_gurobi_test.sh evrptw 50
```

Problems: `vrptw`, `evrptw`, `cvrp`; scales: `15`, `50`, `100`. This starts all
1,000 test instances in the background with 30 workers, one Gurobi thread per
worker, and a 2-hour optimization limit per instance. Results and logs are under
`results/gurobi/<problem>/test/CusN/`. For EVRPTW, the launcher automatically uses
1 copy per physical charging station for Cus15 and 2 copies for Cus50/Cus100.
EVRPTW's optional vehicle-count tie-break is disabled to avoid a second 2-hour
solve; see the benchmark guide for details.

Resume is automatic: re-run the same command to skip completed instance IDs in
that task/scale's `gurobi_summary.csv`. `TIME_LIMIT` results count as completed;
errors, interrupted runs, and missing results retry. The shell shows the checked
summary path; the background log reports exact skipped/pending counts before
solving. If no instances remain, the runner exits without creating workers.

To split EVRPTW Cus100 across five servers, add that server's shard number:

```bash
bash scripts/run_gurobi_test.sh evrptw 100 1  # Use 1, 2, 3, 4, or 5 on each server.
```

Each shard owns 200 consecutive entries in the same frozen bundle, selected before
resume. Shards cover all 1,000 instances without overlap and keep the 30-worker,
2-hour, 2-CS-copy settings. Outputs and resume state are separate under
`results/gurobi/evrptw/test/Cus100/shard_N_of_5/`; `test_shard.json` records the
assigned IDs. Reuse the same shard number when resuming. Without the third
argument, the command still runs the full bundle. See the benchmark guide for
all five commands.
