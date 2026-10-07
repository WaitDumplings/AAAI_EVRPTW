# AAAI_EVRPTW

This repository contains CaliRoute, the training pipeline for real-road CVRP,
VRPTW, and EVRPTW experiments. SL-PPO is the proposed method; PPO, DAPG, and
AWBC share its backbone and environment interface as comparison methods.

## Repository and Dataset

```bash
git clone git@github.com:WaitDumplings/AAAI_EVRPTW.git
cd AAAI_EVRPTW
```

The local workspace has this layout:

```text
AAAI_EVRPTW/
  CaliRoute/                  Model code and experiment pipeline (tracked in Git).
  AAAI_Dataset/               Downloaded dataset (excluded from Git).
    dataset/                  Training and validation splits, with reference solutions.
    test_release/             Frozen final test splits.
```

Dataset download link: **to be added**. Download and extract the dataset into
`AAAI_Dataset/` beside `CaliRoute/` when the link is available. The existing local
dataset is preserved and is not uploaded to GitHub.

CaliRoute defaults to `../AAAI_Dataset/dataset`, resolved relative to its code
root. The frozen final test set is in `../AAAI_Dataset/test_release`; keep it
separate from training and validation.

## Quick Start

After placing the downloaded dataset in `AAAI_Dataset/`, run from the repository
root:

```bash
cd CaliRoute
pip install -r requirements.txt
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo --dry-run --print-config
```

After installing dependencies and downloading the dataset, start training:

```bash
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo --device cuda:0
```

See [CaliRoute documentation](CaliRoute/README.md) for method settings, launch
scripts, and ablations. Generated logs and checkpoints are excluded from Git.
The reward/normalization experiment branch includes one fixed, compact
initialization under `CaliRoute/assets/reward_norm/` so its launcher works on a
fresh checkout with the local dataset; see the CaliRoute guide for details.

## Gurobi benchmarks

The earlier CVRP, VRPTW, and EVRPTW Gurobi solvers are included under
`CaliRoute/EVRPTW_Benchmark/Exact/Gurobi_Solver/`. See the
[benchmark guide](CaliRoute/EVRPTW_Benchmark/Exact/Gurobi_Solver/README.md) for
optional dependencies, batch commands, and configuration-only dry runs.
