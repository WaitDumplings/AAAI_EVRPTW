# CaliRoute Project Layout

The Git repository root is `AAAI_EVRPTW/`:

```text
AAAI_EVRPTW/
  CaliRoute/                  Training code and experiment pipeline.
  AAAI_Dataset/               Local dataset, excluded from Git.
    dataset/                  Training and validation splits.
    test_release/             Frozen final test splits.
```

Paths below are relative to `CaliRoute/`. The default training data root is
`../AAAI_Dataset/dataset`, resolved from the CaliRoute code root and overridable
with `--data-root`. The final test release lives separately at
`../AAAI_Dataset/test_release`. The dataset download link will be added to the
repository README when available.

`caliroute/` is the paper-facing interface. It owns method presets, dataset-path
resolution, ablation flags, and the CLI.

`offline2online/` is the training engine kept from the experiment workspace. Its
trainer still executes PPO, SL-PPO, AWBC, and DAPG so the existing results remain
reproducible while the repository gets a cleaner public API.

`configs/` keeps historical experiment YAMLs. `configs/templates/` contains
generated reference configs for paper runs.

`scripts/` contains plotting, dataset-building, and old launch/watch utilities.
New training should start from `train.py` in the CaliRoute code root unless an
old exact YAML run must be reproduced.

The intended dependency direction is:

```text
train.py/main.py
  -> caliroute.cli
  -> caliroute.config + caliroute.methods
  -> offline2online.trainer
```

Method ownership:

- `slppo`: main method contribution, configured through `caliroute.methods`.
- `ppo`: shared online backbone and baseline.
- `awbc`, `dapg`: baseline presets sharing the PPO backbone but using separate
  offline losses.

Ablation ownership:

- DDE on/off and dynamic action key/bias flags live in `caliroute.config`.
- Encoder distance injection is implemented by `model.use_encoder_distance_bias`.
- Embedding-level distance injection is intentionally left as a future explicit
  model change, so it cannot accidentally run as a mislabeled ablation.
