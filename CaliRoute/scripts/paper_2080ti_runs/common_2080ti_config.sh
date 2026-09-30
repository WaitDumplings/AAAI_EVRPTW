#!/usr/bin/env bash

# Shared fixed configuration for 2080Ti-safe paper runs.
# These values are intentionally not read from environment variables: PPO,
# SL-PPO, DAPG, and AWBC must use the same rollout/update batch geometry.
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../AAAI_Dataset/dataset}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-3009}"
EPOCHS="${EPOCHS:-1500}"
INIT_EPOCH="${INIT_EPOCH:-100}"

NUM_ENVS=64
N_TRAJ=50
EVAL_N_TRAJ=50
ROLLOUT_STEPS=90
EVAL_MAX_STEPS=90
PPO_STEP_CHUNK_SIZE=16
NUM_MINIBATCHES=4
EVAL_INTERVAL=20
EVAL_BATCH_SIZE=128
CHECKPOINT_INTERVAL=50
MAX_GPU_MEM_MIB=11000
GPU_MEM_POLL_SECONDS=15
POLL_SECONDS=60

SLPPO_POOL="${SLPPO_POOL:-weighted}"
