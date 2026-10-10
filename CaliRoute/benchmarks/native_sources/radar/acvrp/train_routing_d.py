##########################################################################################
# Train RADAR ACVRP on fixed Routing-D/RRNCO-format CVRP datasets.
# This keeps the original RADAR model/trainer and only replaces dynamic random generation
# with one no-replacement pass over the provided train.npz per epoch.

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from utils.utils import create_logger, copy_all_src
from ACVRPTrainer import ACVRPTrainer as Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, help="Path to train.npz with distance_matrix, demand, capacity.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cuda-device-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3009)
    parser.add_argument("--desc", default="radar_routing_d_acvrp")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable per-epoch permutation of the fixed dataset.")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--img-save-interval", type=int, default=200)
    return parser.parse_args()


def infer_problem_size(data_file):
    data = np.load(data_file, allow_pickle=False)
    if "distance_matrix" in data.files:
        return int(data["distance_matrix"].shape[1] - 1)
    if "dist" in data.files:
        return int(data["dist"].shape[1] - 1)
    if "locs" in data.files:
        return int(data["locs"].shape[1])
    raise KeyError(f"Cannot infer problem size from {data_file}.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    problem_size = infer_problem_size(args.data_file)
    head_num = 8
    embedding_dim = 256
    qkv_dim = embedding_dim // head_num

    env_params = {
        "node_cnt": problem_size,
        "problem_gen_params": {
            "int_min": 0,
            "int_max": 1000 * 1000,
            "scaler": 1000 * 1000,
        },
        "pomo_size": problem_size,
        "fixed_data_path": args.data_file,
        "fixed_shuffle": not args.no_shuffle,
        "fixed_seed": args.seed,
    }

    model_params = {
        "embedding_dim": embedding_dim,
        "sqrt_embedding_dim": embedding_dim ** (1 / 2),
        "encoder_layer_num": 5,
        "qkv_dim": qkv_dim,
        "sqrt_qkv_dim": qkv_dim ** (1 / 2),
        "head_num": head_num,
        "init": "svd",
        "att_type": "normal",
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "ms_hidden_dim": 16,
        "ms_layer1_init": (1 / 2) ** (1 / 2),
        "ms_layer2_init": (1 / 16) ** (1 / 2),
        "eval_type": "softma",
        "one_hot_seed_cnt": problem_size,
    }

    optimizer_params = {
        "optimizer": {
            "lr": 4e-4,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "milestones": [2001, 2101],
            "gamma": 0.1,
        },
    }

    trainer_params = {
        "use_cuda": use_cuda,
        "cuda_device_num": args.cuda_device_num,
        "epochs": args.epochs,
        "train_episodes": args.train_episodes,
        "train_batch_size": args.batch_size,
        "logging": {
            "model_save_interval": args.save_interval,
            "img_save_interval": args.img_save_interval,
            "log_image_params_1": {
                "json_foldername": "log_image_style",
                "filename": "style.json",
            },
            "log_image_params_2": {
                "json_foldername": "log_image_style",
                "filename": "style_loss.json",
            },
        },
        "model_load": {
            "enable": False,
            "path": "",
            "epoch": 0,
        },
    }

    logger_params = {
        "log_file": {
            "desc": args.desc,
            "filename": "log.txt",
        }
    }

    create_logger(**logger_params)
    logger = logging.getLogger("root")
    logger.info(f"DATA_FILE: {args.data_file}")
    logger.info(f"PROBLEM_SIZE: {problem_size}")
    logger.info(f"SEED: {args.seed}")
    logger.info(f"USE_CUDA: {use_cuda}, CUDA_DEVICE_NUM: {args.cuda_device_num}")
    logger.info(f"env_params{env_params}")
    logger.info(f"model_params{model_params}")
    logger.info(f"optimizer_params{optimizer_params}")
    logger.info(f"trainer_params{trainer_params}")

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    main()
