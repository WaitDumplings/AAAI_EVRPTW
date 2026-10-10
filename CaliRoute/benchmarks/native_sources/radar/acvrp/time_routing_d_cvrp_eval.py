#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from ACVRPEnv import ACVRPEnv as Env
from ACVRPModel import ACVRPModel as Model


def infer_problem_size(data_file):
    data = np.load(data_file, allow_pickle=False)
    if 'distance_matrix' in data.files:
        return int(data['distance_matrix'].shape[1] - 1)
    if 'dist' in data.files:
        return int(data['dist'].shape[1] - 1)
    raise KeyError(f'Cannot infer problem size from {data_file}')


def load_routing_d_npz(data_file):
    data = np.load(data_file, allow_pickle=False)
    if 'distance_matrix' in data.files:
        dist = data['distance_matrix'].astype(np.float32)
    elif 'dist' in data.files:
        dist = data['dist'].astype(np.float32)
    else:
        raise KeyError(f'{data_file} must contain distance_matrix or dist')

    demand = data['demand'].astype(np.float32)
    node_cnt = dist.shape[1] - 1
    if demand.ndim != 2:
        raise ValueError(f'Expected 2-D demand, got {demand.shape}')
    if demand.shape[1] == node_cnt + 1:
        demand = demand[:, 1:]
    if demand.shape[1] != node_cnt:
        raise ValueError(f'Expected {node_cnt} customer demands, got {demand.shape[1]}')
    if 'capacity' in data.files:
        capacity = data['capacity'].astype(np.float32)
        demand = demand / capacity[:, None]
    return dist, demand


def build_model_params(problem_size):
    head_num = 8
    embedding_dim = 256
    qkv_dim = embedding_dim // head_num
    return {
        'embedding_dim': embedding_dim,
        'sqrt_embedding_dim': embedding_dim ** 0.5,
        'encoder_layer_num': 5,
        'qkv_dim': qkv_dim,
        'sqrt_qkv_dim': qkv_dim ** 0.5,
        'head_num': head_num,
        'init': 'svd',
        'att_type': 'normal',
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1 / 2) ** 0.5,
        'ms_layer2_init': (1 / 16) ** 0.5,
        # Keep the repo's original typo/value. In ACVRPModel this falls through to greedy eval.
        'eval_type': 'softma',
        'one_hot_seed_cnt': problem_size,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Time RADAR CVRP fixed-data eval one instance at a time.')
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    problem_size = infer_problem_size(args.data_file)
    dist_np, demand_np = load_routing_d_npz(args.data_file)
    total = dist_np.shape[0] if args.limit is None else min(dist_np.shape[0], args.limit)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        # RADAR's original Trainer relies on CUDA as the default tensor type.
        # Keep that behavior so tensors created inside ACVRPModel.forward are on GPU.
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    env = Env(node_cnt=problem_size, pomo_size=problem_size)
    model = Model(**build_model_params(problem_size)).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    objs = []
    per_instance_s = []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start_total = time.perf_counter()
    with torch.no_grad():
        for i in range(total):
            problems = torch.from_numpy(dist_np[i:i + 1]).to(device=device, dtype=torch.float32)
            demands = torch.from_numpy(demand_np[i:i + 1]).to(device=device, dtype=torch.float32)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            env.load_problems_manual(problems, demands)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
            max_pomo_reward, _ = reward.max(dim=1)
            best_obj = -max_pomo_reward.float()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            objs.append(float(best_obj.item()))
            per_instance_s.append(end - start)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_total = time.perf_counter() - start_total
    result = {
        'model': 'RADAR',
        'problem': f'CVRP Cus{problem_size}',
        'eval_instances': len(objs),
        'batch_size': 1,
        'pomo_size': problem_size,
        'elapsed_total_s': elapsed_total,
        'avg_s_per_instance_total': elapsed_total / max(1, len(objs)),
        'avg_s_per_instance_model': float(np.mean(per_instance_s)) if per_instance_s else None,
        'p50_s_per_instance_model': float(np.median(per_instance_s)) if per_instance_s else None,
        'avg_best_obj': float(np.mean(objs)) if objs else None,
        'ckpt': str(args.ckpt),
        'data_file': str(args.data_file),
        'device': str(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
