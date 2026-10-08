"""Two-process host tests: global moments, PopArt/Adam units and collective failure."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from offline2online import reward_normalization as rn
from offline2online.distributed import DistributedContext


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 1)
        self.use_decomposed_critic = False

    def forward(self, inputs):
        output = self.head(inputs)
        normalizer = getattr(self, '_value_normalizer', None)
        return output if normalizer is None else normalizer.denormalize(output)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = Critic()


def config(**training):
    return {'training': {'reward_norm_mode': rn.MODE, 'gamma': 1., 'gae_lambda': .95,
                         'normalization_beta': .25, 'require_complete_feasible_rollouts': False, **training},
            'env': {'reward_contract': 'strict_distance', 'normalize_reward': True,
                    'reward_distance_scale_km': 2., 'failure_penalty_km': 1000.},
            'offline': {'method': 'sl_ppo'}}


def model_optimizer():
    torch.manual_seed(77)
    agent = Agent().double()
    optimizer = torch.optim.Adam(agent.parameters(), lr=.001, amsgrad=True)
    for parameter in agent.parameters():
        optimizer.state[parameter] = {'step': torch.tensor(1.), 'exp_avg': torch.full_like(parameter, .3),
                                     'exp_avg_sq': torch.full_like(parameter, .4),
                                     'max_exp_avg_sq': torch.full_like(parameter, .5)}
    return agent, optimizer


def assert_states_equal(actual_agent, actual, actual_optimizer, expected_agent, expected, expected_optimizer):
    for left, right in zip(actual_agent.parameters(), expected_agent.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'step'):
            torch.testing.assert_close(actual_optimizer.state[left][name], expected_optimizer.state[right][name], rtol=1e-11, atol=1e-11)
    for module in ('actor', 'critic'):
        for name, value in getattr(actual, module).state_dict().items():
            torch.testing.assert_close(value, getattr(expected, module).state_dict()[name], rtol=1e-12, atol=1e-12)


def batch_with_failures(success):
    success = torch.as_tensor(success, dtype=torch.bool).reshape(1, -1)
    costs = torch.arange(1, success.numel() + 1).reshape_as(success).float() * 10.
    initial = torch.full_like(costs, .25)
    rewards = (-costs / 2. - (~success).float() * 500. - initial).unsqueeze(0)
    return SimpleNamespace(rewards=rewards, valid=torch.ones_like(rewards, dtype=torch.bool),
                           dones=torch.ones_like(rewards, dtype=torch.bool), initial_shaping_potential=initial), costs, success


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        context = DistributedContext(rank=rank, world_size=2, device='cpu')
        agent, optimizer = model_optimizer()
        controller = rn.configure(agent, config(), distributed=context)
        reference_agent, reference_optimizer = model_optimizer()
        reference = rn.configure(reference_agent, config())
        target_parts = [torch.tensor([-10., -20., float('nan')]), torch.tensor([-30., -40., -50., -60.])]
        raw_parts = [torch.tensor([1., 2., float('nan')]), torch.tensor([-3., 4., 5., -6.])]
        valid_parts = [torch.tensor([True, True, False]), torch.tensor([True] * 4)]
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10.
        prediction_before = agent.critic(x).detach()
        normalized = controller.begin_rollout_update(target_parts[rank], raw_parts[rank], valid_parts[rank], agent.critic, optimizer)
        expected = reference.begin_rollout_update(torch.cat(target_parts), torch.cat(raw_parts), torch.cat(valid_parts), reference_agent.critic, reference_optimizer)
        index = slice(0, 3) if rank == 0 else slice(3, 7)
        torch.testing.assert_close(normalized, expected[index])
        torch.testing.assert_close(agent.critic(x), prediction_before, rtol=1e-10, atol=1e-10)
        assert_states_equal(agent, controller, optimizer, reference_agent, reference, reference_optimizer)
        assert controller.diagnostics['normalizer_valid_actions'] == 6
        assert controller.diagnostics['normalizer_statistics_scope'] == 'global_valid_actions'

        # A real synchronized Adam step agrees with the concatenated objective.
        # Count weighting here is explicit; the generic trainer's rank-mean
        # gradient policy is deliberately outside this normalizer's contract.
        target = target_parts[rank][valid_parts[rank]].double()
        inputs = torch.arange(len(target) * 3, dtype=torch.float64).reshape(-1, 3) / 10. + rank
        all_inputs = [torch.arange(6, dtype=torch.float64).reshape(2, 3) / 10.,
                      torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10. + 1]
        loss = ((controller.critic.normalize(agent.critic(inputs).squeeze(-1)) - controller.critic.normalize(target)) ** 2).sum() / 6 * 2
        optimizer.zero_grad(set_to_none=True); loss.backward(); context.synchronize_gradients(agent); optimizer.step()
        targets = torch.cat([part[mask] for part, mask in zip(target_parts, valid_parts)]).double()
        ref_loss = ((reference.critic.normalize(reference_agent.critic(torch.cat(all_inputs)).squeeze(-1)) - reference.critic.normalize(targets)) ** 2).mean()
        reference_optimizer.zero_grad(set_to_none=True); ref_loss.backward(); reference_optimizer.step()
        assert_states_equal(agent, controller, optimizer, reference_agent, reference, reference_optimizer)

        # One rank may have no local valid samples; global calibration/history
        # and both head/optimizer transformations must still be identical.
        second_targets = [torch.tensor([float('nan')]), torch.tensor([-100., -300.])]
        second_raw = [torch.tensor([float('nan')]), torch.tensor([10., -20.])]
        second_valid = [torch.tensor([False]), torch.tensor([True, True])]
        previous_scale = controller.actor.snapshot()
        controller.begin_rollout_update(second_targets[rank], second_raw[rank], second_valid[rank], agent.critic, optimizer)
        reference.begin_rollout_update(torch.cat(second_targets), torch.cat(second_raw), torch.cat(second_valid), reference_agent.critic, reference_optimizer)
        torch.testing.assert_close(controller.scale_used, previous_scale)
        assert_states_equal(agent, controller, optimizer, reference_agent, reference, reference_optimizer)

        # A rank with all failed episodes is permitted; it still contributes to
        # PPO moments, while the solution-level signal is zero on that rank.
        batch, costs, success = batch_with_failures([False, False] if rank == 0 else [True, False])
        actual_costs, identity_error = controller.validate_rollout(batch, costs, success)
        torch.testing.assert_close(actual_costs, costs)
        assert identity_error < 1e-5
        assert not controller.route_advantages(costs, success).any()
        failed_returns = batch.rewards.squeeze(0)
        controller.begin_rollout_update(failed_returns, failed_returns, batch.valid.squeeze(0), agent.critic, optimizer)
        assert controller.diagnostics['normalizer_valid_actions'] == 4

        # Failure/cutoff/NaN on one rank cannot leave the other inside a moment
        # collective. All workers see the error and can reach another barrier.
        for case in ('collector_cutoff', 'identity', 'nonfinite', 'shape'):
            before = {key: value.clone() for key, value in controller.actor.state_dict().items()}
            failed = False
            try:
                if case in ('collector_cutoff', 'identity'):
                    bad_batch, bad_costs, bad_success = batch_with_failures([True, False])
                    if rank == 1:
                        if case == 'collector_cutoff':
                            bad_batch.dones[-1, 0, 1] = False
                        else:
                            bad_batch.rewards[0, 0, 0] += 1.
                    controller.validate_rollout(bad_batch, bad_costs, bad_success)
                else:
                    targets = torch.tensor([1., float('nan') if rank == 1 and case == 'nonfinite' else 2.])
                    raw = torch.ones(1 if rank == 1 and case == 'shape' else 2)
                    controller.begin_rollout_update(targets, raw, torch.ones(2, dtype=torch.bool), agent.critic, optimizer)
            except ValueError:
                failed = True
            assert failed, case
            for key, value in before.items():
                torch.testing.assert_close(controller.actor.state_dict()[key], value, rtol=0, atol=0)
            dist.barrier()

        # Different per-rank normalization configurations are rejected together.
        bad_agent, _ = model_optimizer()
        divergent = config()
        if rank == 1:
            divergent['env']['reward_distance_scale_km'] = 3.
        with pytest.raises(ValueError, match='differs across ranks'):
            rn.configure(bad_agent, divergent, distributed=context)
        dist.barrier()
        torch.save({'success': True, 'updates': int(controller.actor.update_count)}, Path(output) / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_process_global_host_matches_serial_and_errors_collectively(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert torch.load(tmp_path / f'rank{rank}.pt', weights_only=True)['success']


def test_failure_learning_requires_strict_units_and_remains_opt_in():
    agent, _ = model_optimizer()
    cfg = config()
    cfg['env'].pop('reward_contract')
    with pytest.raises(ValueError, match='strict_distance'):
        rn.configure(agent, cfg)
    agent, _ = model_optimizer()
    controller = rn.configure(agent, config(require_complete_feasible_rollouts=True))
    batch, costs, success = batch_with_failures([False, False])
    with pytest.raises(ValueError, match='complete feasible'):
        controller.validate_rollout(batch, costs, success)
