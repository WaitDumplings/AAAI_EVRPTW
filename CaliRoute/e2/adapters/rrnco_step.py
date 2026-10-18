"""Experimental per-action RRNCO interface, not a formal E2 training worker.

The native encoder/decoder/environment are unchanged. Replay stores discrete
neighbor samples, never detached policy embeddings. Flattening is start-major
(S B), as in native rl4co batchify. All arithmetic for losses is FP32.
"""
from contextlib import contextmanager
import torch
from torch import nn


@contextmanager
def fixed_neighbors(policy, indices=None):
    expert = policy.encoder.init_embedding.distance_expert
    original = expert._sample_indices
    captured = {}
    def sample(distance, phase, b, n):
        value = original(distance, phase, b, n) if indices is None else indices.to(distance.device)
        if value.shape != (b, n, expert.sample_size):
            raise ValueError('Replay neighborhood shape/order changed')
        captured['indices'] = value.detach().clone()
        return value
    expert._sample_indices = sample
    try:
        yield captured
    finally:
        del expert._sample_indices


class StepCritic(nn.Module):
    """Same critic in all four proposed arms: current/mean node + time/load."""
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*width+2, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, td, cache, starts):
        from rl4co.utils.ops import batchify
        nodes = batchify(cache.node_embeddings, starts) if starts > 1 else cache.node_embeddings
        bi = torch.arange(nodes.size(0), device=nodes.device)
        features = torch.cat((nodes[bi, td['current_node']], nodes.mean(1),
            td['current_time'], 1-td['used_capacity_linehaul']), -1)
        return self.net(features.float()).squeeze(-1)


def encode_static(policy, initial_state, auxiliary_randomness=None):
    with fixed_neighbors(policy, auxiliary_randomness) as sampled:
        embeddings = policy.encoder(initial_state, phase='train')
    return policy.decoder._precompute_cache(embeddings), sampled['indices']


def step_logits(policy, state, cache, starts):
    from rrnco.models.decoding import process_logits
    if not state['action_mask'].any(-1).all():
        raise ValueError('No legal action; replay must not repair the route')
    logits, feasible = policy.decoder(state, cache, starts)
    if not feasible.any(-1).all():
        raise ValueError('No legal action; replay must not repair the route')
    lp = process_logits(logits.float(), feasible, temperature=policy.temperature,
        tanh_clipping=policy.tanh_clipping, mask_logits=policy.mask_logits)
    return lp, feasible


def run_actions(policy, env, batch, critic, *, starts=3, actions=None,
                neighbors=None, cap=201, greedy=False, forced_start=True,
                start_actions=None):
    """Roll out or teacher-force complete routes. Output shape [time, S*B].

    POMO start is external, has logprob=0 and policy_valid=False, but its
    physical reward participates in value/GAE recursion. Expert replay supplies
    its external first customers via start_actions; these are likewise forced,
    excluded from the likelihood, and never fabricated as learned decisions.
    Last implicit return is priced on the terminal transition; padded depot
    actions are excluded from both policy/value masks. Truncation raises.
    """
    from rl4co.utils.ops import batchify
    td = env.reset(batch.clone())
    cache, samples = encode_static(policy, td, neighbors)
    initial_starts = (env.select_start_nodes(td, num_starts=starts)
        if start_actions is None else start_actions) if forced_start else None
    td = batchify(td, starts) if starts > 1 else td
    physical = batchify(batch['distance_matrix'], starts) if starts > 1 else batch['distance_matrix']
    results = {k: [] for k in ('actions','logp','entropy','policy_valid','alive','terminal','reward','value','mask')}
    for t in range(cap):
        alive = ~td['done'].reshape(-1)
        if not alive.any():
            break
        lp, mask = step_logits(policy, td, cache, starts)
        forced = forced_start and t == 0
        if actions is not None:
            if t >= actions.size(0):
                raise ValueError('Teacher-forced route ends before completion')
            action = actions[t]
            if forced and not torch.equal(action, initial_starts):
                raise ValueError('POMO start order changed during replay')
        elif forced:
            action = initial_starts
        else:
            action = lp.argmax(-1) if greedy else lp.exp().multinomial(1).squeeze(-1)
        if not mask.gather(-1, action[:, None]).all():
            raise ValueError('Teacher-forced action violates native physical mask')
        chosen = lp.gather(-1, action[:, None]).squeeze(-1)
        valid = alive & (not forced)
        selected_lp = torch.where(valid, chosen, 0.)
        finite_lp = torch.where(mask, lp, 0.)
        entropy = -(lp.exp()*finite_lp).sum(-1)
        value = critic(td, cache, starts)
        bi = torch.arange(action.numel(), device=action.device)
        edge = physical[bi, td['current_node'], action]
        td.set('action', action)
        td = env.step(td)['next']
        terminal = td['done'].reshape(-1) & alive
        reward = -torch.where(alive, edge + torch.where(terminal, physical[bi, action, 0], 0.), 0.)
        for key, val in dict(actions=action,logp=selected_lp,entropy=entropy,
            policy_valid=valid,alive=alive,terminal=terminal,reward=reward,
            value=value,mask=mask).items():
            results[key].append(val)
    if not td['done'].all():
        raise RuntimeError('Decode cap exceeded; no silent truncated success')
    if actions is not None and len(results['actions']) != actions.size(0):
        raise ValueError('Teacher-forced sequence has unaccounted trailing actions')
    output = {k: torch.stack(v) for k,v in results.items()}
    output['neighbors'] = samples
    output['physical_cost'] = -output['reward'].sum(0)
    return output


def gae(rollout, gamma=1., lam=.95, reward_scale=100.):
    """Terminal zero bootstrap; forced transitions included, padding excluded."""
    if reward_scale <= 0:
        raise ValueError('reward_scale must be positive')
    values, rewards = rollout['value'].detach(), rollout['reward'].detach()/reward_scale
    advantage = torch.zeros_like(values)
    carry = torch.zeros_like(values[0]); next_value = torch.zeros_like(carry)
    for t in reversed(range(values.size(0))):
        cont = (~rollout['terminal'][t]).float()
        delta = rewards[t] + gamma*cont*next_value - values[t]
        carry = (delta + gamma*lam*cont*carry) * rollout['alive'][t]
        advantage[t] = carry
        next_value = values[t]
    return advantage, advantage + values


def step_ppo_loss(replay, old, *, clip=.2, value_coef=.5, entropy_coef=.01):
    advantage, returns = gae(old)
    mask = old['policy_valid']
    adv = advantage[mask]
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    logratio = replay['logp'][mask]-old['logp'][mask].detach()
    ratio = logratio.exp()
    policy = -torch.minimum(ratio*adv, ratio.clamp(1-clip,1+clip)*adv).mean()
    value = .5*(replay['value'][old['alive']]-returns[old['alive']]).square().mean()
    entropy = replay['entropy'][mask].mean()
    return policy + value_coef*value - entropy_coef*entropy
