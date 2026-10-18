"""Replay every frozen expert through the actual native environment."""
import numpy as np
import torch


@torch.no_grad()
def audit_expert_completion(exp, batch_size=256):
    checked = single = 0
    maximum_error = 0.
    for begin in range(0, len(exp.metadata['instances']), batch_size):
        indices = list(range(begin, min(begin + batch_size, len(exp.metadata['instances']))))
        records = [exp.experts.trajectory_for_instance(exp.metadata['instances'][i]['instance_id']) for i in indices]
        actions = torch.zeros(max(r.length for r in records), len(records), dtype=torch.long, device='cuda')
        for j, record in enumerate(records):
            actions[:record.length, j] = torch.tensor(record.actions, device='cuda')
            single += record.actions.count(0) == 1
        batch = exp.batch(indices)
        td = exp.env.reset(batch.clone())
        physical = batch['distance_matrix']
        cost = torch.zeros(len(records), device='cuda')
        bi = torch.arange(len(records), device='cuda')
        for action in actions:
            alive = ~td['done'].reshape(-1)
            if not td['action_mask'].gather(-1, action[:, None]).all():
                raise ValueError('Expert audit: illegal native action')
            edge = physical[bi, td['current_node'], action]
            td.set('action', action)
            td = exp.env.step(td)['next']
            terminal = td['done'].reshape(-1) & alive
            cost += torch.where(alive, edge + torch.where(terminal, physical[bi, action, 0], 0.), 0.)
        if not td['done'].all():
            raise ValueError('Expert audit: incomplete native route')
        reference = np.asarray([r.objective_distance_km for r in records])
        actual = cost.cpu().numpy()
        if not np.allclose(actual, reference, rtol=2e-5, atol=.01):
            raise ValueError('Expert audit: physical cost differs')
        maximum_error = max(maximum_error, float(np.abs(actual-reference).max()))
        checked += len(records)
    # A singleton single-tour route has no other route's padding to hide a
    # missing final depot. Exercise the real encoder/decoder replay as well.
    from e2.adapters.rrnco_step import run_actions
    index = next(i for i, meta in enumerate(exp.metadata['instances'])
                 if exp.experts.trajectory_for_instance(meta['instance_id']).actions.count(0) == 1)
    record = exp.experts.trajectory_for_instance(exp.metadata['instances'][index]['instance_id'])
    actions = torch.tensor(record.actions, device='cuda')[:, None]
    out = run_actions(exp.policy, exp.env, exp.batch([index]), exp.critic,
                      starts=1, actions=actions, start_actions=actions[0], allow_terminal_padding=True)
    if not out['terminal'][-1, 0] or not out['policy_valid'][-1, 0]:
        raise ValueError('Singleton expert must complete on explicit depot return')
    try:
        run_actions(exp.policy, exp.env, exp.batch([index]), exp.critic,
                    starts=1, actions=actions[:-1], start_actions=actions[0], allow_terminal_padding=True)
    except ValueError as exc:
        if str(exc) != 'Teacher-forced route ends before completion':raise
    else:
        raise AssertionError('Regression fixture did not reproduce the missing-depot failure')
    return dict(passed=True, instances=checked, single_tour_instances=single,
                maximum_cost_error_km=maximum_error, singleton_actor_replay=True,
                missing_depot_regression_reproduced=True)
