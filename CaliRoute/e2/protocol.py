"""Fail-closed protocol checks for E2; native E1 results are not PPO baselines."""
from __future__ import annotations
import math
import statistics

COMMON = ('problem', 'customers', 'training_seed', 'train_data_sha256',
          'val_data_sha256', 'test_data_sha256', 'host_source_sha256',
          'host_initial_parameters_sha256', 'normalization_contract',
          'environment_contract', 'reward_contract', 'training_budget',
          'optimizer_protocol', 'evaluation_protocol')


def require_matched_protocol(base, variant, panel):
    if panel not in ('A', 'B'):
        raise ValueError('panel must be A or B')
    fields = COMMON + (('native_trainer_sha256',) if panel == 'A' else
                       ('actor_architecture_sha256', 'critic_protocol', 'ppo_protocol'))
    for key in fields:
        if key not in base or key not in variant or base[key] is None or variant[key] is None:
            raise ValueError(f'Missing matching evidence: {key}')
        if base[key] != variant[key]:
            raise ValueError(f'Protocol mismatch: {key}')
    for cfg in (base, variant):
        if cfg.get('initialization') != 'scratch':
            raise ValueError('Main E2 comparisons require scratch initialization')
        if cfg.get('self_replay') is not False or cfg.get('incumbent_storage') != 'cost_only':
            raise ValueError('E2 excludes self-replay; own incumbent stores cost only')
    if panel == 'A':
        if base.get('modules') != [] or not variant.get('modules'):
            raise ValueError('Panel A requires unmodified host vs explicit architecture modules')
        if base.get('training_objective') != variant.get('training_objective'):
            raise ValueError('Panel A must preserve the native training objective')
        if not base.get('training_objective'):
            raise ValueError('Missing native training objective')
    else:
        if base.get('modules') != variant.get('modules') or base.get('modules') != []:
            raise ValueError('Initial Panel B must keep the actor free of RDI/AGDA additions')
        if base.get('training_objective') != 'step_ppo':
            raise ValueError('Native REINFORCE is not the Panel B PPO baseline')
        if variant.get('training_objective') not in ('step_ppo_awbc', 'step_ppo_dapg', 'step_ppo_slppo'):
            raise ValueError('Unknown Panel B training objective')
        if base.get('expert_data_sha256') is not None:
            raise ValueError('PPO baseline must not use expert data')
        if not variant.get('expert_data_sha256'):
            raise ValueError('Expert method needs a train-only expert hash')
    return True


def paired_results(base_rows, variant_rows, *, expected_count=1000, k=50, training_seed):
    """Use E1 accepted per-instance schema, never a ratio of aggregate means."""
    def index(rows):
        result = {}
        for row in rows:
            iid = row['instance_id']
            if not isinstance(iid,str) or not iid or iid in result:
                raise ValueError('Missing or duplicate instance ID')
            if row['training_seed'] != training_seed or row['requested_K'] != k or row['actual_K'] != k:
                raise ValueError('Seed/candidate budget mismatch')
            if type(row['feasible']) is not bool:
                raise ValueError('Feasibility must be a boolean')
            if row.get('cost_mismatch_candidate_count', 0) != 0:
                raise ValueError('Cost mismatch requires independent audit before aggregation')
            if row['feasible']:
                cost=row['recomputed_cost_km']
                if not row['completed'] or row['truncated'] or not isinstance(cost,(int,float)) or isinstance(cost,bool) or not math.isfinite(cost) or cost<=0:
                    raise ValueError('Invalid feasible cost or route completion')
            result[iid] = row
        if len(result) != expected_count:
            raise ValueError('Incomplete test inventory')
        return result
    base, variant = index(base_rows), index(variant_rows)
    if base.keys() != variant.keys():
        raise ValueError('Test instance IDs differ')
    common = sorted(i for i in base if base[i]['feasible'] and variant[i]['feasible'])
    def mean_cost(rows):
        values=[r['recomputed_cost_km'] for r in rows if r['feasible']]
        return statistics.mean(values) if values else None
    improvements=[100*(base[i]['recomputed_cost_km']-variant[i]['recomputed_cost_km'])/base[i]['recomputed_cost_km'] for i in common]
    return dict(test_count=expected_count, training_seed=training_seed, K=k,
                base_feasible_count=sum(r['feasible'] for r in base.values()),
                variant_feasible_count=sum(r['feasible'] for r in variant.values()),
                base_all_feasible_mean_km=mean_cost(base.values()),
                variant_all_feasible_mean_km=mean_cost(variant.values()),
                common_count=len(common), common_instance_ids=common,
                base_common_mean_km=mean_cost(base[i] for i in common),
                variant_common_mean_km=mean_cost(variant[i] for i in common),
                improvement_pct=statistics.mean(improvements) if improvements else None)
