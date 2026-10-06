import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_cus100_best.py'
spec = importlib.util.spec_from_file_location('cus100_best_eval', path)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_fixed_test_protocol_preserves_checkpoint_model_and_never_keeps_eval_limits(tmp_path):
    original = {'run_name': 'validation_run', 'model': {'use_rdi_v2': True, 'agda_hidden_dim': 64},
                'env': {'reward_distance_scale': .002}, 'evaluation': {'eval_limit': 8, 'eval_num_batches': 1}}
    cfg = evaluator.build_eval_config(original, tmp_path / 'test', tmp_path / 'gurobi.csv', tmp_path / 'out')
    assert cfg['model'] == original['model'] and cfg['env'] == original['env']
    assert cfg['evaluation']['eval_n_traj'] == 50
    assert cfg['evaluation']['eval_batch_size'] == 32
    assert cfg['evaluation']['eval_max_steps'] == 201
    assert cfg['evaluation']['eval_seed'] == 17003009
    assert cfg['evaluation']['eval_save_routes'] is True
    assert 'eval_limit' not in cfg['evaluation'] and 'eval_num_batches' not in cfg['evaluation']
    assert original['evaluation']['eval_limit'] == 8


def test_checkpoint_must_match_task_scale_seed_and_selected_epoch():
    checkpoint = {'epoch': 940, 'seed': 3009, 'model_state_dict': {}, 'config': {
        'data': {'problem_type': 'vrptw', 'num_customers': 100}, 'training': {'epochs': 1000}, 'model': {'embedding_dim': 256}}}
    selection = {'epoch': 940, 'selection': 'feasibility_then_distance_v1'}
    assert evaluator.validate_checkpoint(checkpoint, 'vrptw', 3009, selection)[1] == 940
    for problem, seed, metadata in [('cvrp', 3009, selection), ('vrptw', 7, selection),
                                    ('vrptw', 3009, dict(selection, epoch=1000))]:
        with pytest.raises(ValueError):
            evaluator.validate_checkpoint(checkpoint, problem, seed, metadata)


def test_metadata_and_gurobi_ids_must_refer_to_exact_frozen_test():
    metadata = {'split': 'test', 'problem_class': 'CVRP', 'num_customers': 100, 'num_instances': 1000}
    evaluator.validate_metadata(metadata, 'cvrp')
    with pytest.raises(ValueError):
        evaluator.validate_metadata(dict(metadata, split='val'), 'cvrp')
    ids = [f'test_{i}' for i in range(1000)]
    references = [{'instance_id': key} for key in ids]
    evaluator.validate_ids(ids, references)
    for bad in [references[:-1], references[:-1] + [references[0]], references[:-1] + [{'instance_id': 'train_999'}]]:
        with pytest.raises(ValueError):
            evaluator.validate_ids(ids, bad)


def test_output_allows_preopened_console_but_cannot_overwrite_results(tmp_path):
    (tmp_path / 'console.log').write_text('caller log')
    evaluator.prepare_output(tmp_path)
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        evaluator.prepare_output(tmp_path)


def test_constructor_matches_training_aliases_without_changing_internal_graph_name():
    class Agent:
        def __init__(self, device='cpu', name='evrptw', use_rdi_v2=False, agda_hidden_dim=32):
            pass
    cfg = {'model': {'name': 'vrptw', 'use_rdi_v2': True, 'agda_hidden_dim': 64,
                     'delta_k': False, 'distance_injection': 'none'},
           'training': {'use_decomposed_critic': False}}
    kwargs = evaluator.constructor_kwargs(cfg, Agent)
    assert 'name' not in kwargs
    assert kwargs['use_rdi_v2'] and kwargs['agda_hidden_dim'] == 64
    assert not kwargs['dynamic_decision_delta_k'] and not kwargs['use_encoder_distance_bias']
    assert not kwargs['use_decomposed_critic']


def test_gap_is_mean_of_individual_percentages_and_infeasible_routes_do_not_improve_it():
    ids = ['a', 'b', 'c']
    references = [{'instance_id': key, 'feasible': 'True', 'objective_distance_km': value, 'status_name': 'TIME_LIMIT'}
                  for key, value in zip(ids, [100, 200, 100])]
    rows = [{'instance_id': key, 'feasible': feasible, 'objective_distance_km': value,
             'reference_objective_distance_km': reference, 'route_validation': {'checked': True, 'valid': feasible}}
            for key, feasible, value, reference in [('a', True, 110, 100), ('b', True, 210, 200), ('c', False, 1, 100)]]
    summary, per_instance = evaluator.summarize_rows(rows, references, ids)
    assert summary['feasible_rate'] == pytest.approx(2 / 3)
    assert summary['mean_distance_km'] == 160
    assert summary['mean_distance_gap_km'] == 10
    assert summary['mean_relative_gap_pct'] == 7.5
    assert per_instance[-1]['distance_gap_km'] is None
    rows[-1]['route_validation']['checked'] = False
    with pytest.raises(ValueError, match='not implemented/executed'):
        evaluator.summarize_rows(rows, references, ids)
