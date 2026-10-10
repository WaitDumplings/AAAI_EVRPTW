"""The reduced default is a new candidate, never relabelled reference results."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from caliroute import recipes


EXPECTED_CHANGES = {
    'model.use_rdi_v2': (True, False),
    'model.use_typed_static_fusion': (True, False),
    'model.use_agda_v2': (True, False),
    'model.use_resource_decoder': (True, False),
    'model.decoder_observation_mode': ('dual', 'feasible'),
    'offline.branch_exploration_enabled': (True, False),
}


def build(tmp_path, **kwargs):
    values = dict(problem='vrptw', data_root=tmp_path/'AAAI_Dataset',
        output_dir=tmp_path/'output', run_name='CORE_CONFIG_TEST')
    values.update(kwargs)
    return recipes.build_recipe_config(**values)


def flatten(mapping, prefix=''):
    result = {}
    for key, value in mapping.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


@pytest.mark.parametrize('problem', ['vrptw', 'evrptw'])
@pytest.mark.parametrize('encoder', ['graph', 'current'])
@pytest.mark.parametrize('hardware,world_size', [('rtx48_single', 1), ('2080ti_dual', 2)])
def test_only_declared_core_branches_differ_from_reference(tmp_path, problem, encoder, hardware, world_size):
    args = dict(problem=problem, encoder=encoder, hardware=hardware, world_size=world_size)
    reference = build(tmp_path, preset='reference', **args)
    core = build(tmp_path, preset='core', **args)
    a, b = flatten(reference), flatten(core)
    changes = {key:(a.get(key),b.get(key)) for key in a.keys()|b.keys()
               if not key.startswith('experiment_protocol.') and a.get(key)!=b.get(key)}
    assert changes == EXPECTED_CHANGES
    recorded = {row['parameter']:(row['reference'],row['used'])
                for row in core['experiment_protocol']['config_changes_from_reference']}
    assert recorded == EXPECTED_CHANGES
    # Seven explicit overrides include exploration_enabled=False, which was
    # already false in the parent; both names must remain false to avoid fallback.
    assert not core['offline']['exploration_enabled']
    assert not core['offline']['branch_exploration_enabled']
    assert core['training'] == reference['training']
    assert core['advantage'] == reference['advantage']
    for key in ('data','env','critic','pbrs','evaluation'):
        assert core[key] == reference[key]


def test_default_is_untrained_core_and_reference_remains_explicit(tmp_path):
    core = build(tmp_path)
    protocol = core['experiment_protocol']
    assert protocol['preset'] == 'core'
    assert protocol['recipe'] == 'aaai_graph_core_v1'
    assert protocol['recipe_status'] == 'untrained_candidate'
    assert protocol['configured_recipe_quality_evidence'] == 'none/untrained'
    assert 'reference_run' not in protocol
    assert protocol['derived_from_reference_run'].startswith('GRAPH_VRPTW100_')
    assert protocol['implementation'] == 'slppo_core'
    assert protocol['architecture']['training_bundle'] == 'core'
    assert 'reference performance does not establish' in protocol['comparison_scope']
    reference = build(tmp_path, preset='reference')['experiment_protocol']
    assert reference['recipe'] == 'aaai_graph_v1'
    assert reference['recipe_status'] == 'recorded_reference_configuration'
    assert reference['reference_run'] == protocol['derived_from_reference_run']
    assert reference['config_changes_from_reference'] == []
    assert 'derived_from_reference_run' not in reference
    assert len(protocol['recipe_sources']) == 3
    assert len(reference['recipe_sources']) == 2
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('world,hardware', [(1,'rtx48_single'),(2,'2080ti_dual')])
def test_effective_search_is_zero_while_archive_expert_and_numeric_protocol_stay(tmp_path, world, hardware):
    core = build(tmp_path, world_size=world, hardware=hardware)
    offline, train, protocol = (core[key] for key in ('offline','training','experiment_protocol'))
    assert offline['policy_replay_enabled']
    assert offline['policy_replay_selection'] == 'structural'
    assert offline['policy_replay_capacity'] == 3
    assert offline['policy_replay_exploration_capacity'] == 4
    assert offline['policy_replay_weight'] == .1
    assert offline['policy_replay_warmup_epochs'] == 25
    assert offline['policy_replay_ramp_epochs'] == 75
    assert offline['policy_replay_max_new_routes']*world == 32
    assert core['advantage']['use_expert_solution_level']
    assert core['advantage']['sl_use_expert_candidate']
    assert core['advantage']['sl_expert_candidate_weight'] == .6
    assert offline['sl_expert_candidate_weight'] == .6
    assert offline['method'] == 'sl_ppo' and offline['sl_coef'] == .5
    assert train['ppo_update_epochs'] == 5 and train['num_minibatches'] == 4
    assert train['learning_rate'] == 1e-4 and train['gamma'] == 1.
    assert protocol['ppo_warmup_epochs'] == 0
    assert not protocol['extra_search_enabled']
    assert protocol['global_exploration_instances'] == 0
    search = protocol['search_budget']
    assert search['enabled'] is False
    assert search['interval'] == offline['exploration_interval'] == 5
    assert search['max_global_trajectories'] == search['max_instances_per_rank'] == 0
    assert search['configured_instances_per_rank'] == offline['exploration_instances'] == 8//world
    assert search['configured_global_instances'] == 8
    assert protocol['global_policy_replay_max_new_routes'] == 32
    assert protocol['global_policy_replay_candidate_budget'] == 10
    assert 'Independent branch-search rollouts are disabled' in protocol['compute_caveat']
    reference = build(tmp_path, preset='reference', world_size=world, hardware=hardware)['experiment_protocol']
    assert reference['global_exploration_instances'] == 8
    assert reference['search_budget']['max_global_trajectories'] == 64


def test_resolved_component_report_describes_edge_only_decoder_and_live_sl_path(tmp_path):
    cfg = build(tmp_path)
    report = cfg['experiment_protocol']['resolved_components']
    assert report['static_graph']['encoder'] == 'joint_node_edge'
    assert report['static_graph']['effective_edge_dim'] == 32
    assert report['static_graph']['node_incoming_outgoing_edge_fusion']
    assert report['static_graph']['edge_value_messages']
    assert report['static_graph']['edge_state_updates']
    assert not report['static_graph']['rdi_v2_residual_bias']
    assert not report['static_graph']['typed_static_fusion']
    assert report['dynamic_decision']['dde']
    assert report['dynamic_decision']['physical_candidate_features']
    assert not report['dynamic_decision']['agda_v2_candidate_gate']
    assert not report['dynamic_decision']['resource_candidate_fusion']
    assert not report['dynamic_decision']['dual_resource_readout']
    assert report['dynamic_decision']['resource_adapter_instantiated']
    assert report['dynamic_decision']['learned_edge_readout']
    assert report['slppo']['expert']['loss_configured']
    assert report['slppo']['online_route_advantage'] == 'leave_one_out_feasible_physical_cost_shared_RMS'
    assert report['slppo']['legacy_group_reference_controls_bypassed_for_online_advantage']
    assert not report['slppo']['search']['requested']
    assert report['evidence']['controlled_recipe_comparisons'] == []
    assert report['evidence']['parent_reference_comparisons'] == ['graph_vs_current_encoder_bundle']
    assert report['evidence']['configured_recipe_quality_evidence'] == 'none/untrained'


def test_source_engine_pin_and_physical_contract_are_shared(tmp_path):
    cfg = build(tmp_path)
    protocol = cfg['experiment_protocol']
    assert protocol['reference_training_commit'] == 'd4364926999d8b55be6ec3aa374c6cb6fc238cea'
    assert cfg['model']['use_physical_input_context']
    assert cfg['model']['use_encoder_distance_bias']
    assert cfg['data']['strict_road_metric']
    assert cfg['env']['prefer_explicit_edge_matrices']
    assert cfg['env']['reward_distance_scale_km'] == cfg['env']['observation_distance_scale_km'] == 43.638668060302734
    assert cfg['model']['decoder_observation_mode'] == 'feasible'
    assert cfg['model']['dynamic_decision_delta_action_key']
    assert cfg['model']['dynamic_decision_action_bias']
    assert cfg['model']['use_static_rollout_cache']
    assert cfg['training']['share_ppo_sl_forward']


def test_explicit_dual_memory_overrides_do_not_restore_disabled_branches(tmp_path):
    cfg = build(tmp_path, world_size=2, hardware='2080ti_dual', global_batch=48,
                ppo_chunk_size=24, expert_chunk_size=32, eval_batch_size=8, seed=0, epochs=300)
    assert cfg['training']['num_envs_per_gpu'] == 24
    assert cfg['training']['post_init_seed'] == 0
    assert cfg['evaluation']['eval_seed'] == 17000000
    assert cfg['training']['ppo_step_chunk_size'] == 24
    assert cfg['offline']['sl_expert_logprob_chunk_size'] == 32
    assert cfg['evaluation']['eval_batch_size'] == 8
    assert cfg['experiment_protocol']['eval_batch_changes_sampling_stream']
    assert not cfg['model']['use_rdi_v2']
    assert not cfg['offline']['branch_exploration_enabled']
    assert cfg['experiment_protocol']['global_exploration_instances'] == 0
    assert cfg['experiment_protocol']['global_policy_replay_candidate_budget'] == 12


def test_invalid_preset_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='preset must be core or reference'):
        build(tmp_path,preset='original')


def test_overlay_contains_only_declared_branches_and_does_not_mutate_reference(tmp_path):
    source_before = recipes.RECIPE_PATH.read_bytes()
    overlay = yaml.safe_load(recipes.CORE_RECIPE_PATH.read_text())
    assert overlay['extends'] == 'aaai_graph_v1'
    assert set(overlay['overrides']) == {'model','offline'}
    assert len(flatten(overlay['overrides'])) == 7
    first = build(tmp_path)
    first['model']['use_joint_graph_encoder'] = False
    first['offline']['exploration_prefix_fractions'].append(.99)
    second = build(tmp_path)
    assert second['model']['use_joint_graph_encoder']
    assert second['offline']['exploration_prefix_fractions'] == [0., .1, .25, .5]
    assert recipes.RECIPE_PATH.read_bytes() == source_before
    with pytest.raises(ValueError,match='unknown configuration field'):
        recipes._apply_overlay(deepcopy(second), {'model': {'undeclared': True}})
