"""Recipe metadata reports actual branches, not historical flag labels."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from caliroute.recipe_components import SCHEMA, describe_components


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / 'docs/experiments/graph_rdi100_20261008_provenance.json').read_text())


def recipe(task='vrptw', encoder='graph'):
    batch = 40 if task == 'vrptw' else 32
    name = f'{encoder.upper()}_{task.upper()}100_S3011_B{batch}_E1500_20261008_r2'
    return deepcopy(REFERENCE['runs'][name]['effective_config'])


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
@pytest.mark.parametrize('encoder', ['graph', 'current'])
def test_audited_recipe_describes_actual_graph_decoder_and_training_branches(task, encoder):
    out = describe_components(recipe(task, encoder))
    assert out['schema'] == SCHEMA
    assert out['static_graph']['encoder'] == ('joint_node_edge' if encoder == 'graph' else 'current_graph_attention')
    assert out['static_graph']['effective_edge_dim'] == (32 if encoder == 'graph' else 16)
    assert out['static_graph']['decoder_reads_current_node_edge_row']
    assert out['static_graph']['edge_state_updates'] == (encoder == 'graph')
    assert out['static_graph']['edge_value_messages'] == (encoder == 'graph')
    d = out['dynamic_decision']
    assert d['dde'] and d['agda_v2_candidate_gate']
    assert d['resource_candidate_fusion'] and d['dual_resource_readout']
    assert d['transitions_shared_between_dde_and_resource_adapter']
    assert d['dde_residuals'] == dict(delta_k=False, delta_v=False, delta_action_key=True, action_bias=True)
    assert out['training']['ppo_passes_cap'] == 5
    assert out['training']['ppo_pass_policy'] == 'fixed_passes'
    assert out['training']['optimizer_attempts_per_epoch_cap'] == 20
    assert out['training']['global_instances_per_rollout'] == (40 if task == 'vrptw' else 32)
    assert out['slppo']['online_route_advantage'] == 'leave_one_out_feasible_physical_cost_shared_RMS'
    assert out['slppo']['expert']['loss_configured']
    assert out['slppo']['replay']['loss_configured']
    assert out['slppo']['search']['enabled_with_archive']


def test_description_is_pure_json_metadata_without_cfg_or_output_aliases():
    cfg = recipe()
    before = deepcopy(cfg)
    out = describe_components(cfg)
    assert cfg == before and json.loads(json.dumps(out)) == out
    out['static_graph']['active_input_channels'].clear()
    out['dynamic_decision']['dde_residuals']['delta_k'] = 'modified'
    out['evidence']['controlled_recipe_comparisons'].clear()
    assert cfg == before
    assert describe_components(cfg)['static_graph']['active_input_channels']
    assert describe_components(cfg)['evidence']['controlled_recipe_comparisons']


def test_rdi_and_agda_local_switches_do_not_claim_whole_module_removal():
    cfg = recipe()
    cfg['model'].update(use_rdi_v2=False, use_agda_v2=False)
    out = describe_components(cfg)
    assert not out['static_graph']['rdi_v2_residual_bias']
    assert out['static_graph']['linear_distance_prior'] and out['static_graph']['directed_edge_relations']
    assert out['routing_inputs']['context_consumed']
    assert not out['dynamic_decision']['agda_v2_candidate_gate']
    assert out['dynamic_decision']['dde'] and out['dynamic_decision']['resource_candidate_fusion']
    assert any('use_rdi_v2=False disables only' in text for text in out['notes'])
    assert any('use_agda_v2=False disables only' in text for text in out['notes'])
    assert out['evidence']['controlled_recipe_comparisons'] == ['graph_vs_current_encoder_bundle']
    assert 'P0_P1_added_to_joint_graph' in out['evidence']['not_independently_established']


def test_disabled_dde_suppresses_children_but_keeps_resource_and_edge_adapter():
    cfg = recipe()
    cfg['model']['use_dynamic_decision_encoder'] = False
    out = describe_components(cfg)['dynamic_decision']
    assert not out['dde'] and not out['agda_v2_candidate_gate']
    assert not out['physical_candidate_features'] and not out['smooth_distance_features']
    assert not any(out['dde_residuals'].values())
    assert not out['transitions_shared_between_dde_and_resource_adapter']
    assert out['resource_adapter_instantiated'] and out['resource_candidate_fusion']
    assert out['learned_edge_readout']


def test_edge_only_resource_adapter_is_still_instantiated():
    cfg = recipe()
    cfg['model'].update(use_resource_decoder=False, decoder_observation_mode='feasible')
    out = describe_components(cfg)['dynamic_decision']
    assert out['resource_adapter_instantiated'] and out['learned_edge_readout']
    assert not out['resource_candidate_fusion'] and not out['dual_resource_readout']
    assert not out['transitions_shared_between_dde_and_resource_adapter']


def test_joint_edge_width_overrides_inactive_legacy_width():
    cfg = recipe()
    cfg['model']['edge_relation_dim'] = 7
    assert describe_components(cfg)['static_graph']['effective_edge_dim'] == 32
    cfg['model'].update(use_joint_graph_encoder=False, use_edge_relation_encoder=True)
    assert describe_components(cfg)['static_graph']['effective_edge_dim'] == 7
    cfg['model']['use_edge_relation_encoder'] = False
    assert describe_components(cfg)['static_graph']['effective_edge_dim'] is None
    assert not describe_components(cfg)['dynamic_decision']['learned_edge_readout']


def test_physical_online_advantage_ignores_legacy_group_reference_controls_only():
    cfg = recipe()
    before = describe_components(cfg)
    cfg['advantage'].update(use_group_advantage=False, use_reference_advantage=False,
        group_adv_std_floor=1e9, group_adv_coef=0., sl_include_reference_in_group_stats=False)
    after = describe_components(cfg)
    assert after == before
    assert after['slppo']['legacy_group_reference_controls_bypassed_for_online_advantage']
    assert after['slppo']['expert_and_replay_in_online_group'] is False
    assert 'excluded_from_actor_RMS' in after['slppo']['expert']['normalization']
    cfg['training']['reward_norm_mode'] = 'legacy'
    old = describe_components(cfg)
    assert old['slppo']['online_route_advantage'] == 'legacy_solution_level_advantage_tensors'
    assert not old['slppo']['legacy_group_reference_controls_bypassed_for_online_advantage']


def test_ppo_does_not_report_sl_training_despite_stale_sl_flags():
    cfg = recipe()
    cfg['offline']['method'] = 'ppo'
    out = describe_components(cfg)
    assert out['training']['ppo_actor_advantage'] == 'GAE_shared_actor_RMS'
    sl = out['slppo']
    assert not sl['enabled'] and not sl['online_loss_configured']
    assert sl['online_route_advantage'] == 'disabled'
    assert not sl['expert']['candidate_branch_requested']
    assert not sl['replay']['archive_enabled'] and not sl['search']['requested']


@pytest.mark.parametrize('primary,alias,expected', [(True, False, True), (False, True, False)])
def test_branch_search_primary_flag_wins_over_alias(primary, alias, expected):
    cfg = recipe()
    cfg['offline'].update(branch_exploration_enabled=primary, exploration_enabled=alias)
    search = describe_components(cfg)['slppo']['search']
    assert search['enabled_with_archive'] == expected and search['requested'] == expected


def test_search_alias_fallback_and_required_archive_are_explicit():
    cfg = recipe()
    del cfg['offline']['branch_exploration_enabled']
    cfg['offline'].update(exploration_enabled=True, policy_replay_enabled=False)
    search = describe_components(cfg)['slppo']['search']
    assert search['requested'] and search['missing_required_archive']
    assert not search['enabled_with_archive']


def test_archive_and_search_are_separate_from_zero_weight_training():
    cfg = recipe()
    cfg['offline']['policy_replay_weight'] = 0.
    sl = describe_components(cfg)['slppo']
    assert sl['replay']['archive_enabled'] and not sl['replay']['loss_configured']
    assert sl['search']['enabled_with_archive']
    cfg['offline']['sl_coef'] = 0.
    sl = describe_components(cfg)['slppo']
    assert sl['enabled'] and not sl['online_loss_configured'] and not sl['expert']['loss_configured']
    assert sl['search']['enabled_with_archive']


def test_expert_alias_and_enable_precedence_match_trainer():
    cfg = recipe()
    cfg['offline']['use_expert_solution_level'] = False
    assert describe_components(cfg)['slppo']['expert']['candidate_branch_requested']
    cfg['advantage'].update(sl_candidate_use_expert_candidate=False, sl_use_expert_candidate=True)
    assert not describe_components(cfg)['slppo']['expert']['candidate_branch_requested']
    del cfg['advantage']['sl_candidate_use_expert_candidate']
    cfg['advantage']['sl_use_expert_candidate'] = False
    assert not describe_components(cfg)['slppo']['expert']['candidate_branch_requested']


def test_physical_online_advantage_does_not_disable_expert_gate_options():
    cfg = recipe()
    cfg['advantage']['sl_candidate_use_current_incumbent_gate'] = False
    assert not describe_components(cfg)['slppo']['expert']['current_incumbent_gate']
    assert describe_components(cfg)['slppo']['expert']['memory_incumbent_gate']
    cfg['advantage']['sl_candidate_use_current_incumbent'] = True
    assert describe_components(cfg)['slppo']['expert']['current_incumbent_gate']


def test_monitor_kl_is_not_stopping_target_and_optimizer_attempts_are_not_passes():
    cfg = recipe()
    cfg['training'].update(monitor_target_kl=.001, ppo_update_epochs=5,
        num_envs_per_gpu=3, num_minibatches=4, gradient_accumulation_steps=2)
    out = describe_components(cfg)['training']
    assert out['ppo_pass_policy'] == 'fixed_passes'
    assert out['minibatches_per_pass'] == 3 and out['optimizer_attempts_per_epoch_cap'] == 10
    assert not out['monitor_target_kl_controls_updates']
    cfg['training']['target_kl'] = .02
    out = describe_components(cfg)['training']
    assert out['ppo_pass_policy'] == 'post_pass_fresh_KL_early_stop'
    assert out['ppo_passes_cap'] == 5 and out['optimizer_attempts_per_epoch_cap'] == 10


def test_missing_topology_is_unknown_and_distance_flag_overrides_mode():
    cfg = recipe()
    cfg.pop('experiment_protocol')
    cfg['model'].update(distance_injection='none', use_encoder_distance_bias=True)
    out = describe_components(cfg)
    assert out['training']['world_size_from_protocol'] is None
    assert out['training']['global_instances_per_rollout'] is None
    assert out['static_graph']['linear_distance_prior']
    del cfg['model']['use_encoder_distance_bias']
    assert not describe_components(cfg)['static_graph']['linear_distance_prior']


def test_runtime_resource_activity_is_not_guessed_from_task_label():
    cfg = recipe()
    cfg['data']['problem_type'] = 'cvrp'
    out = describe_components(cfg)['routing_inputs']
    assert out['problem_type'] == 'cvrp'
    assert out['resource_activity_source'].startswith('per_instance_graph_input_context_flags')
    assert out['context_widths'] == {'node': 12, 'graph': 10}
    assert out['network_requires_road_matrices'] == ['edge_distance', 'edge_time', 'edge_energy']
