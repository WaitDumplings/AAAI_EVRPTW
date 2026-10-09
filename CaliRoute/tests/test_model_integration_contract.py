"""Checkpoint architecture changes must be explicit and cannot alter full resumes."""
import copy
from unittest.mock import patch

import pytest
import torch

from offline2online import input_normalization as inputs
from offline2online import model_integration as integration
from offline2online import trainer
from test_model_design_optimizations import agent


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cfg(**changes):
    return {'env': {'observation_coordinate_mode': 'depot_fixed',
                    'observation_input_context': True, 'observation_distance_scale_km': 43.638668060302734},
            'model': {'use_physical_input_context': True, **changes}}


def model(config):
    result = agent(**config['model'])
    inputs.configure(result, config)
    integration.configure(result, config)
    return result


@pytest.mark.parametrize('changes,match', [
    ({'use_resource_decoder': 'false'}, 'boolean'),
    ({'edge_relation_dim': 0}, 'positive integer'),
    ({'edge_relation_dim': True}, 'positive integer'),
    ({'decoder_observation_mode': 'unknown'}, 'feasible or dual'),
    ({'decoder_observation_mode': 'dual'}, 'requires'),
    ({'use_edge_value_messages': True}, 'require'),
    ({'use_edge_state_updates': True}, 'require'),
    ({'use_joint_graph_encoder': 'false'}, 'boolean'),
    ({'joint_graph_edge_dim': 0}, 'positive integer'),
    ({'joint_graph_edge_dim': True}, 'positive integer'),
    ({'joint_graph_edge_dim': 16.5}, 'positive integer'),
    ({'joint_graph_dropout': .1}, 'deterministic cached PPO'),
    ({'joint_graph_dropout': float('nan')}, 'deterministic cached PPO'),
    ({'joint_graph_dropout': True}, 'deterministic cached PPO'),
    ({'use_joint_graph_encoder': True, 'use_edge_relation_encoder': True}, 'replaces'),
    ({'use_joint_graph_encoder': True, 'use_edge_value_messages': True}, 'replaces'),
    ({'use_joint_graph_encoder': True, 'use_edge_state_updates': True}, 'replaces'),
])
def test_invalid_architecture_fails_early(changes, match):
    with pytest.raises(ValueError, match=match):
        integration.signature(cfg(**changes))
    with patch.object(trainer.DistributedContext, 'initialize', side_effect=AssertionError('device initialized')):
        with pytest.raises(ValueError, match=match):
            trainer.train_from_config(cfg(**changes), seed=3009, device='cpu')


def test_stage_two_requires_physical_input_schema():
    with pytest.raises(ValueError, match='physical input context'):
        integration.signature({'model': {'use_typed_static_fusion': True}})


def test_disabled_contract_preserves_old_checkpoint_loading(tmp_path):
    config = cfg()
    original = model(config)
    path = tmp_path/'legacy.pt'
    torch.save({'config': config, 'model_state_dict': original.state_dict(),
                'input_normalization_signature': inputs.signature(config)}, path)
    target = model(config)
    info = trainer._load_training_checkpoint(target, torch.optim.Adam(target.parameters()), path, 'cpu')
    assert info['model_integration']['load'] == 'full_resume'
    assert not integration.enabled(info['model_integration']['target'])


def test_full_resume_roundtrip_and_weights_only_migration(tmp_path):
    old_config = cfg()
    original = model(old_config)
    path = tmp_path/'old.pt'
    trainer.save_checkpoint(path, original, torch.optim.Adam(original.parameters()), old_config, 1, 3009)
    new_config = cfg(use_typed_static_fusion=True, use_edge_relation_encoder=True,
                     use_resource_decoder=True, decoder_observation_mode='dual')
    upgraded = model(new_config)
    info = trainer._load_agent_checkpoint(upgraded, path, 'cpu', strict=False)
    assert info['model_integration']['migrated']
    assert info['missing_keys'] and not info['unexpected_keys']
    new_path = tmp_path/'new.pt'
    trainer.save_checkpoint(new_path, upgraded, torch.optim.Adam(upgraded.parameters(), lr=.007), new_config, 2, 3009)
    checkpoint = torch.load(new_path, weights_only=False)
    assert checkpoint['model_integration_signature'] == integration.signature(new_config)
    assert checkpoint['model_integration_initialization'] == info['model_integration']
    target = model(new_config)
    opt = torch.optim.Adam(target.parameters(), lr=.02)
    trainer._load_training_checkpoint(target, opt, new_path, 'cpu')
    assert opt.param_groups[0]['lr'] == .007
    for name, value in upgraded.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize('change', [
    {'decoder_observation_mode': 'feasible'}, {'use_edge_value_messages': True},
    {'use_edge_state_updates': True}, {'edge_relation_dim': 8},
])
def test_resume_rejects_semantic_or_parameter_changes_before_mutation(tmp_path, change):
    source = cfg(use_edge_relation_encoder=True, use_resource_decoder=True, decoder_observation_mode='dual')
    original = model(source)
    path = tmp_path/'source.pt'
    trainer.save_checkpoint(path, original, torch.optim.Adam(original.parameters(), lr=.007), source, 2, 3009)
    target_cfg = copy.deepcopy(source)
    target_cfg['model'].update(change)
    target = model(target_cfg)
    before = copy.deepcopy(target.state_dict())
    opt = torch.optim.Adam(target.parameters(), lr=.02)
    with pytest.raises(ValueError, match='changed on resume'):
        trainer._load_training_checkpoint(target, opt, path, 'cpu', strict=False)
    assert opt.param_groups[0]['lr'] == .02
    for name, value in before.items():
        torch.testing.assert_close(value, target.state_dict()[name], atol=0, rtol=0)


def test_stage_two_metadata_cannot_be_removed_or_tampered(tmp_path):
    config = cfg(use_resource_decoder=True, decoder_observation_mode='dual')
    original = model(config)
    path = tmp_path/'source.pt'
    trainer.save_checkpoint(path, original, torch.optim.Adam(original.parameters()), config, 1, 3009)
    checkpoint = torch.load(path, weights_only=False)
    broken = copy.deepcopy(checkpoint)
    broken.pop('model_integration_signature')
    with pytest.raises(ValueError, match='missing'):
        integration.checkpoint_profile(broken)
    broken = copy.deepcopy(checkpoint)
    broken['config']['model']['decoder_observation_mode'] = 'feasible'
    with pytest.raises(ValueError, match='does not match'):
        integration.checkpoint_profile(broken)


def test_missing_backbone_is_not_allowed_by_new_layer_adapter_whitelist():
    old = model(cfg())
    new = model(cfg(use_edge_relation_encoder=True, use_resource_decoder=True))
    weights = old.state_dict()
    weights.pop('backbone.encoder.layers.0.norm1.weight')
    with pytest.raises(RuntimeError, match='norm1.weight'):
        trainer._load_initial_model_state(new, weights, strict=False)


def test_changed_architecture_after_initialization_cannot_be_saved(tmp_path):
    config = cfg(use_resource_decoder=True, decoder_observation_mode='dual')
    net = model(config)
    changed = copy.deepcopy(config)
    changed['model']['decoder_observation_mode'] = 'feasible'
    with pytest.raises(ValueError, match='profile|changed'):
        trainer.save_checkpoint(tmp_path/'bad.pt', net, torch.optim.Adam(net.parameters()), changed, 1, 3009)


def graph_cfg(**changes):
    options = dict(use_joint_graph_encoder=True, joint_graph_edge_dim=32,
                   joint_graph_dropout=0.0, use_resource_decoder=True,
                   decoder_observation_mode='dual')
    options.update(changes)
    return cfg(**options)


def test_joint_graph_requires_physical_inputs_and_records_effective_edge_width():
    with pytest.raises(ValueError, match='physical input context'):
        integration.signature({'model': {'use_joint_graph_encoder': True}})
    profile = integration.signature(graph_cfg())
    assert profile['use_joint_graph_encoder'] is True
    assert profile['joint_graph_edge_dim'] == 32
    assert profile['joint_graph_dropout'] == 0.0
    assert integration.enabled(profile)


def test_graph_disabled_retains_exact_pre_graph_v1_signature():
    original = integration.signature(cfg(use_resource_decoder=True))
    assert set(original) == {
        'schema', 'use_typed_static_fusion', 'use_edge_relation_encoder',
        'use_edge_value_messages', 'use_edge_state_updates', 'use_resource_decoder',
        'agda_physical_candidate_features', 'agda_smooth_distance_features',
        'edge_relation_dim', 'decoder_observation_mode',
    }
    explicit_disabled = cfg(use_resource_decoder=True, use_joint_graph_encoder=False,
                            joint_graph_edge_dim=64, joint_graph_dropout=0.0)
    assert integration.signature(explicit_disabled) == original
    # Pre-feature stage-two checkpoints may omit the two AGDA booleans.
    saved = {key: value for key, value in original.items() if key not in integration.FEATURE_FLAGS}
    assert integration.checkpoint_profile({'config': explicit_disabled,
                                          'model_integration_signature': saved}) == original


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('source_graph,target_graph', [(False, True), (True, False)])
def test_graph_architecture_migration_is_rejected_before_model_or_optimizer_mutation(
        tmp_path, resume, source_graph, target_graph):
    old_config = cfg(use_resource_decoder=True, decoder_observation_mode='dual')
    source_config = graph_cfg() if source_graph else old_config
    target_config = graph_cfg() if target_graph else old_config
    source = model(source_config)
    checkpoint = tmp_path/'architecture.pt'
    trainer.save_checkpoint(checkpoint, source, torch.optim.Adam(source.parameters(), lr=.007),
                            source_config, 1, 3011)
    target = model(target_config)
    before = copy.deepcopy(target.state_dict())
    optimizer = torch.optim.Adam(target.parameters(), lr=.02)
    with pytest.raises(ValueError, match='from scratch'):
        if resume:
            trainer._load_training_checkpoint(target, optimizer, checkpoint, 'cpu', strict=False)
        else:
            trainer._load_agent_checkpoint(target, checkpoint, 'cpu', strict=False)
    assert optimizer.param_groups[0]['lr'] == .02
    assert not hasattr(target, '_model_integration_initialization')
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('change', [
    {'joint_graph_edge_dim': 16}, {'decoder_observation_mode': 'feasible'},
])
def test_graph_signature_changes_are_rejected_even_for_weights_only(tmp_path, resume, change):
    source_config = graph_cfg()
    source = model(source_config)
    checkpoint = tmp_path/'graph.pt'
    trainer.save_checkpoint(checkpoint, source, torch.optim.Adam(source.parameters()),
                            source_config, 1, 3011)
    target_config = graph_cfg(**change)
    target = model(target_config)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='identical'):
        if resume:
            trainer._load_training_checkpoint(target, torch.optim.Adam(target.parameters()),
                                              checkpoint, 'cpu', strict=False)
        else:
            trainer._load_agent_checkpoint(target, checkpoint, 'cpu', strict=False)
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)


def test_matching_graph_checkpoints_roundtrip_weights_and_full_resume(tmp_path):
    config = graph_cfg()
    source = model(config)
    checkpoint = tmp_path/'graph.pt'
    trainer.save_checkpoint(checkpoint, source, torch.optim.Adam(source.parameters(), lr=.007),
                            config, 2, 3011)
    target = model(config)
    info = trainer._load_agent_checkpoint(target, checkpoint, 'cpu', strict=True)
    assert not info['model_integration']['migrated']
    assert not info['missing_keys'] and not info['unexpected_keys']
    optimizer = torch.optim.Adam(target.parameters(), lr=.02)
    info = trainer._load_training_checkpoint(target, optimizer, checkpoint, 'cpu')
    assert optimizer.param_groups[0]['lr'] == .007
    assert info['model_integration']['load'] == 'full_resume'
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)


def test_joint_graph_metadata_required_and_dimension_tampering_rejected():
    config = graph_cfg()
    checkpoint = {'config': config, 'model_integration_signature': integration.signature(config)}
    assert integration.checkpoint_profile(checkpoint) == integration.signature(config)
    missing = {'config': config}
    for require_metadata in (False, True):
        with pytest.raises(ValueError, match='missing'):
            integration.checkpoint_profile(missing, require_metadata=require_metadata)
    checkpoint['config'] = graph_cfg(joint_graph_edge_dim=16)
    with pytest.raises(ValueError, match='does not match'):
        integration.checkpoint_profile(checkpoint)
