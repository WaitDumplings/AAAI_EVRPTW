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
