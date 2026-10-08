"""Versioned stage-two architecture contract and safe checkpoint migration.

Weights-only initialization may add the explicitly supported zero-output modules.
Full training resume must preserve their configuration, including parameter-free
observation-mask semantics. Existing stage-one checkpoints remain compatible.
"""
from __future__ import annotations

import copy

SCHEMA = 'physical_model_integration_v1'
FLAGS = ('use_typed_static_fusion', 'use_edge_relation_encoder',
         'use_edge_value_messages', 'use_edge_state_updates', 'use_resource_decoder')
FEATURE_FLAGS = ('agda_physical_candidate_features', 'agda_smooth_distance_features')


def _canonical_model(model):
    result = {}
    for name in (*FLAGS, *FEATURE_FLAGS):
        value = model.get(name, False)
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be a boolean')
        result[name] = value
    dimension = model.get('edge_relation_dim', 16)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError('edge_relation_dim must be a positive integer')
    mode = model.get('decoder_observation_mode', 'feasible')
    if mode not in ('feasible', 'dual'):
        raise ValueError('decoder_observation_mode must be feasible or dual')
    if mode != 'feasible' and not result['use_resource_decoder']:
        raise ValueError('dual observation requires use_resource_decoder')
    if (result['use_edge_value_messages'] or result['use_edge_state_updates']) and not result['use_edge_relation_encoder']:
        raise ValueError('edge messages/updates require use_edge_relation_encoder')
    result['edge_relation_dim'] = dimension if (result['use_edge_relation_encoder'] or result['use_resource_decoder']) else None
    result['decoder_observation_mode'] = mode
    return result


def enabled(profile):
    return any(profile.get(name, False) for name in (*FLAGS, *FEATURE_FLAGS))


def signature(cfg):
    model = cfg.get('model', {}) or {}
    result = {'schema': SCHEMA, **_canonical_model(model)}
    if any(result[name] for name in FLAGS):
        from .input_normalization import signature as input_signature
        inputs = input_signature(cfg)
        if not inputs['use_physical_input_context']:
            raise ValueError('Stage-two integration requires the physical input context schema')
    return result


def _check_agent(agent, profile):
    actual = getattr(getattr(agent, 'backbone', None), 'model_integration_settings', {})
    if _canonical_model(actual) != {key: value for key, value in profile.items() if key != 'schema'}:
        raise ValueError('Agent architecture does not match model integration profile')


def configure(agent, cfg):
    profile = signature(cfg)
    _check_agent(agent, profile)
    previous = getattr(agent, '_model_integration_signature', None)
    if previous is not None and previous != profile:
        raise ValueError('Model integration is already configured with a different profile')
    agent._model_integration_signature = copy.deepcopy(profile)
    return profile


def checkpoint_profile(checkpoint, *, require_metadata=True):
    computed = signature(checkpoint.get('config', {}) or {})
    saved = checkpoint.get('model_integration_signature')
    if saved is None:
        if require_metadata and enabled(computed):
            raise ValueError('Stage-two checkpoint is missing its model integration signature')
        return computed
    if isinstance(saved, dict):
        saved = copy.deepcopy(saved)
        for name in FEATURE_FLAGS:
            saved.setdefault(name, False)  # pre-feature v1 checkpoints
    if not isinstance(saved, dict) or saved != computed:
        raise ValueError('Checkpoint model integration signature does not match its saved config')
    return copy.deepcopy(saved)


def load_checkpoint_profile(agent, checkpoint, *, resume, checkpoint_path=None):
    source = checkpoint_profile(checkpoint, require_metadata=resume)
    target = getattr(agent, '_model_integration_signature', None)
    if target is None:
        actual = _canonical_model(getattr(getattr(agent, 'backbone', None), 'model_integration_settings', {}))
        if enabled(source) or any(actual[name] for name in (*FLAGS, *FEATURE_FLAGS)):
            raise ValueError('Configure the target model integration before loading stage-two weights')
        target = source
    _check_agent(agent, target)
    if resume and source != target:
        raise ValueError('Model integration changed on resume; use weights-only initialization')
    if resume:
        previous = checkpoint.get('model_integration_initialization')
        if previous is not None:
            agent._model_integration_initialization = copy.deepcopy(previous)
        return {'load': 'full_resume', 'source': source, 'target': copy.deepcopy(target), 'migrated': False}
    result = {'load': 'weights_only', 'source': source, 'target': copy.deepcopy(target),
              'migrated': source != target,
              'source_profile_recorded': checkpoint.get('model_integration_signature') is not None,
              'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None}
    agent._model_integration_initialization = copy.deepcopy(result)
    return result


def checkpoint_metadata(agent, cfg):
    profile = signature(cfg)
    _check_agent(agent, profile)
    configured = getattr(agent, '_model_integration_signature', None)
    if configured is not None and configured != profile:
        raise ValueError('Model integration changed after initialization')
    result = {'model_integration_signature': profile}
    initialization = getattr(agent, '_model_integration_initialization', None)
    if initialization is not None:
        result['model_integration_initialization'] = copy.deepcopy(initialization)
    return result
