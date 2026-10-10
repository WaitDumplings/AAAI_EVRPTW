"""Versioned stage-two architecture contract and safe checkpoint migration.

Weights-only initialization may add the explicitly supported zero-output modules.
Full training resume must preserve their configuration, including parameter-free
observation-mask semantics. Existing stage-one checkpoints remain compatible.
The joint graph encoder is a separately initialized architecture, not a residual
migration adapter: its checkpoints can only load into the same graph profile.
"""
from __future__ import annotations

import copy
import math

SCHEMA = 'physical_model_integration_v1'
FLAGS = ('use_typed_static_fusion', 'use_edge_relation_encoder',
         'use_edge_value_messages', 'use_edge_state_updates', 'use_resource_decoder')
FEATURE_FLAGS = ('agda_physical_candidate_features', 'agda_smooth_distance_features')
JOINT_GRAPH_FLAG = 'use_joint_graph_encoder'
E1_ROW_FLAG = 'e1_base_distance_row'


def _canonical_model(model):
    result = {}
    for name in (*FLAGS, *FEATURE_FLAGS):
        value = model.get(name, False)
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be a boolean')
        result[name] = value
    e1_row = model.get(E1_ROW_FLAG, False)
    if not isinstance(e1_row, bool):
        raise ValueError('e1_base_distance_row must be a boolean')
    if e1_row:
        result[E1_ROW_FLAG] = True  # optional: preserve all historical disabled signatures
    joint_graph = model.get(JOINT_GRAPH_FLAG, False)
    if not isinstance(joint_graph, bool):
        raise ValueError('use_joint_graph_encoder must be a boolean')
    graph_dimension = model.get('joint_graph_edge_dim', 32)
    if (isinstance(graph_dimension, bool) or not isinstance(graph_dimension, int)
            or graph_dimension < 1):
        raise ValueError('joint_graph_edge_dim must be a positive integer')
    graph_dropout = model.get('joint_graph_dropout', 0.0)
    if (isinstance(graph_dropout, bool) or not isinstance(graph_dropout, (int, float))
            or not math.isfinite(graph_dropout) or graph_dropout != 0.0):
        raise ValueError('joint_graph_dropout must be zero for deterministic cached PPO')
    if joint_graph:
        if any(result[name] for name in ('use_edge_relation_encoder', 'use_edge_value_messages',
                                         'use_edge_state_updates')):
            raise ValueError('use_joint_graph_encoder replaces the legacy edge relation encoder/messages/updates')
        # Optional keys preserve all pre-graph v1 signatures when disabled.
        result.update(use_joint_graph_encoder=True, joint_graph_edge_dim=graph_dimension,
                      joint_graph_dropout=0.0)
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
    return any(profile.get(name, False) for name in (*FLAGS, *FEATURE_FLAGS, JOINT_GRAPH_FLAG, E1_ROW_FLAG))


def signature(cfg):
    model = cfg.get('model', {}) or {}
    result = {'schema': SCHEMA, **_canonical_model(model)}
    if any(result[name] for name in FLAGS) or result.get(JOINT_GRAPH_FLAG, False):
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
        if computed.get(JOINT_GRAPH_FLAG, False) or (require_metadata and enabled(computed)):
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
        if enabled(source) or enabled(actual):
            raise ValueError('Configure the target model integration before loading stage-two weights')
        target = source
    _check_agent(agent, target)
    if source != target and (source.get(JOINT_GRAPH_FLAG, False) or target.get(JOINT_GRAPH_FLAG, False)):
        raise ValueError('Joint graph architecture changed; graph checkpoints require an identical '
                         'model integration profile. Start the new architecture from scratch.')
    if source != target and (source.get(E1_ROW_FLAG, False) or target.get(E1_ROW_FLAG, False)):
        raise ValueError('E1 shared distance-row profile changed; use an identical profile or start from scratch')
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
