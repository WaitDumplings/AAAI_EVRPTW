"""Opt-in E3 architecture contract; installed before optimizer/checkpoint load."""
import copy
import torch
from torch import nn


def validate(profile):
    if not isinstance(profile, dict) or set(profile) != {'encoder_metric', 'candidate_mode'}:
        raise ValueError('E3 requires encoder_metric and candidate_mode')
    if profile['encoder_metric'] not in ('none', 'euclidean', 'road'):
        raise ValueError('Invalid E3 encoder metric')
    if profile['candidate_mode'] not in ('road_row', 'static', 'logit_mlp', 'agda'):
        raise ValueError('Invalid E3 candidate mode')
    return copy.deepcopy(profile)


def install(agent, cfg):
    profile = validate(cfg['model']['e3_profile'])
    backbone = agent.backbone
    previous = backbone.model_integration_settings.get('e3_profile')
    if previous is not None:
        if previous != profile:
            raise ValueError('Cannot change E3 profile after construction')
        return
    env = cfg['env']
    if env['observation_coordinate_mode'] != 'depot_fixed' or not env['observation_distance_scale_km'] > 0:
        raise ValueError('E3 geometry requires depot-centered physical km / one fixed scalar')
    if backbone.use_joint_graph_encoder or backbone.edge_relation_encoder is not None or backbone.physical_input_adapter is not None:
        raise ValueError('E3 v1 uses the original bias RDI, without alternative encoder road pathways')
    if backbone.use_encoder_distance_bias != (profile['encoder_metric'] != 'none'):
        raise ValueError('E3 distance switch mismatch')
    decoder = backbone.decoder
    core = decoder.dynamic_graph_kv_encoder
    mode = profile['candidate_mode']
    if (decoder.e1_distance_row is not None) != (mode == 'road_row'):
        raise ValueError('Only E3-A retains the shared E1 decoder road-row branch')
    if core.enabled != (mode in ('logit_mlp', 'agda')):
        raise ValueError('E3 dynamic candidate switch mismatch')
    backbone.e3_encoder_metric = profile['encoder_metric']
    if mode == 'logit_mlp':
        d, f = core.embedding_dim, core.candidate_feature_dim
        # Match the actually used B2 action-key+bias branch (not dormant K/V).
        target = (3*d*d + sum(p.numel() for m in (core.candidate_delta_base,
                  core.candidate_action_key_delta_proj, core.action_bias_proj) for p in m.parameters()) + 2)
        width = 4*d + f  # node, query, candidate descriptors, decision token, state token
        hidden = max(1, round((target - 2*width - 1) / (width + 2)))
        with torch.random.fork_rng(devices=[]):
            head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        core.e3_logit_head = head.to(next(core.parameters()))
        core.e3_parameter_match = dict(b2_active_candidate_parameters=target,
            b1_candidate_parameters=sum(p.numel() for p in head.parameters()), hidden=hidden,
            input_order=['node_embedding', 'driver_query', 'candidate_features', 'decision_token', 'state_token'])
    # Unused historical branches remain in state_dict for shared initialization,
    # but do not masquerade as trainable parameters or acquire optimizer state.
    core.requires_grad_(mode in ('logit_mlp', 'agda'))
    if mode in ('logit_mlp', 'agda'):
        for name in ('candidate_feature_proj', 'candidate_key_delta_proj', 'candidate_value_delta_proj', 'key_scale', 'value_scale'):
            getattr(core, name).requires_grad_(False)
        if mode == 'logit_mlp':
            for name in ('node_state_proj','decision_state_proj','step_state_proj','candidate_delta_base',
                         'candidate_action_key_delta_proj','action_bias_proj','action_key_scale','action_bias_scale'):
                getattr(core,name).requires_grad_(False)
    if profile['encoder_metric'] == 'none':backbone.dist_bias_scale.requires_grad_(False)
    backbone.model_integration_settings['e3_profile'] = profile
