"""E3 initial five matched CVRP100 experiments, plus the matching future B2."""
from e1.configs import build_config, content_hash
VARIANTS = {
 'A0_no_bias': ('none', 'road_row'),
 'AE_euclidean_bias': ('euclidean', 'road_row'),
 'AR_road_bias': ('road', 'road_row'),
 'B0_static_candidates': ('road', 'static'),
 'B1_feature_matched_logit': ('road', 'logit_mlp'),
 'B2_agda': ('road', 'agda'),
}
PLANS = {
 '2080ti_a': [('A0_no_bias', [0,1]), ('AE_euclidean_bias', [2,3])],
 '2080ti_3': [('AR_road_bias', [0,1])],
 '2080ti_b': [('B0_static_candidates', [0,1]), ('B1_feature_matched_logit', [2,3])],
}

def build(variant, **kwargs):
    metric, mode = VARIANTS[variant]
    cfg = build_config('ppo_rdi_agda', **kwargs)
    cfg['model'].update(use_joint_graph_encoder=False, use_edge_relation_encoder=False,
        use_edge_state_updates=False, use_edge_value_messages=False, use_physical_input_context=False,
        use_typed_static_fusion=False, use_resource_decoder=False, use_rdi_v2=False, use_agda_v2=False,
        use_residual_edge_bias=False, use_post_charge_adapter=False,
        use_encoder_distance_bias=metric!='none', distance_injection='encoder' if metric!='none' else 'none',
        e1_base_distance_row=mode=='road_row', e1_no_road_input=False,
        use_dynamic_decision_encoder=mode in ('logit_mlp','agda'),
        dynamic_decision_delta_k=False, dynamic_decision_delta_v=False,
        dynamic_decision_delta_action_key=True, dynamic_decision_action_bias=True,
        agda_physical_candidate_features=False, agda_smooth_distance_features=False,
        e3_profile=dict(encoder_metric=metric,candidate_mode=mode))
    cfg['env']['observation_input_context'] = False
    cfg['experiment_protocol'].update(experiment='E3', variant=variant,
        actor_sha256=content_hash(cfg['model']), initialization='scratch',
        e3_schema='e3_original_bias_v1', reuse_e1_result=False,
        actual_algorithm='PPO', legacy_worker_method_label='ppo_rdi_agda',
        distance_contract='All rewards, dynamics and reported costs use directed road km; only encoder bias varies',
        e3_mismatch_diagnostic=dict(k_nearest=10, metric='mean customer outgoing top10 set disagreement',
            bins=[0,1/3,2/3,1], fixed_before_test=True))
    return cfg
