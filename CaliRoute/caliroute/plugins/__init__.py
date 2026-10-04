"""Portable tensor contracts for optional CaliRoute components.

RDI: directed edge tensors -> attention bias.
AGDA: candidate features and dynamic residuals -> gated dynamic residuals.
SLPPO: route likelihoods and route weights -> solution-level surrogate loss.
"""
from .rdi import RoadDistanceInjection
from .agda import AdaptiveGraphDecisionAdapter, AdaptiveGraphAttention

__all__ = ['RoadDistanceInjection', 'AdaptiveGraphDecisionAdapter', 'AdaptiveGraphAttention']

from .slppo import (RoutePPOResult, solution_level_ppo_loss, clipped_route_surrogate,
                    normalized_route_advantages, replay_weight_schedule)
__all__ += ['RoutePPOResult', 'solution_level_ppo_loss', 'clipped_route_surrogate',
            'normalized_route_advantages', 'replay_weight_schedule']
