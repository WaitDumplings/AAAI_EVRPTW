"""Hierarchical EVRP-TW-D generator.

The package implements a two-stage benchmark generator:

1. A city/region service-territory graph with a static road graph and latent demand pool.
2. Daily EVRP-TW-D instances sampled from that region.
"""

from .generation.generator import HierarchyDatasetGenerator

__all__ = ["HierarchyDatasetGenerator"]
