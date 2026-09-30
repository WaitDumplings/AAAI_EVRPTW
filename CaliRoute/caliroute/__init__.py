"""CaliRoute paper-facing training interface."""

from .config import build_training_config
from .methods import canonical_method, method_preset

__all__ = ["build_training_config", "canonical_method", "method_preset"]
