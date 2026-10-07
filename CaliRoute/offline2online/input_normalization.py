"""Validate and persist the model/environment physical input contract.

Input changes are allowed for explicitly weights-only initialization, where the
source and target profiles are recorded. Full optimizer resume requires the
same profile. Older checkpoints retain their legacy behavior through their
saved config; new physical-input checkpoints require explicit profile metadata.

The fixed-unit guarantee applies when observation_distance_scale_km is explicit,
as required for either new input mode. Legacy observations with no explicit
unit still inherit reward/dataset scale at environment construction; that
historical fallback is not resolved or newly frozen by this profile.
"""
from __future__ import annotations

import copy

from caliroute.input_normalization import input_normalization_signature

PROFILE_SCHEMA = "routing_input_normalization_v1"


def signature(cfg):
    env_profile = input_normalization_signature(cfg)
    model = cfg.get("model", {}) or {}
    enabled = bool(model.get("use_physical_input_context", False))
    if enabled != env_profile["observation_input_context"]:
        raise ValueError(
            "model.use_physical_input_context and env.observation_input_context must agree"
        )
    hidden_dim = model.get("physical_input_context_hidden_dim", 32)
    if enabled and (isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1):
        raise ValueError("physical_input_context_hidden_dim must be a positive integer")
    return {
        "schema": PROFILE_SCHEMA,
        **env_profile,
        "use_physical_input_context": enabled,
        "physical_input_context_hidden_dim": hidden_dim if enabled else None,
    }


def _uses_new_inputs(profile):
    return profile["observation_coordinate_mode"] != "legacy_minmax" or profile["observation_input_context"]


def _check_agent(agent, profile):
    backbone = getattr(agent, "backbone", None)
    enabled = getattr(backbone, "physical_input_adapter", None) is not None
    if enabled != profile["use_physical_input_context"]:
        raise ValueError("Agent physical input adapter does not match the configured observation profile")
    if enabled:
        adapter = backbone.physical_input_adapter
        if adapter.node_mlp[0].out_features != profile["physical_input_context_hidden_dim"]:
            raise ValueError("Agent physical input adapter hidden dimension does not match its profile")
        if (adapter.node_feature_dim != len(profile["node_context_features"])
                or adapter.graph_feature_dim != len(profile["graph_context_features"])):
            raise ValueError("Agent physical input context dimensions do not match the observation schema")


def configure(agent, cfg):
    profile = signature(cfg)
    _check_agent(agent, profile)
    previous = getattr(agent, "_input_normalization_signature", None)
    if previous is not None and previous != profile:
        raise ValueError("Input normalization is already configured with a different profile")
    agent._input_normalization_signature = copy.deepcopy(profile)
    return profile


def checkpoint_profile(checkpoint, *, require_physical_metadata=True):
    """Read and cross-check profile against config, with legacy-only fallback."""
    cfg = checkpoint.get("config", {}) or {}
    computed = signature(cfg)
    saved = checkpoint.get("input_normalization_signature")
    if saved is None:
        if require_physical_metadata and _uses_new_inputs(computed):
            raise ValueError("Physical input checkpoint is missing its input normalization signature")
        return computed
    if not isinstance(saved, dict) or saved != computed:
        raise ValueError("Checkpoint input normalization signature does not match its saved config")
    return copy.deepcopy(saved)


def load_checkpoint_profile(agent, checkpoint, *, resume, checkpoint_path=None):
    source = checkpoint_profile(checkpoint, require_physical_metadata=resume)
    target = getattr(agent, "_input_normalization_signature", None)
    if target is None:
        # Preserve historical direct loader calls for legacy models. New input
        # modes require an explicitly configured target, never implicit adoption.
        if _uses_new_inputs(source) or getattr(getattr(agent, "backbone", None), "physical_input_adapter", None) is not None:
            raise ValueError("Configure the target input normalization before loading physical input weights")
        target = source
    _check_agent(agent, target)
    if resume and source != target:
        raise ValueError("Input normalization changed on resume; use weights-only initialization")
    if resume:
        initialization = copy.deepcopy(checkpoint.get("input_normalization_initialization"))
        if initialization is not None:
            agent._input_normalization_initialization = initialization
        return {"source": source, "target": copy.deepcopy(target), "migrated": False, "load": "full_resume"}
    record = {
        "load": "weights_only", "migrated": source != target,
        "source": source, "target": copy.deepcopy(target),
        "source_profile_recorded": checkpoint.get("input_normalization_signature") is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    agent._input_normalization_initialization = copy.deepcopy(record)
    return record


def checkpoint_metadata(agent, cfg):
    profile = signature(cfg)
    _check_agent(agent, profile)
    configured = getattr(agent, "_input_normalization_signature", None)
    if configured is not None and configured != profile:
        raise ValueError("Input normalization config changed after model initialization")
    result = {"input_normalization_signature": profile}
    initialization = getattr(agent, "_input_normalization_initialization", None)
    if initialization is not None:
        result["input_normalization_initialization"] = copy.deepcopy(initialization)
    return result
