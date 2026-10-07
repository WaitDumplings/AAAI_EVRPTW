"""Portable init preserves the training starting point without optimizer baggage."""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reward_norm_initialization as init


def full_checkpoint():
    return {
        'epoch': 300,
        'seed': 3009,
        'config': {
            'env': {'normalize_reward': True, 'reward_distance_scale_km': 43.638668060302734},
            'model': {'embedding_dim': 4},
            'critic': {'use_decomposed_critic': False},
            'data': {'train_dataset_path': '/source/server/private/data'},
            'offline': {'init_checkpoint_path': '/source/server/old.pt'},
            'training': {'learning_rate': 0.00003},
        },
        'model_state_dict': OrderedDict([
            ('weight', torch.arange(12, dtype=torch.float64).reshape(3, 4).t()),
            ('bias', torch.tensor([1., -0.5], dtype=torch.float32)),
            ('counter', torch.tensor(12, dtype=torch.int64)),
            ('scalar_fp16', torch.tensor(0.5, dtype=torch.float16)),
            ('scalar_bf16', torch.tensor(-0.25, dtype=torch.bfloat16)),
            ('enabled', torch.tensor(True)),
        ]),
        'optimizer_state_dict': {'state': {0: {'exp_avg': torch.ones(4)}}, 'param_groups': []},
        'training_resume_state': {'sampler_state_complete': True, 'epoch': 300},
        'policy_route_pool_state': {'routes': [[0, 1, 0]]},
        'reward_normalization_state': {'actor': {'update_count': torch.tensor(18)}},
    }


def write_source(tmp_path):
    source = tmp_path / 'full.pt'
    payload = full_checkpoint()
    torch.save(payload, source)
    return source, payload


def assert_same_weights(left, right):
    assert set(left) == set(right)
    for name, value in left.items():
        other = right[name]
        assert value.dtype == other.dtype
        assert value.shape == other.shape
        assert torch.equal(value, other)


def test_export_removes_resume_and_server_paths_but_preserves_exact_initialization(tmp_path):
    source, original = write_source(tmp_path)
    original_bytes = source.read_bytes()
    destination = tmp_path / 'portable.pt'
    metadata = init.export_initialization(source, destination)
    payload = torch.load(destination, map_location='cpu', weights_only=True)
    assert set(payload) == {'epoch', 'seed', 'config', 'model_state_dict'}
    assert payload['epoch'] == 300
    assert payload['seed'] == 3009
    assert payload['config'] == {name: original['config'][name] for name in ('env', 'model', 'critic')}
    assert_same_weights(payload['model_state_dict'], original['model_state_dict'])
    assert source.read_bytes() == original_bytes
    assert json.loads(destination.with_suffix('.json').read_text()) == metadata
    loaded, provenance = init.load_initialization(destination, expected_epoch=300,
        metadata_path=destination.with_suffix('.json'))
    assert_same_weights(loaded['model_state_dict'], original['model_state_dict'])
    assert isinstance(provenance, dict)
    assert provenance


def test_fingerprint_is_independent_of_key_order_layout_and_device_copy():
    state = full_checkpoint()['model_state_dict']
    canonical = init.state_dict_digest(state)
    reordered = OrderedDict((key, value.detach().clone().contiguous()) for key, value in reversed(list(state.items())))
    assert init.state_dict_digest(reordered) == canonical


@pytest.mark.parametrize('change', ['value', 'dtype', 'shape', 'key', 'missing'])
def test_fingerprint_detects_different_model_starting_points(change):
    state = full_checkpoint()['model_state_dict']
    other = copy.deepcopy(state)
    if change == 'value':
        other['weight'][0, 0] += 1
    elif change == 'dtype':
        other['weight'] = other['weight'].float()
    elif change == 'shape':
        other['weight'] = other['weight'].reshape(-1)
    elif change == 'key':
        other['renamed_weight'] = other.pop('weight')
    else:
        del other['enabled']
    assert init.state_dict_digest(other) != init.state_dict_digest(state)


def test_corrupt_bundle_is_rejected_before_deserializing(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path)
    destination = tmp_path / 'portable.pt'
    init.export_initialization(source, destination)
    destination.write_bytes(destination.read_bytes() + b'corrupt')
    def forbidden_load(*args, **kwargs):
        pytest.fail('Corrupt bundle reached torch.load before SHA verification')
    monkeypatch.setattr(torch, 'load', forbidden_load)
    with pytest.raises(ValueError):
        init.load_initialization(destination, expected_epoch=300,
            metadata_path=destination.with_suffix('.json'))


def test_wrong_epoch_is_rejected_for_full_and_compact_checkpoints(tmp_path):
    source, _ = write_source(tmp_path)
    destination = tmp_path / 'portable.pt'
    init.export_initialization(source, destination)
    for path, metadata in ((source, None), (destination, destination.with_suffix('.json'))):
        with pytest.raises(ValueError, match='[Ee]poch'):
            init.load_initialization(path, expected_epoch=301, metadata_path=metadata)


def test_explicit_legacy_checkpoint_needs_no_sidecar(tmp_path):
    source, original = write_source(tmp_path)
    payload, provenance = init.load_initialization(source, expected_epoch=300)
    assert payload['epoch'] == 300
    assert_same_weights(payload['model_state_dict'], original['model_state_dict'])
    assert isinstance(provenance, dict)
    assert provenance


def test_model_fingerprint_is_verified_independently_of_archive_sha(tmp_path):
    source, _ = write_source(tmp_path)
    destination = tmp_path / 'portable.pt'
    metadata = init.export_initialization(source, destination)
    payload = torch.load(destination, map_location='cpu', weights_only=True)
    payload['model_state_dict']['weight'][0, 0] += 1
    torch.save(payload, destination)
    # A matching file checksum alone must not conceal changed initialization.
    metadata['sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='[Mm]odel'):
        init.load_initialization(destination, expected_epoch=300,
            metadata_path=destination.with_suffix('.json'))


def test_export_refuses_to_overwrite_an_existing_fixed_initialization(tmp_path):
    source, _ = write_source(tmp_path)
    destination = tmp_path / 'portable.pt'
    init.export_initialization(source, destination)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        init.export_initialization(source, destination)
    assert destination.read_bytes() == before
