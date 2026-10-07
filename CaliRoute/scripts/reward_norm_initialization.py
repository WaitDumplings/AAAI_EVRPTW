"""Export and verify the shared, weights-only reward/norm initialization.

This artifact initializes fresh training; it cannot resume the source run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_dict_digest(state):
    """Fingerprint named tensor values independently of checkpoint serialization."""
    import torch
    if not isinstance(state, dict) or not state:
        raise ValueError('Initialization requires a nonempty model_state_dict')
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('model_state_dict must contain named tensors only')
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_initialization(path, expected_epoch, metadata_path=None):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing initialization: {path}. Pull the experiment branch to obtain '
            'assets/reward_norm/vrptw100_update5_epoch0300.pt, or pass '
            '--init-checkpoint /path/to/the/same/epoch300/checkpoint.pt.')
    checksum = file_digest(path)
    metadata = json.loads(Path(metadata_path).read_text()) if metadata_path is not None else None
    if metadata is not None and checksum != metadata['sha256']:
        raise ValueError(f'Initialization SHA256 mismatch: {path}; restore the tracked asset with git restore')
    # The distributed compact artifact contains tensors and basic Python types.
    # Explicit legacy training checkpoints may also contain NumPy sampler state.
    payload = torch.load(path, map_location='cpu', weights_only=metadata is not None)
    if not isinstance(payload, dict) or int(payload.get('epoch', -1)) != expected_epoch:
        actual = payload.get('epoch') if isinstance(payload, dict) else None
        raise ValueError(f'Checkpoint epoch must equal {expected_epoch}; got {actual}')
    state_sha256 = state_dict_digest(payload.get('model_state_dict'))
    if metadata is not None and state_sha256 != metadata['model_state_sha256']:
        raise ValueError('Initialization model-state SHA256 mismatch')
    provenance = dict(sha256=checksum, model_state_sha256=state_sha256,
        epoch=payload['epoch'], seed=payload.get('seed'), bundled=metadata is not None)
    if metadata is not None:
        provenance['source_checkpoint_sha256'] = metadata['source_checkpoint_sha256']
    return payload, provenance


def export_initialization(source, destination):
    import torch
    source, destination = Path(source), Path(destination)
    metadata_path = destination.with_suffix('.json')
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f'Refusing to replace an existing fixed initialization: {destination}')
    before = file_digest(source)
    original = torch.load(source, map_location='cpu', weights_only=False)
    fingerprint = state_dict_digest(original['model_state_dict'])
    compact = dict(epoch=int(original['epoch']), seed=int(original['seed']),
        config={section: original['config'][section] for section in ('env', 'model', 'critic')
                if section in original['config']}, model_state_dict=original['model_state_dict'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)
    # Re-read serialized tensors to prove no casts, quantization or key changes.
    restored = torch.load(destination, map_location='cpu', weights_only=True)
    if state_dict_digest(restored['model_state_dict']) != fingerprint or file_digest(source) != before:
        raise ValueError('Source changed or model values changed during initialization export')
    metadata = dict(format_version=1, purpose='weights-only initialization; not a training-resume checkpoint',
        filename=destination.name, sha256=file_digest(destination), bytes=destination.stat().st_size,
        model_state_sha256=fingerprint, tensor_count=len(compact['model_state_dict']),
        source_checkpoint_filename=source.name, source_checkpoint_sha256=before,
        epoch=compact['epoch'], seed=compact['seed'], env=compact['config']['env'])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_initialization(args.source, args.output), indent=2))
