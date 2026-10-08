"""Scratch experiments cannot silently load weights or execute another source tree."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_scratch_comparison as launch
import run_reward_norm_comparison as shared
import original_scratch_config as original


def base_config():
    return yaml.safe_load((ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml').read_text())


def arm(tmp_path, name, base=None):
    return launch.build_arm(base_config() if base is None else base, arm=name,
        output=tmp_path / name, run_name='SCRATCH_' + name.upper(),
        data_root=tmp_path / 'AAAI_Dataset', seed=3010)


def checkpoint_paths(cfg):
    for section in ('training', 'offline'):
        for key, value in cfg.get(section, {}).items():
            if 'checkpoint' in key and any(word in key for word in ('init', 'initial', 'resume', 'reference')) and not key.endswith('_strict'):
                yield section, key, value


def test_four_arms_are_from_scratch_with_common_original_optimizer_budget(tmp_path):
    base = base_config()
    for section in ('training', 'offline'):
        base[section].update(init_checkpoint_path='old-init.pt', initial_checkpoint_path='old-alias.pt',
                             resume_checkpoint_path='old-resume.pt', reference_checkpoint_path='old-reference.pt')
    base['offline']['resume_start_epoch'] = 250
    base['evaluation'].update(eval_limit=16, eval_num_batches=1)
    before = copy.deepcopy(base)
    configs = {name: arm(tmp_path, name, base) for name in launch.ARMS}
    assert base == before
    for name, cfg in configs.items():
        assert not list(checkpoint_paths(cfg))
        assert not any(key.startswith('resume_') for section in ('training', 'offline') for key in cfg[section])
        training, evaluation = cfg['training'], cfg['evaluation']
        assert training['epochs'] == 300
        assert training['num_envs_per_gpu'] == 64
        assert training['n_traj'] == 50
        assert training['ppo_update_epochs'] == 5
        assert training['num_minibatches'] == 4
        assert training['rollout_steps'] == 201
        assert training['learning_rate'] == 1e-4
        assert training['ent_coef'] == .01
        assert training.get('lr_schedule', 'constant') == 'constant'
        assert cfg['offline']['sl_coef'] == .5
        assert cfg['offline']['use_priority_sampler'] is False
        assert cfg['data']['train_sample_mode'] == 'shuffle_cycle'
        assert evaluation['eval_before_training'] is True
        assert evaluation['eval_seed'] == 17003010
        assert evaluation['eval_interval'] == 50
        assert evaluation['eval_n_traj'] == 50
        assert evaluation['eval_max_steps'] == 201
        assert 'eval_limit' not in evaluation and 'eval_num_batches' not in evaluation
        assert cfg['experiment_protocol']['initialization_mode'] == 'scratch'
    legacy = configs['legacy']
    assert legacy['env'] == original.ORIGINAL_PUBLIC_SLPPO['env']
    assert legacy['model'] == original.ORIGINAL_PUBLIC_SLPPO['model']
    assert 'reward_norm_mode' not in legacy['training']
    assert 'reward_contract' not in legacy['env']
    assert 'prefer_explicit_edge_matrices' not in legacy['env']
    assert legacy['training']['gamma'] == .99
    update_override = next(item for item in legacy['experiment_protocol']['protocol_overrides'] if item['parameter'] == 'training.ppo_update_epochs')
    assert (update_override['original'], update_override['used']) == (4, 5)
    for name in ('physics', 'archive', 'explore'):
        cfg = configs[name]
        assert cfg['training']['gamma'] == 1.
        assert cfg['env']['reward_contract'] == 'strict_distance'
        assert cfg['training']['reward_norm_mode'] == 'physical_shared_popart'
        assert cfg['model'] == configs['physics']['model']
        assert cfg['env'] == configs['physics']['env']
        assert cfg['experiment_protocol']['initial_evaluation_equivalence_group'] == 'physics_archive_explore_random_initialization'
        train = dict(cfg['training']); common = dict(configs['physics']['training'])
        train.pop('monitor_output_dir'); common.pop('monitor_output_dir')
        assert train == common


@pytest.mark.parametrize('name', launch.ARMS)
def test_preflight_keeps_full_training_shape_but_discards_its_state(tmp_path, name):
    cfg = arm(tmp_path, name)
    before = copy.deepcopy(cfg)
    preflight = launch.build_preflight(cfg, tmp_path / name / 'preflight')
    assert cfg == before
    assert not list(checkpoint_paths(preflight))
    for key in ('num_envs_per_gpu', 'n_traj', 'ppo_update_epochs', 'num_minibatches', 'rollout_steps', 'learning_rate', 'ppo_step_chunk_size'):
        assert preflight['training'][key] == cfg['training'][key]
    assert preflight['training']['epochs'] == 2
    assert preflight['evaluation']['eval_limit'] == 4
    assert not preflight['evaluation']['eval_before_training']
    assert preflight['run_name'] != cfg['run_name']


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Tiny git/data fixtures exercise real snapshot/prepare without GPU or weights."""
    repo = tmp_path / 'repo'
    root = repo / 'CaliRoute'
    for relative, content in {
        '.gitignore': 'results/\n',
        'model.py': 'CURRENT = True\n',
        'scripts/run_original_scratch.py': '# external adapter fixture\n',
        'scripts/run_reward_norm_comparison.py': '# supervisor fixture\n',
        'assets/init.pt': 'must never enter scratch source\n',
        'weights.pt': 'must never enter scratch source\n',
        'other/model.pth': 'must never enter scratch source\n',
        'other/model.ckpt': 'must never enter scratch source\n',
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / 'results').mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture source'], cwd=repo, check=True)
    data_root = repo / 'AAAI_Dataset'
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')), ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        directory = data_root / 'dataset/vrptw' / split / 'Cus100'
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b'tiny fixture; never unpickled\n')
    historical = {'offline2online/trainer.py': b'ORIGINAL_TRAINER = True\n',
                  'EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py': b'ORIGINAL_ENV = True\n'}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as output:
        for name, blob in historical.items():
            member = tarfile.TarInfo('CaliRoute/' + name)
            member.size, member.mode = len(blob), 0o644
            output.addfile(member, io.BytesIO(blob))
    check_output = subprocess.check_output
    git_archive_calls = []
    def historical_git(command, **kwargs):
        if command == ['git', 'rev-parse', launch.ORIGINAL_COMMIT + '^{commit}']:
            return launch.ORIGINAL_COMMIT + '\n'
        if command[:2] == ['git', 'archive']:
            git_archive_calls.append(command)
            return archive.getvalue()
        return check_output(command, **kwargs)
    monkeypatch.setattr(launch, 'CODE_ROOT', root)
    monkeypatch.setattr(shared, 'CODE_ROOT', root)
    monkeypatch.setattr(subprocess, 'check_output', historical_git)
    monkeypatch.setattr(shared, 'probe_requested_gpus', lambda _: None)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Scratch preparation tried torch.load'))
    monkeypatch.setattr(shared, 'load_initialization', lambda *a, **k: pytest.fail('Scratch preparation tried load_initialization'))
    real_popen = subprocess.Popen
    launches = []
    def popen(command, *args, **kwargs):
        if command[0] == 'git':
            return real_popen(command, *args, **kwargs)
        launches.append((command, kwargs))
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    args = launch.make_parser().parse_args(['--prepare-only', '--run-id', 'SCRATCH_FIXTURE',
        '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml'),
        '--data-root', str(data_root)])
    run = launch.prepare(args)
    manifest = json.loads((run / 'manifest.json').read_text())
    return SimpleNamespace(run=run, root=root, manifest=manifest, historical=historical,
                           archive_calls=git_archive_calls, launches=launches, args=args)


def test_prepare_freezes_exact_pinned_history_and_excludes_initialization_assets(prepared):
    state = prepared
    manifest = state.manifest
    assert not state.launches
    assert manifest['initialization_mode'] == 'scratch'
    assert manifest['protocol']['ppo_update_epochs'] == 5
    assert manifest['protocol']['global_batch'] == 64
    assert manifest['protocol']['n_traj'] == 50
    assert manifest['protocol']['learning_rate'] == 1e-4
    assert manifest['protocol']['entropy_coef'] == .01
    assert manifest['protocol']['sl_coef'] == .5
    assert manifest['protocol']['initial_evaluation_pairs'] == [['physics', 'archive'], ['physics', 'explore']]
    assert manifest['init_checkpoint'] is None and manifest['init_checkpoint_sha256'] is None
    assert manifest['protocol']['original_commit'] == 'f388343dbb1d54bbd3f76dd29ca95208070d31a8'
    assert state.archive_calls == [['git', 'archive', '--format=tar', launch.ORIGINAL_COMMIT, 'CaliRoute']]
    frozen = Path(manifest['code_root'])
    assert not (frozen / 'assets').exists()
    assert not any(path.suffix in {'.pt', '.pth', '.ckpt'} for path in frozen.rglob('*') if 'results' not in path.parts)
    historic = manifest['additional_sources']['legacy']
    assert historic['source']['git_commit'] == launch.ORIGINAL_COMMIT
    for name, blob in state.historical.items():
        assert (Path(historic['code_root']) / name).read_bytes() == blob
    assert manifest['arms']['legacy']['code_root'] == historic['code_root']
    for name, spec in manifest['arms'].items():
        assert spec['required_validation_epochs'] == [0, 50, 100, 150, 200, 250, 300]
        assert spec['preflight']['required_validation_epochs'] == [2]
        if name == 'legacy':
            assert '--source-root' in spec['command']
            assert historic['code_root'] in spec['command']
        else:
            assert spec['code_root'] == str(frozen)
            assert 'offline2online.train' in spec['command']
    # Mutating the live checkout cannot change the frozen original/current source.
    (state.root / 'model.py').write_text('CURRENT = False\n')
    assert (frozen / 'model.py').read_text() == 'CURRENT = True\n'
    shared.verify_manifest(manifest)


def test_launch_starts_only_frozen_supervisor_after_scratch_verification(prepared):
    args = copy.copy(prepared.args)
    args.run_id, args.launch, args.prepare_only = 'SCRATCH_LAUNCH_FIXTURE', True, False
    run = launch.prepare(args)
    assert len(prepared.launches) == 1
    command, kwargs = prepared.launches[0]
    manifest = json.loads((run / 'manifest.json').read_text())
    assert command[-2:] == ['--supervise', str(run)]
    assert command[3] == str(Path(manifest['code_root']) / 'scripts/run_reward_norm_comparison.py')
    assert kwargs['cwd'] == Path(manifest['code_root'])
    assert kwargs['start_new_session'] is True


@pytest.mark.parametrize('preflight', [False, True])
def test_unknown_execution_directory_is_rejected(prepared, preflight):
    manifest = copy.deepcopy(prepared.manifest)
    spec = manifest['arms']['legacy']
    if preflight:
        spec = spec['preflight']
    spec['code_root'] = str(prepared.root)
    with pytest.raises(ValueError, match='Unverified execution source'):
        shared.verify_manifest(manifest)


def test_historical_source_tampering_is_rejected(prepared):
    metadata = prepared.manifest['additional_sources']['legacy']
    path = Path(metadata['code_root']) / 'offline2online/trainer.py'
    path.write_text('MODIFIED_TRAINER = True\n')
    with pytest.raises(ValueError, match='additional source changed'):
        shared.verify_manifest(prepared.manifest)


@pytest.mark.parametrize('location', ['top', 'source_metadata', 'arm', 'preflight', 'config_init', 'config_resume', 'config_reference', 'config_pretrained'])
def test_verified_scratch_manifest_still_rejects_checkpoint_injection(prepared, location):
    manifest = copy.deepcopy(prepared.manifest)
    checkpoint = prepared.run / 'forbidden.pt'
    checkpoint.write_bytes(b'checkpoint never opened')
    checksum = shared.digest(checkpoint)
    spec = manifest['arms']['physics']
    if location == 'top':
        manifest.update(init_checkpoint=str(checkpoint), init_checkpoint_sha256=checksum)
    elif location == 'source_metadata':
        manifest['source_init_checkpoint'] = str(checkpoint)
    elif location in ('arm', 'preflight'):
        stage = spec if location == 'arm' else spec['preflight']
        stage.update(resume_checkpoint=str(checkpoint), resume_checkpoint_sha256=checksum)
    else:
        path = Path(spec['config'])
        cfg = yaml.safe_load(path.read_text())
        cfg['offline'][location.removeprefix('config_') + '_checkpoint_path'] = str(checkpoint)
        path.write_text(yaml.safe_dump(cfg))
        # Configuration integrity alone is insufficient: forbidden loading paths
        # must also be rejected after a deliberate config/checksum update.
        spec['config_sha256'] = shared.digest(path)
    with pytest.raises(ValueError, match='[Ss]cratch|checkpoint|initialization'):
        shared.verify_manifest(manifest)


@pytest.mark.parametrize('omitted', [0, 50, 150, 300])
def test_training_completion_requires_every_full_validation_epoch(tmp_path, omitted):
    spec = dict(epochs=300, log_dir=str(tmp_path), checkpoint_dir=str(tmp_path),
                validation_instances=1000, required_validation_epochs=[0, 50, 100, 150, 200, 250, 300])
    (tmp_path / 'train_log.csv').write_text('epoch\n' + ''.join(f'{epoch}\n' for epoch in range(1, 301)))
    (tmp_path / 'checkpoint_final.pt').touch()
    epochs = spec['required_validation_epochs']
    def history(rows):
        (tmp_path / 'eval_log.csv').write_text('epoch,eval_status,eval_num_instances\n' + ''.join(f'{epoch},ok,1000\n' for epoch in rows))
    detail = {'latest_validation': {'epoch': '300', 'eval_status': 'ok', 'eval_num_instances': '1000'}}
    history(epochs)
    assert shared.successful_training(spec, detail)
    history([epoch for epoch in epochs if epoch != omitted])
    assert not shared.successful_training(spec, detail)
    history(epochs)
    text = (tmp_path / 'eval_log.csv').read_text().replace(f'{omitted},ok,1000', f'{omitted},ok,4')
    (tmp_path / 'eval_log.csv').write_text(text)
    assert not shared.successful_training(spec, detail)
