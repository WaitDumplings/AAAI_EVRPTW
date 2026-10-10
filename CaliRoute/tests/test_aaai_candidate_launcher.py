"""Candidate launch preparation must not drift into historical defaults."""
from pathlib import Path
import json
import sys
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_aaai_candidate as candidate
from test_scratch_comparison import prepared as scratch_prepared


def test_default_is_prepare_without_launch():
    args = candidate.make_parser().parse_args(['--problem', 'vrptw', '--hardware', '2080ti_dual'])
    assert not args.launch and not args.print_config
    assert args.preset == 'core'


def test_print_config_has_no_output_or_process_side_effect(tmp_path, monkeypatch, capsys):
    # Source verification is covered independently; no actual trainer starts.
    monkeypatch.setattr(candidate, 'verify_training_engine', lambda: {'checked': True})
    monkeypatch.setattr(candidate.runtime, 'prepare', lambda *a, **kw: pytest.fail('Must not prepare/launch in print mode'))
    monkeypatch.setattr(candidate, 'CODE_ROOT', tmp_path)
    args = candidate.make_parser().parse_args(['--problem', 'vrptw', '--hardware', '2080ti_dual', '--print-config'])
    assert candidate.prepare(args) is None
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg['training']['num_envs_per_gpu'] == 20
    assert cfg['training']['ppo_update_epochs'] == 5
    assert cfg['training']['ppo_step_chunk_size'] == 48
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('preset', ['core', 'reference'])
@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
@pytest.mark.parametrize('hardware,gpus,world', [('rtx48_single', '0', 1), ('2080ti_dual', '2,3', 2)])
def test_prepare_exports_recipe_actual_eval_batch_and_no_training_process(
        scratch_prepared, monkeypatch, task, hardware, gpus, world, preset):
    state = scratch_prepared
    monkeypatch.setattr(candidate.runtime, 'CODE_ROOT', state.root)
    monkeypatch.setattr(candidate, 'CODE_ROOT', state.root)
    monkeypatch.setattr(candidate, 'verify_training_engine', lambda: {'checked': True})
    recipe_name = 'aaai_graph_core_v1' if preset == 'core' else 'aaai_graph_v1'
    recipe = state.root / f'configs/recipes/{recipe_name}.yaml'
    recipe.parent.mkdir(parents=True)
    recipe.write_text('{}\n')  # Lifecycle reads it; the recipe builder owns config.
    data = state.root.parent / 'AAAI_Dataset'
    for split,count in [('train',5000),('val',1000)]:
        folder = data / 'dataset' / task / split / 'Cus100';folder.mkdir(parents=True, exist_ok=True)
        (folder / 'instances.pkl').write_bytes(b'prepare only fixture')
        (folder / 'metadata.json').write_text(json.dumps(dict(num_instances=count,num_customers=100,
            num_charging_stations=20 if task=='evrptw' else 0)))
        (folder / ('expert_solutions.csv' if split=='train' else 'gurobi_summary.csv')).write_text('instance_id\nfixture\n')
    monkeypatch.setattr(candidate.runtime, 'build_config', lambda *a, **kw: pytest.fail('Historical constructor must not run'))
    args = candidate.make_parser().parse_args(['--preset',preset,'--problem',task,'--hardware',hardware,'--gpus',gpus,
        '--data-root',str(data),'--eval-batch-size','8','--seed','3010','--epochs','300',
        '--run-id',f'TEST_{task}_{hardware}','--prepare-only'])
    run = candidate.prepare(args)
    assert not state.launches
    manifest = json.loads((run/'manifest.json').read_text());cfg = yaml.safe_load(Path(manifest['arms']['graph']['config']).read_text())
    assert manifest['gpus'] == [int(x) for x in gpus.split(',')]
    assert manifest['protocol']['world_size'] == world
    assert manifest['protocol']['eval_batch_size'] == cfg['evaluation']['eval_batch_size'] == 8
    assert manifest['protocol']['gpu_preflight']['eval_batch_size'] == 8
    assert manifest['protocol']['require_preflight_health']
    assert manifest['protocol']['recipe'] == recipe_name
    assert manifest['protocol']['initialization_mode'] == 'scratch'
    assert manifest['protocol']['seed'] == cfg['training']['post_init_seed'] == 3010
    assert manifest['protocol']['epochs'] == cfg['training']['epochs'] == 300
    preflight = yaml.safe_load(Path(manifest['arms']['graph']['preflight']['config']).read_text())
    assert preflight['experiment_protocol']['eval_interval'] == preflight['evaluation']['eval_interval'] == 2
    search_interval = 5 if preset == 'core' else 1
    assert preflight['experiment_protocol']['search_budget']['interval'] == preflight['offline']['exploration_interval'] == search_interval
    assert preflight['experiment_protocol']['resolved_components']['slppo']['search']['interval_epochs'] == search_interval
    assert preflight['experiment_protocol']['resolved_components']['slppo']['search']['requested'] == (preset == 'reference')
    assert cfg['experiment_protocol']['resolved_components']['slppo']['search']['interval_epochs'] == 5
    candidate.runtime.shared.verify_manifest(manifest)
