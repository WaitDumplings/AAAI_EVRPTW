#!/usr/bin/env python3
"""Evaluate older controlled VRPTW campaigns using their frozen model code.

These campaigns audited train/val at preparation, unlike CVRP's three-split
schema. Audit independent test now without editing training provenance.
"""
import argparse
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def check_split(train_audit, test):
    if test['count'] != 1000 or len(test['instances']) != 1000:
        raise ValueError('Expected exactly 1000 test instances')
    ids = [r['instance_id'] for r in test['instances']]
    if len(set(ids)) != 1000:
        raise ValueError('Duplicate test IDs')
    for split in ('train', 'val'):
        for key in ('instance_id', 'semantic_sha256'):
            if {r[key] for r in test['instances']} & {r[key] for r in train_audit[split]['instances']}:
                raise ValueError(f'Test overlaps {split}: {key}')


def check_rows(rows, audit, method, seed, checkpoint_sha):
    ids = [str(r['instance_id']) for r in rows]
    expected = {str(r['instance_id']) for r in audit['instances']}
    if len(ids) != 1000 or len(set(ids)) != 1000 or set(ids) != expected:
        raise ValueError('Test identities differ from the independently audited split')
    for r in rows:
        if (r.get('method') != method or r.get('training_seed') != seed or r.get('split') != 'test'
                or r.get('checkpoint_id') != checkpoint_sha or r.get('actual_K') != 50
                or r.get('requested_K') != 50):
            raise ValueError('Test method/seed/split/checkpoint/K mismatch')


def assert_accepted(manifest, method):
    from e1.configs import digest
    from e1.lifecycle import assert_test_ready
    folder = Path(manifest['methods'][method]['run_dir']) / 'test'
    accepted = read(folder / 'manifest.json')
    if accepted.get('accepted') is not True or accepted.get('test_audit_schema') != 'vrptw_post_training_v1':
        raise ValueError('Missing accepted VRPTW test marker')
    if digest(folder / 'data_audit.json') != accepted['test_audit_sha256']:
        raise ValueError('Test audit changed')
    audit = read(folder / 'data_audit.json')
    if digest(audit['source_path']) != accepted['test_data_sha256'] or accepted['test_data_sha256'] != audit['source_sha256']:
        raise ValueError('Test data changed')
    check_split(read(manifest['audit_path']), audit)
    check = assert_test_ready(manifest, method, accepted['checkpoint'])
    if check['checkpoint_sha256'] != accepted['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if digest(folder / 'instances.jsonl') != accepted['instances_sha256']:
        raise ValueError('Test outputs changed')
    rows = [json.loads(line) for line in (folder / 'instances.jsonl').read_text().splitlines() if line.strip()]
    check_rows(rows, audit, method, manifest['training_seed'], accepted['checkpoint_sha256'])
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--method', choices=('ppo_base','ppo_rdi_agda'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    manifest = read(args.campaign / 'manifest.json')
    # All model/env/evaluation and checkpoint validation imports come from frozen code.
    sys.path.insert(0, manifest['code_root'])
    import yaml
    from e1.runner import verify_campaign, _runtime
    from e1.lifecycle import assert_test_ready
    from e1.native import export_native_dataset
    from e1.configs import digest
    from e1.evaluation import evaluate_checkpoint, atomic_json
    verify_campaign(args.campaign)
    if manifest['schema'] != 'aaai_e1_vrptw100_v1':
        raise ValueError('This adapter requires the controlled VRPTW100 campaign schema')
    spec = manifest['methods'][args.method]
    cfg = yaml.safe_load(Path(spec['config']).read_text())
    acceptance = assert_test_ready(manifest, args.method, args.checkpoint)
    val = Path(cfg['evaluation']['eval_path'])
    if val.parent.name != 'val':
        raise ValueError('Unexpected validation dataset layout')
    if val.parents[1].name != 'vrptw' or val.parents[2].name != 'dataset':
        raise ValueError('Unexpected training data root layout')
    test_path = val.parents[3] / 'test_release' / 'vrptw' / 'test' / val.name
    if not (test_path / 'instances.pkl').is_file():
        raise FileNotFoundError(test_path / 'instances.pkl')
    folder = Path(spec['run_dir']) / 'test'
    if folder.exists():
        raise FileExistsError('Existing final test artifacts; inspect before explicit retry')
    held = None
    try:
        if args.device.startswith('cuda'):
            gpu = int(args.device.split(':')[1]) if ':' in args.device else 0
            runtime = _runtime(); cards = runtime.gpu_snapshot()
            runtime.validate_requested_gpus([gpu], cards)
            if not runtime.idle_gpu(cards[gpu]):
                raise ValueError('Requested evaluation GPU is busy')
            held = runtime.acquire_gpu_lock(cards[gpu]['uuid'])
            if held is None or not runtime.idle_gpu(runtime.gpu_snapshot()[gpu]):
                raise ValueError('Evaluation GPU became reserved/busy')
        folder.mkdir(exist_ok=False)
        audit = export_native_dataset(test_path, folder/'audit_arrays.npz',split='test',expected_customers=100,problem_type='vrptw')
        check_split(read(manifest['audit_path']), audit)
        atomic_json(folder/'data_audit.json', audit)
        (folder/'audit_arrays.npz').unlink()
        result = evaluate_checkpoint(cfg,args.checkpoint,folder/'instances.jsonl',test_path=test_path,device=args.device,batch_size=args.batch_size)
        rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text().splitlines() if line.strip()]
        check_rows(rows,audit,args.method,manifest['training_seed'],acceptance['checkpoint_sha256'])
        if digest(audit['source_path']) != audit['source_sha256']:
            raise ValueError('Test data changed during evaluation')
        atomic_json(folder/'manifest.json',dict(accepted=True,**acceptance,checkpoint=str(args.checkpoint.resolve()),
            test_audit_schema='vrptw_post_training_v1',test_audit_sha256=digest(folder/'data_audit.json'),
            test_data_sha256=audit['source_sha256'],instances_sha256=digest(folder/'instances.jsonl'),
            evaluation_adapter=str(Path(__file__).resolve()),evaluation_adapter_sha256=digest(__file__),
            batch_size=args.batch_size,result=result))
        assert_accepted(manifest,args.method)
        print(json.dumps(result,indent=2))
    finally:
        if held is not None:
            held.close()


if __name__ == '__main__':
    main()
