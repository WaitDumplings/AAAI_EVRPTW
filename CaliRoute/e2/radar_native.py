"""Matched native RADAR variants; separate from the unapproved E2-B queue."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from e1.native import atomic_json, file_hash, json_hash, verify_files
from e1.native_worker import (RadarRunner, run_training, restore_checkpoint,
    evaluate_runner, now, evaluation_rng)

BASE_INITIAL_SHA = '237f292a787d19593e2a34a0bea0dbc03b512762630c042b8bfdd056cc77bc8f'


def make_runner(cfg, run):
    source=run/'native_source';sys.path[:0]=[str(source/'acvrp'),str(source)]
    import ACVRPTrainer as trainer
    from e2.adapters.radar import install_radar_agda
    from e2.adapters.radar_rdi import install_radar_rdi
    native=trainer.Model
    def model(**kw):
        host=native(**kw)
        if cfg['e2_variant']=='rdi':return install_radar_rdi(host)
        host=install_radar_agda(host)
        host.decoder.activation_checkpointing=cfg['agda_activation_checkpointing']
        host.decoder.trajectory_chunk_size=cfg['agda_trajectory_chunk_size']
        return host
    trainer.Model=model
    try:runner=RadarRunner(cfg,run,torch.device('cuda:0'))
    finally:trainer.Model=native
    ids=[id(p) for g in runner.optimizer.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in runner.model.parameters()}:
        raise ValueError('Plugin optimizer coverage is incomplete or duplicated')
    h=hashlib.sha256()
    # Hash in native state_dict order, removing wrapper names and plugin tensors.
    for k,v in runner.model.state_dict().items():
        if k.startswith('rdi.') or '.agda.' in k:continue
        k=k.replace('decoder.native.','decoder.').replace('mixed_score_MHA.native.','mixed_score_MHA.')
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    if h.hexdigest()!=BASE_INITIAL_SHA:
        raise ValueError('Shared host initialization differs from reproduced E1 seed3009')
    runner.architecture.update(e2_variant=cfg['e2_variant'],host_initial_sha256=h.hexdigest(),
        native_svd_retained=True,native_sinkhorn_retained=True,native_trainer_retained=True)
    return runner


def seed_all(cfg):
    torch.set_num_threads(1);torch.cuda.set_device(0)
    random.seed(cfg['training_seed']);np.random.seed(cfg['training_seed']);torch.manual_seed(cfg['training_seed']);torch.cuda.manual_seed_all(cfg['training_seed'])


def verify_run(run):
    cfg=json.loads((run/'native_resolved.json').read_text());manifest=json.loads((run/'native_manifest.json').read_text())
    if json_hash(cfg)!=manifest['config_sha256']:raise ValueError('Changed configuration')
    verify_files(run/'native_source',manifest['source']);verify_files(run/'native_harness',manifest['harness'])
    for split,meta in manifest['datasets'].items():
        path=run/'native_data'/f'{split}.npz'
        if file_hash(path)!=meta['npz_sha256'] or json.loads(path.with_suffix('.json').read_text())!=meta:
            raise ValueError('Changed dataset '+split)
    return cfg,manifest


def run_preflight(run):
    cfg,manifest=verify_run(run);seed_all(cfg)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    runner=make_runner(cfg,run)
    # Actual native batch64, all 50 trajectories and native objective. This run
    # is isolated and never resumed as a formal experiment.
    run_training(runner,cfg,run,manifest,resume=False,max_updates=2)
    trained_hash=file_hash(run/'native_checkpoints/last.pt')
    eval_cfg=deepcopy(cfg);eval_cfg['eval_limit']=2
    eval_cfg['experiment_protocol'].update(smoke_only=True,hardware_preflight_only=True)
    result=evaluate_runner(runner,eval_cfg,run,split='val',checkpoint_id=trained_hash,
        output=run/'preflight_validation.jsonl',inference_seed=cfg['inference_seed'])
    # Reload full optimizer/scheduler/RNG/sampler state, then repeat the same
    # fixed-seed validation; selected physical costs must match exactly.
    restored=make_runner_after_reset(cfg,run)
    restore_checkpoint(restored,cfg,manifest,run/'native_checkpoints/last.pt')
    second=evaluate_runner(restored,eval_cfg,run,split='val',checkpoint_id=trained_hash,
        output=run/'preflight_validation_reload.jsonl',inference_seed=cfg['inference_seed'])
    if result['feasible_instances']!=2 or result['mean_distance_km']!=second['mean_distance_km']:
        raise RuntimeError('Preflight physical validation or checkpoint recovery failed')
    rows=[json.loads(x) for x in (run/'native_train.jsonl').read_text().splitlines()]
    output=dict(passed=True,variant=cfg['e2_variant'],batch=cfg['batch_size'],n_traj=cfg['n_traj'],
        optimizer_updates=2,host_initial_sha256=BASE_INITIAL_SHA,validation=result,reload_validation=second,
        harness_sha256=manifest['harness']['sha256'],
        agda_trajectory_chunk_size=cfg['agda_trajectory_chunk_size'],
        batch_seconds=[r['update_wall_time_s'] for r in rows],
        max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved(),
        elapsed_s=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),finished_at=now())
    atomic_json(run/'preflight.json',output);print(json.dumps(output,indent=2),flush=True)


def make_runner_after_reset(cfg,run):
    seed_all(cfg)
    return make_runner(cfg,run)


def run_formal(run,*,resume=False):
    cfg,manifest=verify_run(run)
    preflight=json.loads(Path(cfg['preflight_result']).read_text())
    if not preflight['passed'] or preflight['batch']!=cfg['batch_size'] or preflight['n_traj']!=cfg['n_traj']:
        raise ValueError('Full-batch GPU preflight is required')
    if (preflight['harness_sha256']!=manifest['harness']['sha256'] or
            preflight['agda_trajectory_chunk_size']!=cfg['agda_trajectory_chunk_size']):
        raise ValueError('Preflight source or memory configuration differs')
    if not resume and (run/'native_train.jsonl').exists():raise FileExistsError('Existing formal training; use explicit resume')
    seed_all(cfg);runner=make_runner(cfg,run)
    atomic_json(run/'native_architecture.json',runner.architecture)
    started=time.perf_counter()
    run_training(runner,cfg,run,manifest,resume=resume,max_updates=None)
    status=json.loads((run/'native_status.json').read_text())
    if not status['training_complete']:raise RuntimeError('Budget not completed')
    best=run/'native_checkpoints/best.pt';state=restore_checkpoint(runner,cfg,manifest,best)
    if tuple(state['best_metric'])!=tuple(status['best_metric']):raise ValueError('Best checkpoint selection mismatch')
    evaluation_cfg=deepcopy(cfg);evaluation_cfg['eval_batch_size']=1
    target=run/'test';target.mkdir(exist_ok=True)
    result=evaluate_runner(runner,evaluation_cfg,run,split='test',checkpoint_id=file_hash(best),
        output=target/'instances.jsonl',inference_seed=cfg['inference_seed'])
    rows=[json.loads(x) for x in (target/'instances.jsonl').read_text().splitlines()]
    if len(rows)!=1000 or len({r['instance_id'] for r in rows})!=1000 or any(r['actual_K']!=50 or r['cost_mismatch_candidate_count'] for r in rows):
        raise RuntimeError('Test inventory or physical-cost audit failed')
    atomic_json(target/'manifest.json',dict(accepted=True,method='radar',variant=cfg['e2_variant'],
        formal_training_complete=True,checkpoint=str(best),checkpoint_sha256=file_hash(best),
        instances_sha256=file_hash(target/'instances.jsonl'),test_data_sha256=manifest['datasets']['test']['source_sha256'],
        result=result,training_seed=cfg['training_seed'],best_metric=list(state['best_metric'])))
    atomic_json(target/'instances.summary.json',result)
    status.update(test_complete=True,test_accepted=True,test_result=result,
        training_and_test_wall_time_s=time.perf_counter()-started,updated=now())
    atomic_json(run/'native_status.json',status)


def prepare_variant(root, baseline_run, campaign, variant, *, chunk=50, preflight=False):
    """Copy frozen source; hardlink immutable verified data into this run."""
    run=campaign/('preflight' if preflight else 'runs')/variant
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True)
    cfg=json.loads((baseline_run/'native_resolved.json').read_text())
    baseline_manifest=json.loads((baseline_run/'native_manifest.json').read_text())
    datasets={s:json.loads((baseline_run/'native_data'/f'{s}.json').read_text()) for s in ['train','val','test']}
    ids=[{r['instance_id'] for r in datasets[s]['instances']} for s in ['train','val','test']]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Split overlap')
    cfg.update(e2_variant=variant,run_id=f'E2_RADAR_CVRP100_{variant.upper()}_S3009_A6000_V1',
        agda_activation_checkpointing=True,agda_trajectory_chunk_size=chunk,
        backup_dir=str(campaign/'checkpoint_backups'/('preflight_' if preflight else '')/variant),
        preflight_result=str(campaign/'preflight'/variant/'preflight.json'))
    cfg['checkpoint_backup_dir']=cfg['backup_dir']
    cfg['native_source_root']=str(run/'native_source')
    cfg['experiment_protocol'].update(panel='E2-A',modules=[variant.upper()],
        e1_matched_baseline_run=str(baseline_run),native_svd_retained=True,
        training_objective='native_REINFORCE_POMO',smoke_only=preflight)
    (run/'native_data').mkdir()
    for split,meta in datasets.items():
        for suffix in ['npz','json']:
            src=baseline_run/'native_data'/f'{split}.{suffix}'
            if suffix=='npz' and file_hash(src)!=meta['npz_sha256']:raise ValueError('Baseline data hash mismatch')
            os.link(src,run/'native_data'/src.name)
    shutil.copytree(baseline_run/'native_source',run/'native_source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    frozen=run/'native_harness';records=[]
    # Snapshot only runtime packages. No dependency on a mutable checkout once
    # launched; snapshot records enter every checkpoint's harness checksum.
    for folder in ['e1','e2','caliroute','offline2online']:
        for src in sorted((root/folder).rglob('*')):
            if src.is_file() and src.suffix in {'.py','.json','.yaml','.yml','.txt'} and '__pycache__' not in src.parts:
                rel=src.relative_to(root);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
                records.append(dict(path=str(rel),sha256=file_hash(dest)))
    script=root/'scripts/run_e2_radar_native.py';dest=frozen/'scripts'/script.name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(script,dest)
    records.append(dict(path='scripts/'+script.name,sha256=file_hash(dest)))
    manifest=dict(config_sha256=json_hash(cfg),source=baseline_manifest['source'],
        harness=dict(files=records,sha256=json_hash(records)),datasets=datasets)
    atomic_json(run/'native_resolved.json',cfg);atomic_json(run/'native_manifest.json',manifest)
    atomic_json(run/'native_status.json',dict(state='prepared',variant=variant,training_complete=False,test_complete=False))
    return run
