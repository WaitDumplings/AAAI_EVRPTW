from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from e1.configs import SCHEMA, build_config, content_hash, digest
from e1.lifecycle import (assert_can_start, assert_checkpoint_owner, assert_test_ready,
                          acquire_run_lock, assert_accepted_test_result)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path,method='ppo_base'):
    campaign=tmp_path/'campaign';run=campaign/'runs'/method;run.mkdir(parents=True)
    index=campaign/'assets/instance_index.csv';index.parent.mkdir()
    with index.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['instance_id','split']);writer.writeheader()
        writer.writerows(dict(instance_id=f'test_{i}',split='test') for i in range(1000))
    audit=campaign/'assets/assets_audit.json'
    write(audit,dict(test_data_ready=True,outputs={'instance_index.csv':{'sha256':digest(index)}},
        splits={'test':{'files':{'instances.pkl':{'sha256':'t'*64}}}}))
    cfg=build_config(method,data_root=tmp_path/'data',output_dir=run,expert_pool=tmp_path/'pool',
        distance_unit_km=5.,epochs=2,world_size=1 if method=='radar' else 2)
    cfg['experiment_protocol'].update(data_audit_sha256=digest(audit),
        source_content_sha256='s'*64,expert_pool_sha256='e'*64 if method in ('awbc','dapg','slppo') else None)
    config=run/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    manifest=dict(schema=SCHEMA,campaign=str(campaign),code_root=str(campaign/'source/CaliRoute'),
        source={'content_sha256':'s'*64},training_seed=3009,audit_path=str(audit),
        methods={method:dict(config=str(config),config_sha256=digest(config),run_dir=str(run))})
    return manifest,cfg,run


def checkpoints(manifest,cfg,run,method='ppo_base'):
    if method!='radar':
        folder=Path(manifest['code_root'])/'results/checkpoints/Cus_100_CS_0'/cfg['run_name']/'seed_3009'
        best=folder/'checkpoint_best.pt';last=folder/'checkpoint_final.pt';folder.mkdir(parents=True)
        payload=dict(config=cfg,seed=3009,epoch=1,model_state_dict={},training_resume_state={})
        torch.save(payload,best)
        final=deepcopy(payload);final['epoch']=cfg['training']['epochs']
        final['training_resume_state']=dict(completed_epoch=final['epoch'],evaluation_pending=False,
            best_validation_selection={'epoch':1},experiment_budget={'instance_exposures':128,'sampled_trajectories':6400})
        torch.save(final,last)
        write(run/'training_result.json',dict(state='completed',last_checkpoint=str(last)))
    else:
        resolved=deepcopy(cfg);native=dict(config_sha256=content_hash(resolved),
            source={'sha256':'n'*64},harness={'sha256':'h'*64},datasets={'train':{'npz_sha256':'d'*64}})
        write(run/'native_resolved.json',resolved);write(run/'native_manifest.json',native)
        folder=run/'native_checkpoints';folder.mkdir();best=folder/'best.pt';last=folder/'last.pt'
        payload=dict(schema='aaai_e1_native_checkpoint_v1',method='radar',model={},config_sha256=native['config_sha256'],
            source_sha256='n'*64,harness_sha256='h'*64,dataset_hashes={'train':'d'*64},
            counters={'instance_exposures':64},best_metric=[1000,-300.])
        torch.save(payload,best);final=deepcopy(payload);final['counters']['instance_exposures']=128;torch.save(final,last)
        write(run/'native_status.json',dict(state='completed',training_complete=True,total_instance_exposures=128,
            counters={'instance_exposures':128},checkpoint={'sha256':digest(last)},best_metric=[1000,-300.]))
    return best,last


def test_start_ignores_only_isolated_smoke_preflight_and_lock(tmp_path):
    manifest,cfg,run=fixture(tmp_path)
    (run/'smoke').mkdir();(run/'hardware_preflight').mkdir()
    with acquire_run_lock(run):assert_can_start(manifest,'ppo_base')
    (run/'train.log').write_text('')
    with pytest.raises(FileExistsError,match='resume'):assert_can_start(manifest,'ppo_base')


@pytest.mark.parametrize('artifact',['training_result.json','monitoring','validation','native_manifest.json','native_data','test'])
def test_start_rejects_formal_artifacts(tmp_path,artifact):
    manifest,cfg,run=fixture(tmp_path);(run/artifact).touch()
    with pytest.raises(FileExistsError):assert_can_start(manifest,'ppo_base')


def test_run_lock_is_nonblocking_and_reusable(tmp_path):
    run=tmp_path/'run'
    with acquire_run_lock(run):
        with pytest.raises(RuntimeError,match='already owned'):
            with acquire_run_lock(run):pass
    with acquire_run_lock(run):pass
    assert (run/'.e1_run.lock').exists()


@pytest.mark.parametrize('key,value',[('run_name','other'),('seed',3010),('method','dapg'),
    ('data_audit_sha256','x'*64),('expert_pool_sha256','e'*64),('source_content_sha256','x'*64),('smoke_only',True)])
def test_controlled_checkpoint_rejects_foreign_identity(tmp_path,key,value):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    payload=torch.load(best,weights_only=False)
    if key=='run_name':payload['config'][key]=value
    elif key=='seed':payload[key]=value
    else:payload['config']['experiment_protocol'][key]=value
    torch.save(payload,best)
    with pytest.raises(ValueError):assert_checkpoint_owner(manifest,'ppo_base',best)


def test_owner_and_test_accept_identical_backup_not_foreign_weights(tmp_path):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    backup=tmp_path/'backup.pt';backup.write_bytes(best.read_bytes())
    assert assert_checkpoint_owner(manifest,'ppo_base',backup)['epoch']==1
    assert assert_test_ready(manifest,'ppo_base',backup)['formal_training_complete']
    with pytest.raises(ValueError,match='selected'):assert_test_ready(manifest,'ppo_base',last)


@pytest.mark.parametrize('field',['state','epoch','completed_epoch','evaluation_pending','instance_exposures','sampled_trajectories','best_epoch'])
def test_final_test_blocked_until_actual_budget_and_validation_complete(tmp_path,field):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    final=torch.load(last,weights_only=False)
    if field=='state':write(run/'training_result.json',dict(state='running',last_checkpoint=str(last)))
    elif field=='epoch':final['epoch']=1
    elif field=='completed_epoch':final['training_resume_state'][field]=1
    elif field=='evaluation_pending':final['training_resume_state'][field]=True
    elif field=='best_epoch':final['training_resume_state']['best_validation_selection']['epoch']=2
    else:final['training_resume_state']['experiment_budget'][field]-=1
    torch.save(final,last)
    with pytest.raises(ValueError):assert_test_ready(manifest,'ppo_base',best)


def test_native_owner_budget_and_source_hashes(tmp_path):
    manifest,cfg,run=fixture(tmp_path,'radar');best,last=checkpoints(manifest,cfg,run,'radar')
    assert assert_test_ready(manifest,'radar',best)['formal_training_complete']
    payload=torch.load(best,weights_only=False);payload['source_sha256']='wrong';torch.save(payload,best)
    with pytest.raises(ValueError,match='source_sha256'):assert_checkpoint_owner(manifest,'radar',best)


def test_native_status_cannot_claim_unfinished_checkpoint_complete(tmp_path):
    manifest,cfg,run=fixture(tmp_path,'radar');best,last=checkpoints(manifest,cfg,run,'radar')
    payload=torch.load(last,weights_only=False);payload['counters']['instance_exposures']=127;torch.save(payload,last)
    with pytest.raises(ValueError,match='budget'):assert_test_ready(manifest,'radar',best)


def accepted(manifest,run,best):
    folder=run/'test';folder.mkdir();output=folder/'instances.jsonl'
    rows=[dict(instance_id=f'test_{i}',method='ppo_base',training_seed=3009,split='test',
               checkpoint_id=digest(best),requested_K=50,actual_K=50) for i in range(1000)]
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    record=dict(accepted=True,checkpoint_sha256=digest(best),instances_sha256=digest(output),test_data_sha256='t'*64)
    write(folder/'manifest.json',record)
    return output,rows,record


def test_accepted_result_requires_explicit_hash_bound_marker(tmp_path):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    output,rows,record=accepted(manifest,run,best)
    assert assert_accepted_test_result(manifest,'ppo_base')['accepted']
    (run/'test/manifest.json').unlink()
    with pytest.raises(FileNotFoundError):assert_accepted_test_result(manifest,'ppo_base')
    record['accepted']=False;write(run/'test/manifest.json',record)
    with pytest.raises(ValueError,match='acceptance'):assert_accepted_test_result(manifest,'ppo_base')


@pytest.mark.parametrize('field,value',[('instance_id','wrong'),('actual_K',49),('requested_K',400),
    ('training_seed',3010),('split','val'),('checkpoint_id','bad'),('method','slppo')])
def test_rejected_thousand_rows_cannot_masquerade_as_completed_test(tmp_path,field,value):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    output,rows,record=accepted(manifest,run,best);rows[0][field]=value
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows));record['instances_sha256']=digest(output)
    write(run/'test/manifest.json',record)
    with pytest.raises(ValueError):assert_accepted_test_result(manifest,'ppo_base')


def test_accepted_result_rejects_modified_bytes(tmp_path):
    manifest,cfg,run=fixture(tmp_path);best,last=checkpoints(manifest,cfg,run)
    output,rows,record=accepted(manifest,run,best);output.write_text(output.read_text()+'\n')
    with pytest.raises(ValueError,match='output changed'):assert_accepted_test_result(manifest,'ppo_base')
