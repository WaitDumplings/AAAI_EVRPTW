"""Run the immutable validation-selected checkpoint on the independent test."""
import argparse,copy,json
from pathlib import Path
import torch,yaml
from e1.configs import digest
from e1.evaluation import evaluate_checkpoint,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-audit',type=Path,required=True)
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());audit=json.loads(a.data_audit.read_text())
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    saved=copy.deepcopy(payload['config'])
    if 'entropy_initial_coef' not in cfg['training']:
        if saved['training'].get('entropy_initial_coef')!=cfg['training']['ent_coef']:
            raise ValueError('Entropy initialization differs from frozen protocol')
        saved['training'].pop('entropy_initial_coef',None)
    for section in ('training','offline'):
        for key in ('resume_checkpoint_path','resume_start_epoch','stop_after_epoch'):
            saved.get(section,{}).pop(key,None)
    if saved!=cfg or payload['seed']!=cfg['experiment_protocol']['training_seed']:
        raise ValueError('Best checkpoint does not belong to the immutable formal config')
    if payload['config']['experiment_protocol'].get('hardware_preflight_only'):raise ValueError('Cannot test preflight weights')
    test=Path(audit['test']['path']);assert digest(test)==audit['test']['sha256']
    target=a.config.parent/'test/instances.jsonl';target.parent.mkdir(exist_ok=True)
    summary=evaluate_checkpoint(cfg,a.checkpoint,target,test_path=test.parent,device='cuda:0',batch_size=8)
    rows=[json.loads(line) for line in target.read_text().splitlines()];ids=[r['instance_id'] for r in rows]
    expected=set(audit['test']['instance_ids'])
    if len(ids)!=1000 or len(set(ids))!=1000 or set(ids)!=expected:raise ValueError('Test instance coverage mismatch')
    ch=digest(a.checkpoint)
    for r in rows:
        if r['actual_K']!=50 or r['requested_K']!=50 or r['checkpoint_id']!=ch or r['split']!='test':raise ValueError('Test protocol mismatch')
        if r['feasible'] and r.get('selected_cost_matches_reported') is False:raise ValueError('Physical road cost disagrees')
    atomic_json(target.parent/'manifest.json',dict(accepted=True,variant=cfg['experiment_protocol']['variant'],
        checkpoint=str(a.checkpoint),checkpoint_sha256=ch,best_epoch=payload['epoch'],test_sha256=audit['test']['sha256'],
        instances_sha256=digest(target),summary=summary,seed=payload['seed'],K=50,batch=8,
        timing='batch-amortized candidate generation + independent validation + feasible selection; excludes model/data loading'))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
