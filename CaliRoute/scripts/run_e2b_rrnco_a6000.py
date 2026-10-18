#!/usr/bin/env python3
"""Only the authorized first queue items: GPU0 B01, GPU1 B04."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from e1.native import atomic_json,file_hash,json_hash
from e2.rrnco_ppo import COMMON
from e2.rrnco_ppo_launch import prepare,EXPERT,now,update_table
from caliroute.methods import method_preset


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--campaign',type=Path,default=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009'))
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(assignment={'0':'B01 PPO','1':'B04 SL-PPO'},common=COMMON),indent=2));return
    a.campaign.mkdir(parents=True,exist_ok=True)
    lock=(a.campaign/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={k.strip():int(v) for k,v in (row.split(',') for row in memory.splitlines())}
    if used['0']>1024 or used['1']>1024:raise RuntimeError(f'GPU occupied: {used}')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    state=dict(state='preparing',pid=os.getpid(),git_head=head,checkout=str(ROOT),started_at=now(),workers={})
    def save():
        state['updated_at']=now();atomic_json(a.campaign/'scheduler_a6000/status.json',state)
    save();handles=[]
    try:
        for mode in ['preflight','train']:
            pending={}
            for gpu,method in enumerate(['ppo','slppo']):
                run=a.campaign/('preflight' if mode=='preflight' else 'runs')/method
                if not run.exists():prepare(ROOT,a.campaign,method,preflight=mode=='preflight')
                if mode=='preflight' and (run/'preflight.json').exists():
                    d=json.loads((run/'preflight.json').read_text())
                    if not d['passed'] or d['protocol']!=COMMON:raise ValueError('Incomplete preflight')
                    continue
                if (run/'train.jsonl').exists() or (run/'native_checkpoints/last.pt').exists():
                    raise FileExistsError(f'Existing work needs explicit recovery: {run}')
                log=run/('preflight.log' if mode=='preflight' else 'console.log')
                handle=log.open('a');handles.append(handle)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                cmd=[sys.executable,'-u',str(run/'native_harness/scripts/run_e2b_rrnco_worker.py'),mode,'--run-dir',str(run)]
                proc=subprocess.Popen(cmd,env=env,cwd=run,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                pending[method]=(proc,run);state['workers'][method]=dict(pid=proc.pid,gpu=gpu,mode=mode,state='running',run=str(run),log=str(log))
            state['state']=mode;save()
            while pending:
                for method,(proc,run) in list(pending.items()):
                    code=proc.poll()
                    if code is None:continue
                    success=code==0
                    if success:
                        evidence=json.loads((run/('preflight.json' if mode=='preflight' else 'test/manifest.json')).read_text())
                        success=evidence.get('passed' if mode=='preflight' else 'accepted',False)
                    state['workers'][method].update(state='completed' if success else 'failed',exit_code=code,finished_at=now());del pending[method];save()
                if pending:time.sleep(10)
            if any(v['state']=='failed' for v in state['workers'].values()):raise RuntimeError('Worker failed; see its log')
            if mode=='preflight':
                reports={m:json.loads((a.campaign/'preflight'/m/'preflight.json').read_text()) for m in ['ppo','slppo']}
                for key in ['actor_initial_sha256','critic_initial_sha256']:
                    if reports['ppo'][key]!=reports['slppo'][key]:raise ValueError('Arms initialization mismatch')
                atomic_json(a.campaign/'smoke_results.json',reports)
                methods={m:dict(offline=method_preset(m).offline_config(),advantage=method_preset(m).advantage_config(),
                    status='authorized_first_batch' if m in ['ppo','slppo'] else 'not_started_not_queued') for m in ['ppo','awbc','dapg','slppo']}
                for method in ['awbc','dapg','slppo']:
                    methods[method]['offline'].update(use_priority_sampler=False,policy_replay_enabled=False,
                        branch_exploration_enabled=False,exploration_enabled=False,strict_replay=True)
                slcfg=json.loads((a.campaign/'preflight/slppo/native_resolved.json').read_text())['e1_method_config']
                methods['slppo'].update(offline=slcfg['offline'],advantage=slcfg['advantage'])
                for method in methods:methods[method]['implementation_validated']=method in ['ppo','slppo']
                atomic_json(a.campaign/'protocol.json',dict(frozen=True,common=COMMON,common_sha256=json_hash(COMMON),
                    methods=methods,expert_pool_sha256=file_hash(EXPERT),ppo_reads_experts=False,
                    seed_count=1,git_head=head,source_contract='E1 unchanged pure functions plus native RRNCO host adapter',
                    native_e1_result_is_ppo_baseline=False,
                    actor_initial_sha256=reports['ppo']['actor_initial_sha256'],critic_initial_sha256=reports['ppo']['critic_initial_sha256']))
                update_table(a.campaign)
        state['state']='completed';save()
    except Exception as exc:
        state.update(state='failed',error=repr(exc));save();raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':main()
