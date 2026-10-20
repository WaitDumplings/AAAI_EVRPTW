#!/usr/bin/env python3
"""B02 AWBC on GPU0 and B03 DAPG on GPU1, after accepted B01/B04."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from e1.native import atomic_json,file_hash,json_hash
from e2.rrnco_ppo import COMMON
from e2.rrnco_ppo_launch import prepare,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print({'GPU0':'B02 AWBC','GPU1':'B03 DAPG','common':COMMON});return
    c=a.campaign;lock=(c/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((c/'protocol.json').read_text())
    if protocol['common']!=COMMON:raise ValueError('Common protocol differs from B01/B04')
    for method in ['ppo','slppo']:
        m=json.loads((c/'runs'/method/'test/manifest.json').read_text())
        if not m.get('accepted') or m['protocol_sha256']!=json_hash(COMMON):raise ValueError('First batch is not accepted')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(x)>1024 for x in used.splitlines()):raise RuntimeError('GPU occupied')
    for method in ['awbc','dapg']:
        if (c/'runs'/method).exists() or (c/'preflight'/method).exists():raise FileExistsError('Existing second batch needs explicit recovery')
    old=json.loads((c/'scheduler_a6000/status.json').read_text());atomic_json(c/'scheduler_a6000/first_batch_completed.json',old)
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    state=dict(state='preparing_second_batch',pid=os.getpid(),started_at=now(),git_head=head,workers=old['workers'])
    def save():state['updated_at']=now();atomic_json(c/'scheduler_a6000/status.json',state)
    try:
        for mode in ['preflight','train']:
            pending={}
            for gpu,method in enumerate(['awbc','dapg']):
                run=prepare(ROOT,c,method,preflight=mode=='preflight')
                log=run/('preflight.log' if mode=='preflight' else 'console.log')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with log.open('a') as f:
                    proc=subprocess.Popen([sys.executable,'-u',str(run/'native_harness/scripts/run_e2b_rrnco_worker.py'),mode,'--run-dir',str(run)],cwd=run,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                pending[method]=(proc,run)
                state['workers'][method]=dict(pid=proc.pid,gpu=gpu,state='running',mode=mode,run=str(run),log=str(log))
            state['state']='second_batch_'+mode;save()
            while pending:
                for method,(proc,run) in list(pending.items()):
                    code=proc.poll()
                    if code is None:continue
                    p=run/('preflight.json' if mode=='preflight' else 'test/manifest.json')
                    evidence=json.loads(p.read_text()) if p.exists() else {}
                    success=code==0 and evidence.get('passed' if mode=='preflight' else 'accepted',False)
                    state['workers'][method].update(state='completed' if success else 'failed',exit_code=code,finished_at=now());del pending[method]
                    save()
                if pending:time.sleep(10)
            if any(state['workers'][m]['state']=='failed' for m in ['awbc','dapg']):raise RuntimeError('Second-batch worker failed; see logs')
            if mode=='preflight':
                reports=json.loads((c/'smoke_results.json').read_text())
                for method in ['awbc','dapg']:
                    report=json.loads((c/'preflight'/method/'preflight.json').read_text())
                    for key in ['actor_initial_sha256','critic_initial_sha256']:
                        if report[key]!=protocol[key]:raise ValueError('Initialization differs from first batch')
                    if report['expert_sha256']!=protocol['expert_pool_sha256']:raise ValueError('Expert pool differs')
                    if not report['imitation_audit']['passed']:raise ValueError('Native imitation audit failed')
                    reports[method]=report
                    cfg=json.loads((c/'preflight'/method/'native_resolved.json').read_text())
                    protocol['methods'][method].update(offline=cfg['e1_method_config']['offline'],
                        implementation_validated=True,status='authorized_second_batch',git_head=head,
                        expert_step_interface=report['e1_unchanged_source'],warmup_updates=report['warmup_updates'])
                atomic_json(c/'smoke_results.json',reports);atomic_json(c/'protocol.json',protocol)
        state['state']='completed';save()
    except Exception as exc:
        state.update(state='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
