"""Prepare and supervise the first E3 five runs; no experiment runs on import."""
import argparse,copy,csv,fcntl,hashlib,inspect,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
import yaml
from e1.configs import ROOT,digest,content_hash,fit_train_distance_unit
from e1.evaluation import atomic_json
from e1.runner import _runtime
from e3.configs import PLANS,build


def dump(path,cfg):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg,sort_keys=False))

def audit_data(root):
    from offline2online.instance_adapter import iter_adapted_instances
    audit={}
    for split,n in [('train',5000),('val',1000),('test',1000)]:
        path=root/(f'test_release/cvrp/test/Cus100/instances.pkl' if split=='test' else f'dataset/cvrp/{split}/Cus100/instances.pkl')
        if not path.is_file():raise FileNotFoundError(path)
        ids=[str(i.instance_id) for i in iter_adapted_instances(path.parent,problem_type='cvrp',strict_road_metric=True)]
        if len(ids)!=n or len(set(ids))!=n:raise ValueError(f'{split}: expected {n} unique instances')
        audit[split]=dict(path=str(path),sha256=digest(path),count=n,instance_ids=ids)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(audit[a]['instance_ids'])&set(audit[b]['instance_ids']):raise ValueError('Cross-split ID overlap')
    return audit

def initialization(cfg):
    import torch
    from offline2online.models import Agent
    from offline2online.input_normalization import configure as inputs
    from offline2online.model_integration import configure
    torch.manual_seed(cfg['experiment_protocol']['training_seed'])
    agent=Agent(device='cpu',**{k:v for k,v in cfg['model'].items() if k in inspect.signature(Agent).parameters})
    inputs(agent,cfg);configure(agent,cfg)
    weights=agent.state_dict();common=hashlib.sha256()
    for key,v in sorted(weights.items()):
        if '.e1_distance_row.' in key or '.e3_logit_head.' in key:continue
        common.update(key.encode());common.update(v.detach().cpu().numpy().tobytes())
    core=agent.backbone.decoder.dynamic_graph_kv_encoder
    return dict(shared_parameters_sha256=common.hexdigest(),registered_parameters=sum(p.numel() for p in agent.parameters()),
        trainable_parameters=sum(p.numel() for p in agent.parameters() if p.requires_grad),
        candidate_parameter_match=getattr(core,'e3_parameter_match',None),seed=cfg['experiment_protocol']['training_seed'])

def prepare(a):
    if a.output.exists():raise FileExistsError('Campaign already exists; use --resume, never overwrite')
    a.output.mkdir(parents=True);audit=audit_data(a.data_root);atomic_json(a.output/'data_audit.json',audit)
    unit=fit_train_distance_unit(a.data_root/'dataset/cvrp/train/Cus100');atomic_json(a.output/'distance_unit.json',unit)
    # Generate configurations before the snapshot so preflight cannot see later edits.
    configs={};inits={}
    backup=a.output.parent/'checkpoint_backups'/a.output.name
    for name,gpus in PLANS[a.server]:
        run=a.output/name
        cfg=build(name,data_root=a.data_root,output_dir=run,distance_unit_km=unit['km'],seed=a.seed,
            epochs=a.epochs,world_size=2,global_batch=64,ppo_chunk=min(a.ppo_chunk,16) if name=='B1_feature_matched_logit' else a.ppo_chunk,eval_batch=8,backup_root=backup)
        cfg['run_name']=a.output.name+'_'+name
        cfg['experiment_protocol']['data_audit_sha256']=digest(a.output/'data_audit.json')
        configs[name]=cfg;inits[name]=initialization(cfg);atomic_json(run/'initialization.json',inits[name])
    if len({i['shared_parameters_sha256'] for i in inits.values()})!=1:raise ValueError('Initial shared weights mismatch')
    code=a.output/'source/CaliRoute';source=_runtime().source_snapshot(code,include_initialization_assets=False)
    (code/'results').unlink();(a.output/'artifacts').mkdir();(code/'results').symlink_to(a.output/'artifacts',target_is_directory=True)
    specs={}
    for name,gpus in PLANS[a.server]:
        cfg=configs[name];cfg['experiment_protocol']['source_content_sha256']=source['content_sha256']
        run=a.output/name;dump(run/'config.yaml',cfg)
        smoke=copy.deepcopy(cfg);smoke['run_name']+='__PREFLIGHT'
        smoke['training'].update(epochs=2,checkpoint_interval=1,latest_checkpoint_interval=1,
            monitor_interval=1,monitor_output_dir=str(run/'preflight/monitoring'))
        smoke['evaluation'].update(eval_interval=2,eval_limit=4,eval_batch_size=4,eval_output_dir=str(run/'preflight/validation'))
        smoke['experiment_protocol']['hardware_preflight_only']=True
        dump(run/'preflight/config.yaml',smoke)
        specs[name]=dict(gpus=gpus,config=str(run/'config.yaml'),config_sha256=digest(run/'config.yaml'),
            preflight_config=str(run/'preflight/config.yaml'),preflight_sha256=digest(run/'preflight/config.yaml'),run_dir=str(run))
    manifest=dict(schema='aaai_e3_initial_five_v1',server=a.server,created_at=time.time(),state='prepared',
        campaign=str(a.output),code_root=str(code),source=source,methods=specs,
        data_audit_sha256=digest(a.output/'data_audit.json'),distance_unit=unit,epochs=a.epochs,seed=a.seed,
        global_batch=64,n_traj=50,ppo_passes=4,optimizer_attempts_per_loop=16,
        initialization='scratch; preflight checkpoints never reused',initializations=inits)
    atomic_json(a.output/'manifest.json',manifest)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv'],text=True)
    atomic_json(a.output/'hardware.json',dict(gpus=hardware,python=sys.version,packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)))
    return manifest

def verify(m):
    for name,h in m['source']['files'].items():
        if digest(Path(m['code_root'])/name)!=h:raise ValueError('Frozen source changed: '+name)
    audit=Path(m['campaign'])/'data_audit.json'
    if digest(audit)!=m['data_audit_sha256']:raise ValueError('Data audit changed')
    for item in json.loads(audit.read_text()).values():
        if digest(item['path'])!=item['sha256']:raise ValueError('Data changed: '+item['path'])
    for spec in m['methods'].values():
        for key,h in [('config','config_sha256'),('preflight_config','preflight_sha256')]:
            if digest(spec[key])!=spec[h]:raise ValueError('Frozen config changed')

def ckpt_dir(m,cfg):
    return Path(m['campaign'])/'artifacts/checkpoints/Cus_100_CS_0'/cfg['run_name']/f'seed_{m["seed"]}'

def latest_rows(m,cfg):
    log=Path(m['campaign'])/'artifacts/logs/Cus_100_CS_0'/cfg['run_name']/f'seed_{m["seed"]}'
    result={}
    for name in ('train','eval'):
        file=log/f'{name}_log.csv'
        if file.exists():
            with file.open() as f:rows=list(csv.DictReader(f))
            if rows:result['latest_'+name]=rows[-1]
    return result

def publish(m,states):
    table=ROOT/'docs/e3/E3_experiment_table.csv'
    if not table.exists():return
    lock=table.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with table.open() as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    for row in rows:
        name=row['config_id'];spec=m['methods'].get(name)
        if not spec or row['task']!='CVRP':continue
        state=states[name];run=Path(spec['run_dir'])
        row.update(server=m['server'],gpu=','.join(map(str,spec['gpus'])),training_seed=str(m['seed']),seed_status='已冻结',
            training_plan='新增',status=state['state'],run_path=str(run),config_path=spec['config'])
        accepted=run/'test/manifest.json'
        if accepted.exists():
            evidence=json.loads(accepted.read_text())
            if evidence.get('accepted'):
                s=evidence['summary'];row.update(status='独立 test 已验收',evaluation_status='accepted',
                    feasible_count=str(s['feasible_count']),mean_cost_km=str(s['mean_distance_km']),
                    inference_s_per_instance=str(s['mean_runtime_s']),checkpoint_path=evidence['checkpoint'],
                    instances_path=str(run/'test/instances.jsonl'),parameter_count=str(m['initializations'][name]['trainable_parameters']))
    tmp=table.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    tmp.replace(table);lock.close()
    render=subprocess.run([sys.executable,str(ROOT/'scripts/render_e3_table.py')],capture_output=True,text=True)
    if render.returncode:
        print('Table CSV updated; rendering unavailable: '+render.stderr[-1000:],flush=True)

def supervise(m,resume=False):
    runtime=_runtime();a=Path(m['campaign']);cards=runtime.gpu_snapshot()
    locks=[];active={}
    try:
        for gpu in sorted({g for s in m['methods'].values() for g in s['gpus']}):
            if gpu not in cards:raise ValueError(f'GPU {gpu} not available')
            lock=runtime.acquire_gpu_lock(cards[gpu]['uuid'])
            if lock is None:raise RuntimeError(f'GPU {gpu} reserved')
            locks.append(lock)
            if not runtime.idle_gpu(cards[gpu]):raise RuntimeError(f'GPU {gpu} busy; no jobs killed')
        verify(m)
        m['state']='running';atomic_json(a/'manifest.json',m)
        configs={name:yaml.safe_load(Path(s['config']).read_text()) for name,s in m['methods'].items()}
        states={};active={};logs={}
        for name,s in m['methods'].items():
            run=Path(s['run_dir']);prior=run/'status.json'
            state=json.loads(prior.read_text()) if prior.exists() else {'state':'prepared'}
            if state.get('state')=='completed' and (run/'test/manifest.json').exists():states[name]=state;continue
            if (run/'training_result.json').exists() and json.loads((run/'training_result.json').read_text()).get('state')=='completed':stage='test'
            elif (run/'preflight/accepted.json').exists():stage='training'
            else:stage='preflight'
            states[name]=dict(state=stage,started_at=state.get('started_at',time.time()),gpus=s['gpus'])
        def launch(name):
            spec=m['methods'][name];state=states[name];run=Path(spec['run_dir']);stage=state['state'];cfg=configs[name]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,spec['gpus'])),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=m['code_root'],EVRPTW_DB_ROOT=m['code_root'],PYTHONUNBUFFERED='1')
            if stage=='test':
                best=ckpt_dir(m,cfg)/'checkpoint_best.pt'
                if not best.exists():raise FileNotFoundError('Missing validation-selected best checkpoint')
                cmd=[sys.executable,'-m','e3.evaluate','--config',spec['config'],'--checkpoint',str(best),'--data-audit',str(a/'data_audit.json')]
                logfile=run/'test.log'
            else:
                path=spec['preflight_config'] if stage=='preflight' else spec['config']
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','e1.worker','--method','ppo_rdi_agda','--config',path,'--device','cuda']
                logfile=run/('preflight/train.log' if stage=='preflight' else 'train.log')
                if resume and logfile.exists():
                    actual=yaml.safe_load(Path(path).read_text());latest=ckpt_dir(m,actual)/'checkpoint_latest.pt'
                    if not latest.exists():raise RuntimeError('Interrupted run has no latest checkpoint; use a new campaign')
                    cmd+=['--resume',str(latest)]
            logfile.parent.mkdir(parents=True,exist_ok=True);logs[name]=logfile.open('a')
            active[name]=subprocess.Popen(cmd,cwd=m['code_root'],env=env,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
            state.update(pid=active[name].pid,command=cmd,log=str(logfile),stage_started_at=time.time())
            atomic_json(run/'status.json',state);print(name,stage,'PID',active[name].pid,flush=True)
        for name in states:
            if states[name]['state']!='completed':launch(name)
        publish(m,states)
        while active:
            for name,proc in list(active.items()):
                cfg=configs[name];s=states[name];spec=m['methods'][name];run=Path(spec['run_dir'])
                s.update(latest_rows(m,cfg));code=proc.poll()
                if code is None:continue
                logs[name].close();del active[name]
                if code!=0:
                    s.update(state='failed',returncode=code,failed_at=time.time());continue
                if s['state']=='preflight':
                    result=json.loads((run/'preflight/training_result.json').read_text())
                    if result['state']!='completed':raise RuntimeError('Incomplete preflight')
                    scfg=yaml.safe_load(Path(spec['preflight_config']).read_text())
                    evals=list((run/'preflight/validation').glob('*.summary.json'))
                    if not evals:raise RuntimeError('Preflight did not independently validate routes')
                    pre_rows=latest_rows(m,scfg)
                    if int(float(pre_rows.get('latest_train',{}).get('epoch',0)))!=2:
                        raise RuntimeError('Preflight did not complete two training iterations')
                    atomic_json(run/'preflight/accepted.json',dict(accepted=True,epochs=2,world_size=2,config_sha256=spec['preflight_sha256'],completed_at=time.time()))
                    s['state']='training';launch(name)
                elif s['state']=='training':
                    result=json.loads((run/'training_result.json').read_text())
                    if result['state']!='completed':raise RuntimeError('Incomplete formal training')
                    s['training_process_wall_s']=time.time()-s['stage_started_at'];s['state']='test';launch(name)
                else:
                    accepted=json.loads((run/'test/manifest.json').read_text())
                    if not accepted.get('accepted'):raise RuntimeError('Test not accepted')
                    s.update(state='completed',completed_at=time.time())
            for name,s in states.items():atomic_json(Path(m['methods'][name]['run_dir'])/'status.json',s)
            atomic_json(a/'status.json',dict(updated_at=time.time(),server=m['server'],arms=states))
            hardware=runtime.gpu_snapshot()
            with (a/'hardware.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpus=hardware))+'\n')
            if active:time.sleep(15)
        publish(m,states)
        m['state']='completed' if all(s['state']=='completed' for s in states.values()) else 'failed'
        atomic_json(a/'manifest.json',m)
        if m['state']=='failed':raise RuntimeError('One or more arms failed; inspect status/train.log')
    except BaseException:
        # Stop only workers owned by this scheduler, never unrelated GPU jobs.
        for proc in active.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc in active.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        raise
    finally:
        for lock in locks:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--server',choices=PLANS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parent/'AAAI_Dataset');p.add_argument('--seed',type=int,default=3009)
    p.add_argument('--epochs',type=int,default=1500);p.add_argument('--ppo-chunk',type=int,default=16);p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.data_root=a.data_root.resolve()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=a.output.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        m=json.loads((a.output/'manifest.json').read_text()) if a.resume else prepare(a)
        if m['server']!=a.server:raise ValueError('Cannot change server on resume')
        supervise(m,resume=a.resume)
    except Exception as error:
        atomic_json(a.output/'supervisor_error.json',dict(error=repr(error),time=time.time()));raise
if __name__=='__main__':main()
