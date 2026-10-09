#!/usr/bin/env python3
"""One or two GPUs for one EVRPTW100 or VRPTW100 original/optimized scratch run.

Shell wrappers launch in the background. Original weights, models and losses
come from f388343; an external adapter supplies distributed execution and
measurements. No initialization or resume checkpoint is accepted.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
import run_reward_norm_comparison as shared
import run_scratch_comparison as scratch


def default_data_root():
    """Accept both a dataset inside the checkout's parent and beside the repo."""
    candidates = (CODE_ROOT.parent / 'AAAI_Dataset', CODE_ROOT.parent.parent / 'AAAI_Dataset')
    return next((path for path in candidates if (path / 'dataset').is_dir()), candidates[0])


def dependency_status(after_runs):
    """A queued comparison may start only after every preceding run succeeds."""
    waiting = []
    for directory in after_runs:
        path = Path(directory) / 'status.json'
        try:
            state = json.loads(path.read_text())['state']
        except (OSError, ValueError, KeyError, TypeError) as error:
            return 'failed', f'Cannot verify prerequisite {path}: {error}'
        if state == 'completed':
            continue
        if state in ('failed', 'interrupted'):
            return 'failed', f'Prerequisite {directory} ended with state={state}'
        if state not in ('prepared', 'waiting_dependency', 'waiting_gpu', 'running'):
            return 'failed', f'Prerequisite {directory} has unknown state={state!r}'
        waiting.append(str(directory))
    return ('waiting', 'Waiting for successful completion: ' + ', '.join(waiting)) if waiting else ('completed', None)


def build_config(base, *, variant, output, run_name, data_root, seed=3010,
                 epochs=1500, eval_interval=50, batch_per_gpu=32, chunk_size=8,
                 expert_chunk_size=64, learning_rate=1e-4, task="evrptw",
                 encoder_variant="current", world_size=2):
    if task not in ('evrptw', 'vrptw'):
        raise ValueError('task must be evrptw or vrptw')
    if isinstance(world_size, bool) or world_size not in (1, 2) or not isinstance(world_size, int):
        raise ValueError('world_size must be 1 or 2')
    topology = 'dual' if world_size == 2 else 'single'
    charging_stations = 20 if task == 'evrptw' else 0
    horizon = 512 if task == 'evrptw' else 201
    if variant not in ('original','optimized'):
        raise ValueError('variant must be original or optimized')
    if encoder_variant not in ('current', 'graph'):
        raise ValueError('encoder_variant must be current or graph')
    if encoder_variant == 'graph' and variant != 'optimized':
        raise ValueError('The graph encoder requires --variant optimized; the original baseline is unchanged')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for name, value in dict(epochs=epochs, eval_interval=eval_interval,
                            batch_per_gpu=batch_per_gpu,chunk_size=chunk_size,
                            expert_chunk_size=expert_chunk_size).items():
        shared.positive_integer(value,name)
    if batch_per_gpu % 4:
        raise ValueError('batch-per-gpu must divide into four equal minibatches')
    if chunk_size > horizon:
        raise ValueError(f'chunk-size must be <={horizon} for {task}')
    output, data_root = Path(output), Path(data_root).resolve()
    # The inherited VRPTW template has a 201-action horizon. Apply the requested
    # task's horizon and chunk together below, after constructing that template.
    template_chunk_size=min(chunk_size, 201)
    cfg=scratch.build_arm(base,arm='legacy' if variant=='original' else 'explore',
        output=output,run_name=run_name,data_root=data_root,seed=seed,epochs=epochs,
        eval_interval=eval_interval,chunk_size=template_chunk_size,legacy_chunk_size=template_chunk_size,
        legacy_expert_chunk_size=expert_chunk_size,learning_rate=learning_rate)
    train=data_root/'dataset'/task/'train/Cus100'; val=data_root/'dataset'/task/'val/Cus100'
    cfg['dataset_name']=f'Geo-{task.upper()}-v1'
    cfg['data'].update(problem_type=task,num_customers=100,num_charging_stations=charging_stations,
        train_dataset_path=str(train),train_sample_mode='shuffle_cycle')
    cfg['env'].update(charging_mode='fixed_full',max_steps_factor=4)
    cfg['training'].update(num_envs_per_gpu=batch_per_gpu,n_traj=50,rollout_steps=horizon,
        ppo_update_epochs=5,target_kl=None,num_minibatches=4,gradient_accumulation_steps=1,
        ppo_step_chunk_size=chunk_size,checkpoint_interval=50,
        distributed_timeout_minutes=120,monitor_interval=10,
        monitor_output_dir=str(output/'monitoring'),profile_timing=True)
    cfg['offline'].update(expert_dataset_path=str(train),expert_solution_path=str(train/'expert_solutions.csv'),
        sl_expert_logprob_chunk_size=expert_chunk_size)
    cfg['advantage']['sl_expert_logprob_chunk_size']=expert_chunk_size
    if variant=='original':
        cfg['offline']['original_share_static_expert_observations']=True
    if variant=='optimized':
        cfg['training']['require_complete_feasible_rollouts']=task != 'evrptw'
        cfg['offline']['exploration_instances']=8 // world_size  # fixed global budget: 64 search trajectories/event
        cfg['model']['use_joint_graph_encoder'] = encoder_variant == 'graph'
        if encoder_variant == 'graph':
            cfg['model'].update(joint_graph_edge_dim=32, joint_graph_dropout=0.0,
                use_edge_relation_encoder=False, use_edge_value_messages=False,
                use_edge_state_updates=False)
    cfg['evaluation'].update(eval_path=str(val),gurobi_summary_path=str(val/'gurobi_summary.csv'),
        eval_max_steps=horizon,eval_n_traj=50,eval_batch_size=16,eval_save_routes=True,
        eval_before_training=True,eval_seed=17000000+seed,eval_output_dir=str(output/'evaluations'))
    for key in ('eval_limit','eval_num_batches'):
        cfg['evaluation'].pop(key,None)
    cfg['experiment_protocol'].update(phase=f'{task}100_{topology}_from_scratch',arm=variant,task=f'{task}100',
        implementation='legacy' if variant=='original' else 'explore',target_kl=None,
        seed=seed,epochs=epochs,world_size=world_size,ppo_chunk_size=chunk_size,global_instances_per_rollout=batch_per_gpu*world_size,
        global_trajectories_per_rollout=batch_per_gpu*world_size*50,
        global_instances_per_optimizer_step=batch_per_gpu*world_size//4,
        batch_controls=dict(instances=batch_per_gpu*world_size,per_rank_instances=batch_per_gpu,trajectories=50,
                            minibatches=4,ppo_passes=5,rollout_steps=horizon),
        env_action_limit=4*(101+charging_stations),physical_charging_stations=charging_stations,charging_mode='fixed_full',
        initial_evaluation_equivalence_group=None,
        evaluation=f'same independent {task} route validator; fixed isolated RNG; all 1000 validation instances, references optional per instance',
        distributed_execution=('mean of rank-local masked objectives; scaled gradient averaging at optimizer boundaries; rank-local sampler and policy memory' if world_size == 2 else 'single-process objective; one sampler and policy archive'),
        expert_coverage='Missing experts do not remove PPO instances; actual reference counts are recorded at preparation',
        input_units='native original training-set D0' if variant=='original' else 'fixed physical unit 43.638668060302734 km; distance/time/energy remain dimensionally consistent',
        original_runtime=f'f388343 model/environment/loss source plus external {world_size}-rank execution, read-only static expert-array storage sharing and evaluation adapter' if variant=='original' else None,
        failure_handling='native original reward' if variant=='original' else 'strict distance plus explicit 1000km failure guard; complete failed episodes remain in PPO, feasible-only SL',
        comparison_scope='before/after complete implementations from scratch under common sampling and update budget; not an isolated architecture ablation',
        extra_search_enabled=variant=='optimized',
        search_budget=dict(interval=5,max_instances_per_rank=8 // world_size,trajectories_per_instance=8,max_global_trajectories=64) if variant=='optimized' else None)
    cfg['experiment_protocol'].update(encoder_variant=encoder_variant,
        architecture=dict(use_joint_graph_encoder=encoder_variant == 'graph',
            joint_graph_edge_dim=32 if encoder_variant == 'graph' else None,
            joint_graph_dropout=0.0 if encoder_variant == 'graph' else None,
            training_bundle='legacy' if variant == 'original' else 'explore'))
    if encoder_variant == 'graph':
        cfg['experiment_protocol'].update(
            comparison_scope='architecture ablation against current explore: replace static graph embedding/encoder and matching latent edge projections; preserve physical input units, reward, PPO, AGDA and exploration settings',
            graph_edge_decoder='evolved joint graph edge states feed the existing resource decoder; old edge-relation encoder is disabled',
            initial_evaluation_equivalence_group=None)
    # Replace inherited VRPTW controls rather than describing them as current.
    if variant=='original':
        changed={'training.num_envs_per_gpu','training.rollout_steps','training.ppo_step_chunk_size','evaluation.eval_max_steps','evaluation.eval_batch_size'}
        cfg['experiment_protocol']['protocol_overrides']=[item for item in cfg['experiment_protocol']['protocol_overrides'] if item['parameter'] not in changed]
        cfg['experiment_protocol']['protocol_overrides'].extend([
            dict(parameter='data.problem_type',original='vrptw preset',used=task,reason='requested task; original EVRPTW environment'),
            dict(parameter='training.num_envs_per_gpu',original=128,used=batch_per_gpu,reason='declared per-rank and global rollout budget'),
            dict(parameter='training.rollout_steps',original=120,used=horizon,reason='512 covers EVRPTW native timeout; 201 covers VRPTW customer/depot routes'),
            dict(parameter='training.ppo_step_chunk_size',original=32,used=chunk_size,reason='declared task-specific time chunk; same minibatch update count'),
            dict(parameter='evaluation.eval_max_steps',original=120,used=horizon,reason='same complete-route horizon for both models'),
            dict(parameter='evaluation.eval_batch_size',original=1000,used=16,reason='common evaluation memory budget'),
            dict(parameter='execution.world_size',original=1,used=world_size,reason='external synchronous execution adapter; original model/loss bytes unchanged'),
            dict(parameter='offline.original_share_static_expert_observations',original=False,used=True,reason='share identical read-only static expert arrays during construction; retain every sample and original forward/loss; avoid duplicating static edge arrays per expert step')])
    scratch.assert_scratch(cfg)
    return cfg


def preflight_config(cfg, output):
    result=copy.deepcopy(cfg);result['run_name']+='_PREFLIGHT'
    result['training'].update(epochs=2,monitor_interval=1,post_update_kl_interval=1,monitor_output_dir=str(output/'monitoring'))
    # Exercise one full evaluation batch: reducing trajectories or batch size
    # can hide an OOM that otherwise appears at formal epoch zero.
    validation_instances=result['evaluation']['eval_batch_size']
    result['evaluation'].update(eval_before_training=False,eval_interval=2,eval_limit=validation_instances,
        eval_output_dir=str(output/'evaluations'))
    if result['offline'].get('branch_exploration_enabled'):
        result['offline']['exploration_interval']=1
    result['experiment_protocol'].update(phase=cfg['experiment_protocol']['task']+('_dual' if cfg['experiment_protocol']['world_size'] == 2 else '_single')+'_scratch_preflight',epochs=2,
        validation_instances=validation_instances,comparison_scope='full training and evaluation allocations; one validation batch; discard all resulting state')
    return result


def dataset_inputs(data_root, task="evrptw"):
    charging_stations = 20 if task == "evrptw" else 0
    inputs={}
    for split, count in [('train',5000),('val',1000)]:
        directory=data_root/'dataset'/task/split/'Cus100'
        names=['instances.pkl','metadata.json','expert_solutions.csv' if split=='train' else 'gurobi_summary.csv']
        for name in names:
            path=directory/name
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; --data-root must be AAAI_Dataset, not its dataset child')
            inputs[f'{task}/{split}/Cus100/{name}']=dict(path=str(path),sha256=shared.digest(path))
        metadata=json.loads((directory/'metadata.json').read_text())
        if any(int(metadata.get(key,0 if key=='num_charging_stations' else -1))!=value for key,value in dict(num_customers=100,num_charging_stations=charging_stations,num_instances=count).items()):
            raise ValueError(f'Unexpected split metadata: {directory}; require {count} instances of Cus100 CS{charging_stations}')
    with (data_root/'dataset'/task/'train/Cus100/expert_solutions.csv').open() as handle:
        expert_count=sum(1 for _ in csv.DictReader(handle))
    return inputs,expert_count


def prepare(args, *, config_builder=None, arm_label=None):
    gpus=shared.parse_gpus(args.gpus)
    if len(gpus) not in (1, 2):
        raise ValueError('One or two distinct GPU IDs are required per model')
    if args.single_gpu and len(gpus) != 1:
        raise ValueError('--single-gpu requires exactly one GPU ID')
    world_size = len(gpus)
    topology = 'DUAL' if world_size == 2 else 'SINGLE'
    after_runs = list(dict.fromkeys(str(path.resolve()) for path in args.after_run))
    for directory in after_runs:
        if not (Path(directory) / 'manifest.json').is_file():
            raise ValueError(f'--after-run requires an experiment directory with manifest.json: {directory}')
    dependency_state, reason = dependency_status(after_runs)
    if dependency_state == 'failed':
        raise ValueError(reason)
    hardware=shared.probe_requested_gpus(gpus)
    if not 1<=args.poll_seconds<=60 or not 2<=args.idle_checks<=10:
        raise ValueError('poll-seconds must be 1..60, idle-checks 2..10')
    data_root=args.data_root.resolve();inputs,expert_count=dataset_inputs(data_root, args.task)
    encoder_suffix = '_GRAPH' if args.encoder_variant == 'graph' else ''
    run_id=args.run_id or f'{args.task.upper()}100_{topology}_SCRATCH_{args.variant.upper()}{encoder_suffix}_S{args.seed}_E{args.epochs}_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    if Path(run_id).name!=run_id or run_id in ('.','..'):
        raise ValueError('run-id must be a fresh directory name')
    experiment=CODE_ROOT/'results/optimization'/run_id
    base=yaml.safe_load(args.base_config.resolve().read_text())
    # Validate requested settings before creating output or copying source.
    output=experiment/args.variant
    cfg=(config_builder or build_config)(base,variant=args.variant,output=output,run_name=run_id+'_'+args.variant.upper(),
        data_root=data_root,seed=args.seed,epochs=args.epochs,eval_interval=args.eval_interval,
        batch_per_gpu=args.batch_per_gpu,chunk_size=args.chunk_size,expert_chunk_size=args.expert_chunk_size,learning_rate=args.learning_rate,task=args.task,
        encoder_variant=args.encoder_variant,world_size=world_size)
    experiment.mkdir(parents=True,exist_ok=False);output.mkdir()
    frozen=experiment/'source/CaliRoute';source=shared.source_snapshot(frozen,include_initialization_assets=False)
    original=experiment/'original_source/CaliRoute';additional={}
    if args.variant=='original':
        additional['original']=dict(code_root=str(original),source=scratch.original_snapshot(original))
    spec={}
    for preflight,config,destination in [(False,cfg,output),(True,preflight_config(cfg,output/'preflight'),output/'preflight')]:
        destination.mkdir(exist_ok=True);path=destination/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
        command=[sys.executable,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={world_size}','--max-restarts=0']
        if args.variant=='original':
            command += [str(frozen/'scripts/run_original_scratch.py'),'--source-root',str(original)]
            shared.write_json(destination/'original_provenance.json',config['experiment_protocol'])
        else:
            command += ['--module','offline2online.train']
        command += ['--config',str(path),'--seed',str(args.seed),'--device','cuda']
        value=dict(config=str(path),config_sha256=shared.digest(path),output_dir=str(destination),
            code_root=str(original if args.variant=='original' else frozen),command=command,
            epochs=2 if preflight else args.epochs,validation_instances=config['evaluation']['eval_limit'] if preflight else 1000,
            required_validation_epochs=[2] if preflight else shared.validation_epochs(args.epochs,args.eval_interval),
            log_dir=str(CODE_ROOT/'results/logs'/f"Cus_100_CS_{cfg['data']['num_charging_stations']}"/config['run_name']/f'seed_{args.seed}'),
            checkpoint_dir=str(CODE_ROOT/'results/checkpoints'/f"Cus_100_CS_{cfg['data']['num_charging_stations']}"/config['run_name']/f'seed_{args.seed}'))
        if preflight:spec['preflight']=value
        else:spec.update(value)
    protocol=dict(cfg['experiment_protocol']);protocol.update(global_batch=args.batch_per_gpu*world_size,n_traj=50,
        num_minibatches=4,ppo_update_epochs=5,learning_rate=args.learning_rate,lr_schedule='constant',
        eval_interval=args.eval_interval,eval_n_traj=50,eval_batch_size=16,validation_instances=1000,
        epochs=args.epochs,expert_rows_at_prepare=expert_count,world_size_per_arm=world_size,original_commit=scratch.ORIGINAL_COMMIT,
        initial_evaluation_pairs=[],initial_evaluation_consistency_scope='different architectures need not have identical epoch-zero policy',
        best_checkpoint_caveat='Native original best uses distance among feasible cases; modern best prioritizes feasibility. Compare common epochs or choose periodic checkpoints by the same feasibility-first rule.',
        evaluation_references='Absent Gurobi references only exclude gap metrics; never exclude those validation instances.',
        gpu_preflight=dict(epochs=2,world_size=world_size,batch_per_rank=args.batch_per_gpu,n_traj=50,ppo_passes=5,
            validation_instances=cfg['evaluation']['eval_batch_size'],eval_batch_size=cfg['evaluation']['eval_batch_size'],
            eval_n_traj=cfg['evaluation']['eval_n_traj']))
    manifest=dict(created_at_utc=shared.now(),initialization_mode='scratch',init_checkpoint=None,init_checkpoint_sha256=None,
        source_init_checkpoint=None,source_init_epoch=None,code_root=str(frozen),source=source,additional_sources=additional,
        inputs=inputs,arms={arm_label or args.variant:spec},gpus=gpus,protocol=protocol,
        hardware_at_prepare=list(hardware.values()) if hardware is not None else None,poll_seconds=args.poll_seconds,idle_checks=args.idle_checks,
        after_runs=after_runs)
    shared.write_json(experiment/'manifest.json',manifest)
    shared.write_json(experiment/'status.json',dict(state='prepared',arms={arm_label or args.variant:dict(state='prepared')}))
    shared.verify_manifest(manifest)
    if args.launch:
        with (experiment/'supervisor.log').open('a',buffering=1) as log:
            process=subprocess.Popen([sys.executable,'-B','-u',str(frozen/'scripts/run_evrptw_dual_scratch.py'),'--supervise',str(experiment)],
                cwd=frozen,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (experiment/'supervisor.pid').write_text(str(process.pid)+'\n')
    print(experiment,flush=True)
    return experiment


def lock_gpu_pair(cards,gpus):
    locks=[]
    try:
        # A consistent acquisition order avoids two partially held pairs.
        for gpu in sorted(gpus):
            lock=shared.acquire_gpu_lock(cards[gpu]['uuid'])
            if lock is None:
                for held in locks:held.close()
                return None
            locks.append(lock)
        return locks
    except BaseException:
        for lock in locks:lock.close()
        raise


def supervise(experiment):
    manifest=json.loads((experiment/'manifest.json').read_text());arm,formal=next(iter(manifest['arms'].items()))
    supervisor_lock=(experiment/'supervisor.lock').open('a+')
    fcntl.flock(supervisor_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if json.loads((experiment/'status.json').read_text())['state']!='prepared':
        raise ValueError('Only a fresh prepared experiment can start; no implicit checkpoint resume')
    try:
        shared.verify_manifest(manifest)
    except Exception as error:
        shared.write_json(experiment/'status.json',dict(state='failed',error=f'Input verification: {error}',finished_at_utc=shared.now()))
        supervisor_lock.close()
        raise
    detail=dict(state='queued',target_epochs=formal['epochs'],completed_training_epochs=0,gpu=','.join(map(str,manifest['gpus'])))
    status=dict(state='waiting_gpu',started_at_utc=shared.now(),supervisor_pid=os.getpid(),arms={arm:detail})
    stopped=False;process=None;stream=None;locks=[];idle=0
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def snapshot():
        status['updated_at_utc']=shared.now();shared.write_json(experiment/'status.json',status)
        shared.write_json(experiment/'comparison.json',shared.comparison_report(manifest,status))
    def spawn(stage):
        nonlocal process,stream
        spec=formal['preflight'] if stage=='preflight' else formal
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,manifest['gpus'])),
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(Path(spec['output_dir'])/'numba_cache'))
        for key in ('PYTHONPATH','EVRPTW_DB_ROOT'):env.pop(key,None)
        stream=(Path(spec['output_dir'])/'console.log').open('a',buffering=1)
        process=subprocess.Popen(spec['command'],cwd=spec['code_root'],env=env,stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        detail.update(state='running',stage=stage,pid=process.pid,stage_started_at_utc=shared.now())
        status['state']='running'
        print(shared.now(),arm,stage,'GPUs',detail['gpu'],'pid',process.pid,flush=True)
    try:
        while True:
            dependency_state, reason = dependency_status(manifest.get('after_runs', []))
            if process is None and dependency_state == 'failed':
                detail.update(state='failed', error=reason)
                status.update(state='failed', error=reason, finished_at_utc=shared.now())
                snapshot()
                break
            if process is None and dependency_state == 'waiting':
                status.update(state='waiting_dependency', dependency_wait_reason=reason)
                idle=0
            else:
                status.pop('dependency_wait_reason', None)
                if process is None:
                    status['state']='waiting_gpu'
            cards=None
            try:
                cards=shared.gpu_snapshot();shared.validate_requested_gpus(manifest['gpus'],cards)
                status.pop('gpu_poll_warning',None)
                with (experiment/'hardware.jsonl').open('a') as h:
                    h.write(json.dumps(dict(time_utc=shared.now(),gpus=list(cards.values()),stage=detail.get('stage')))+'\n')
            except (OSError,subprocess.SubprocessError,ValueError,KeyError,IndexError) as error:
                cards=None;idle=0;status['gpu_poll_warning']=f'{type(error).__name__}: {error}'
            if process is None and not stopped and cards is not None and dependency_state == 'completed':
                available=all(shared.idle_gpu(cards[g]) for g in manifest['gpus']);idle=idle+1 if available else 0
                status['consecutive_idle_pair_checks']=idle
                if idle>=manifest['idle_checks']:
                    acquired=lock_gpu_pair(cards,manifest['gpus'])
                    if acquired is not None:
                        locks=acquired  # owns all requested GPU locks even if the recheck raises
                        rechecked=shared.gpu_snapshot()
                        if all(shared.idle_gpu(rechecked[g]) for g in manifest['gpus']):
                            spawn('preflight')
                        else:
                            for lock in acquired:lock.close()
                            locks=[]
            if process is not None:
                stage=detail['stage'];spec=formal['preflight'] if stage=='preflight' else formal
                progress=detail.setdefault('preflight_progress',{}) if stage=='preflight' else detail
                shared.refresh_progress(spec,progress)
                if stopped and process.poll() is None:shared.terminate_group(process)
                code=process.poll()
                if code is not None:
                    shared.refresh_progress(spec,progress,finished=(code==0))
                    success=code==0 and shared.successful_training(spec,progress)
                    progress.update(exit_code=code,finished_at_utc=shared.now());stream.close();stream=None
                    if stage=='preflight' and success and not stopped:
                        if manifest['protocol'].get('require_preflight_health'):
                            from run_graph_reproduction_2080ti import validate_preflight
                            progress['numerical_health'] = validate_preflight(spec, len(manifest['gpus']))
                        progress['state']='completed';spawn('training')
                    else:
                        detail['state']='interrupted' if stopped else ('completed' if success else 'failed')
                        if not success and not stopped:detail['error']=f'{stage} exit={code}; require complete training and full scheduled validation'
                        status['state']=detail['state'];status['finished_at_utc']=shared.now();snapshot();break
            elif stopped:
                detail['state']=status['state']='interrupted';status['finished_at_utc']=shared.now();snapshot();break
            snapshot();time.sleep(manifest['poll_seconds'])
    except BaseException as error:
        if process is not None and process.poll() is None:shared.terminate_group(process)
        status.update(state='failed',error=f'{type(error).__name__}: {error}',finished_at_utc=shared.now())
        detail.update(state='failed',error=status['error']);snapshot();traceback.print_exc();raise
    finally:
        # A failed supervisor must keep its GPU locks until torchrun and the
        # process group have exited, including SIGKILL escalation if needed.
        while process is not None and process.poll() is None:
            shared.terminate_group(process)
            time.sleep(.25)
        if stream:stream.close()
        for lock in locks:lock.close()
        supervisor_lock.close()


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('evrptw','vrptw'),default='evrptw')
    parser.add_argument('--variant',choices=('original','optimized'))
    parser.add_argument('--encoder-variant', choices=('current', 'graph'), default='current',
                        help='graph replaces the optimized static encoder; current preserves the existing experiment')
    parser.add_argument('--supervise',type=Path)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--single-gpu',action='store_true',help='Require exactly one GPU ID for this task')
    parser.add_argument('--seed',type=int,default=3010)
    parser.add_argument('--epochs',type=int,default=1500)
    parser.add_argument('--eval-interval',type=int,default=50)
    parser.add_argument('--batch-per-gpu',type=int,default=32)
    parser.add_argument('--chunk-size',type=int,default=8)
    parser.add_argument('--expert-chunk-size',type=int,default=64)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--base-config',type=Path,default=CODE_ROOT/'configs/experiments/physics_exploration_vrptw100.yaml')
    parser.add_argument('--data-root',type=Path,default=default_data_root())
    parser.add_argument('--run-id')
    parser.add_argument('--after-run',type=Path,action='append',default=[],metavar='EXPERIMENT_DIR',
                        help='Wait for this experiment to complete successfully before reserving GPUs; repeat for multiple prerequisites')
    parser.add_argument('--poll-seconds',type=int,default=10)
    parser.add_argument('--idle-checks',type=int,default=3)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--launch',action='store_true')
    return parser


def main():
    parser=make_parser();args=parser.parse_args()
    if args.supervise:supervise(args.supervise.resolve())
    elif args.variant:prepare(args)
    else:parser.error('--variant is required when preparing or launching')


if __name__=='__main__':main()
