"""First E2 batch: hardware assignment, dependencies and explicit readiness.

This is preparation metadata, not a substitute for native/PPO host adapters.
No E1 launcher is used as a fallback for an unfinished E2 integration.
"""

PROFILES = ('2080ti_a', 'a6000', '2080ti_3')


def job(name, problem, backbone, panel, objective, modules=(), after=(), baseline=None):
    return dict(id=name, problem=problem, customers=100, backbone=backbone,
                panel=panel, objective=objective, modules=list(modules),
                after=list(after), baseline=baseline, initialization='scratch',
                training_seed=3009, eval_K=50, test_count=1000,
                world_size=1, self_replay=False, incumbent_storage='cost_only',
                status='adapter_integration_pending', training_entrypoint=None,
                post_training='validation-selected best checkpoint -> independent test -> E2 pair table')


def server_plan(server, baseline_policy='pending'):
    if server not in PROFILES:
        raise ValueError(f'Unknown E2 server profile: {server}')
    if baseline_policy not in ('pending', 'reuse', 'rerun'):
        raise ValueError('baseline_policy must be pending, reuse or rerun')
    if server == '2080ti_a':
        queues = {
            '0': [job('pomo_cvrp100_base','cvrp','pomo','A','native_pomo_reinforce')],
            '1': [job('pomo_cvrp100_rdi','cvrp','pomo','A','native_pomo_reinforce',('RDI',),baseline='pomo_cvrp100_base')],
            '2': [job('radar_cvrp100_agda','cvrp','radar','A','native_pomo_reinforce',('AGDA',),baseline='radar_cvrp100_base')],
            '3': []}
        if baseline_policy == 'rerun':
            queues['3']=[job('radar_cvrp100_base','cvrp','radar','A','native_pomo_reinforce')]
    elif server == 'a6000':
        queues={
            '0': [job('rrnco_vrptw100_ppo','vrptw','rrnco','B','step_ppo'),
                  job('rrnco_vrptw100_awbc','vrptw','rrnco','B','step_ppo_awbc',after=('rrnco_vrptw100_ppo',),baseline='rrnco_vrptw100_ppo')],
            '1': [job('rrnco_vrptw100_slppo','vrptw','rrnco','B','step_ppo_slppo',baseline='rrnco_vrptw100_ppo'),
                  job('rrnco_vrptw100_dapg','vrptw','rrnco','B','step_ppo_dapg',after=('rrnco_vrptw100_slppo',),baseline='rrnco_vrptw100_ppo')]}
    else:
        queues={
            '0': [job('evrptw_rl_evrptw100_base','evrptw','evrptw_rl_reimplementation','A','native_policy_gradient')],
            '1': [job('evrptw_rl_evrptw100_agda','evrptw','evrptw_rl_reimplementation','A','native_policy_gradient',('AGDA',),baseline='evrptw_rl_evrptw100_base')],
            '2': [job('rrnco_vrptw100_agda','vrptw','rrnco','A','native_pomo_reinforce',('AGDA',),baseline='rrnco_vrptw100_native_base')]}
        if baseline_policy == 'rerun':
            queues['2'].insert(0,job('rrnco_vrptw100_native_base','vrptw','rrnco','A','native_pomo_reinforce'))
            queues['2'][1]['after']=['rrnco_vrptw100_native_base']
    for gpu, tasks in queues.items():
        for task in tasks:
            task['physical_gpu']=int(gpu)
            task['CUDA_VISIBLE_DEVICES']=gpu
            task['logical_device']='cuda:0'
            task['training_budget']=dict(status='needs_frozen_native_protocol',instance_exposures=96000,batch_size=64,n_traj=50,expected_updates=1517,learning_rate=4e-4) if task['backbone'] in ('radar','rrnco') and task['panel']=='A' else dict(status='pending_matched_pair_protocol')
    return dict(schema='e2_server_plan_v1',server=server,baseline_policy=baseline_policy,
                baseline_reuse_approved=False,
                reuse_note='reuse is a request, not acceptance; matching archive/hash audit still required',
                campaign=f'E2_CUS100_S3009_V1_{server}',queues=queues,
                queue_completion='Next task starts only after training and accepted best-checkpoint test of predecessor',
                hardware_policy='Single GPU per job. Keep effective batches/exposure fixed within comparisons; tune execution chunks only.',
                reserved_server='2080ti_b remains on E1 EVRPTW100 AWBC/DAPG',
                launchable=False)


def blockers(plan):
    reasons=[]
    for tasks in plan['queues'].values():
        for task in tasks:
            if task['training_entrypoint'] is None:
                reasons.append(f"{task['id']}: end-to-end E2 trainer/adapter and real-instance acceptance are not implemented")
    if plan['server']!='a6000' and plan['baseline_policy']!='rerun':
        reasons.append('E1 native baseline archive and matching-protocol audit pending')
    if plan['server']=='a6000':
        reasons.append('RRNCO step-level PPO recurrent/static replay checks and quartet budget are pending; native E1 REINFORCE is not a PPO substitute')
    return reasons
