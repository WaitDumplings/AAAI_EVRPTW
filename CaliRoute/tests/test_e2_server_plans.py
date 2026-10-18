import json
from pathlib import Path
import subprocess
import sys
from e2.server_plans import PROFILES, server_plan, blockers

ROOT=Path(__file__).resolve().parents[1]


def test_ten_jobs_no_e1_b_and_single_gpu_each():
    plans=[server_plan(p) for p in PROFILES]
    tasks=[t for p in plans for q in p['queues'].values() for t in q]
    assert len(tasks)==10 and len({t['id'] for t in tasks})==10
    assert all(t['world_size']==1 and t['customers']==100 and t['training_seed']==3009 for t in tasks)
    assert {p['server'] for p in plans}=={'2080ti_a','2080ti_3','a6000'}
    assert all(blockers(p) and not p['launchable'] for p in plans)


def test_quartet_and_queued_dependencies():
    p=server_plan('a6000')
    assert [t['objective'] for t in p['queues']['0']]==['step_ppo','step_ppo_awbc']
    assert [t['objective'] for t in p['queues']['1']]==['step_ppo_slppo','step_ppo_dapg']
    for queue in p['queues'].values():
        assert queue[1]['after']==[queue[0]['id']]
        assert all(t['modules']==[] and not t['self_replay'] for t in queue)


def test_conditional_native_baselines_and_reuse_not_approval():
    assert server_plan('2080ti_a')['queues']['3']==[]
    assert server_plan('2080ti_a','rerun')['queues']['3'][0]['id']=='radar_cvrp100_base'
    q=server_plan('2080ti_3','rerun')['queues']['2']
    assert len(q)==2 and q[1]['after']==[q[0]['id']]
    assert not server_plan('2080ti_a','reuse')['baseline_reuse_approved']


def test_prepare_idempotent_but_cannot_claim_training(tmp_path):
    cmd=[sys.executable,str(ROOT/'scripts/run_e2_batch.py'),'a6000','--output',str(tmp_path),'--prepare']
    for _ in range(2):subprocess.run(cmd,check=True,capture_output=True,text=True)
    p=tmp_path/'E2_CUS100_S3009_V1_a6000'
    status=json.loads((p/'status.json').read_text())
    assert status['state']=='prepared_not_training' and not status['formal_training_started']
    assert len(list((p/'runs').glob('*/planned_run.json')))==4
    attempt=subprocess.run(cmd[:-1]+['--execute'],capture_output=True,text=True)
    assert attempt.returncode==2 and 'No training process started' in attempt.stderr


def test_shell_profiles_run_from_other_directory(tmp_path):
    for profile in PROFILES:
        result=subprocess.run(['bash',str(ROOT/f'scripts/run_e2_{profile}.sh')],cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert f'Server: {profile}' in result.stdout and 'Preview only' in result.stdout
