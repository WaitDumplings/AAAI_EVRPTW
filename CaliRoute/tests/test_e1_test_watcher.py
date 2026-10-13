import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('watch_e1_tests', Path(__file__).parents[1] / 'scripts/watch_e1_tests.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_waits_for_scheduler_exit_even_when_rank_zero_finished(tmp_path):
    job = dict(scheduler=str(tmp_path / 'scheduler'), method='ppo_base')
    spec = dict(run_dir=str(tmp_path / 'run'))
    assert w.readiness(job, spec) == 'waiting_for_scheduler'
    dump(tmp_path / 'run/training_result.json', {'state':'completed'})
    dump(tmp_path / 'scheduler/status.json', {'methods':{'ppo_base':{'state':'training'}}})
    assert w.readiness(job, spec) == 'waiting_for_training'
    dump(tmp_path / 'scheduler/status.json', {'methods':{'ppo_base':{'state':'failed'}}})
    assert w.readiness(job, spec) == 'blocked_training_failed'
    dump(tmp_path / 'scheduler/status.json', {'methods':{'ppo_base':{'state':'completed'}}})
    assert w.readiness(job, spec) == 'ready'


def test_report_merge_preserves_other_tasks_remote_rows_and_unknown_gaps(tmp_path):
    report = tmp_path / 'report'; report.mkdir()
    fields = ['method','status','feasible_count','test_count','all_feasible_mean_km','common_mean_km','common_gap_to_gurobi_pct','runtime_s_per_instance','training_hours']
    base = dict.fromkeys(fields, 'TBD'); base.update(method='PPO (base)',test_count='1000')
    remote = dict(base, method='AWBC',status='remote',all_feasible_mean_km='106.7609')
    w.csv_write(report / 'main_results_partial.csv', fields, [base,remote])
    overview = tmp_path / 'overview.csv'
    foreign = dict(base,problem='CVRP',customers='100',all_feasible_mean_km='300')
    w.csv_write(overview, ['problem','customers']+fields, [foreign,dict(base,problem='VRPTW',customers='100'),dict(remote,problem='VRPTW',customers='100')])
    dump(report / 'evidence.json', {'remote_updates':[{'source':'user'}]})
    job = dict(label='PPO (base)',problem='VRPTW',customers=100,report_dir=str(report))
    summary = dict(feasible_count=999,instances=1000,mean_distance_km=222.5,mean_runtime_s=.12)
    plan = dict(overview=str(overview))
    for _ in range(2):
        w.merge_report(plan, job, summary, {'checkpoint_epoch':1450})
    rows = w.csv_read(overview)[1]
    assert rows[0] == foreign
    assert rows[1]['all_feasible_mean_km']=='222.5'
    assert rows[1]['feasible_count']=='999'
    assert rows[1]['common_gap_to_gurobi_pct']=='TBD'
    assert rows[1]['training_hours']=='TBD'
    assert rows[2]['all_feasible_mean_km']=='106.7609'
    assert len(w.csv_read(report / 'test_results_partial.csv')[1])==1
    assert json.loads((report/'evidence.json').read_text())['remote_updates']==[{'source':'user'}]


def test_partial_test_is_blocked_without_launch_or_table_update(tmp_path, monkeypatch):
    from e1 import runner
    class Runtime:
        def gpu_snapshot(self): return {}
    monkeypatch.setattr(runner, '_runtime', lambda: Runtime())
    run = tmp_path / 'run'; (run/'test').mkdir(parents=True)
    monkeypatch.setattr(w, 'job_info', lambda job: ({},{'run_dir':str(run)},{},tmp_path/'best.pt'))
    monkeypatch.setattr(w, 'readiness', lambda *args: 'ready')
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Must not launch')))
    status = {'jobs':{}}
    w.poll({'jobs':[{'id':'one','method':'ppo_base'}]},status,tmp_path,execute=True)
    assert status['jobs']['one']['state']=='blocked'
    assert 'Partial test' in status['jobs']['one']['error']
