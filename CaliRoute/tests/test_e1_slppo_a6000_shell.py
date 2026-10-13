"""A6000 SL-PPO queue must select only SL-PPO and preserve statistical budgets."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from e1.configs import build_config

ROOT=Path(__file__).resolve().parents[1]


def fixture(tmp_path,fail=False):
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ('run_e1_slppo_a6000.sh','e1_small_scale_hardware.sh'):
        shutil.copy2(ROOT/'scripts'/name,scripts/name)
    (scripts/'run_e1.sh').write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$E1_TEST_CALLS"
if [[ "${E1_TEST_FAIL:-}" == yes && "$1" == smoke && "$*" == *CVRP50* ]]; then exit 7; fi
''')
    env=dict(os.environ,E1_TEST_CALLS=str(tmp_path/'calls'),E1_A6000_SLPPO_ROOT=str(tmp_path/'results'),
        E1_PYTHON='/usr/bin/python3',E1_TEST_FAIL='yes' if fail else 'no')
    for key in list(env):
        if key.startswith('E1_CVRP') or key in ('E1_BATCH2_ASSETS_ROOT','E1_A6000_SLPPO_SCALES','E1_SLPPO_PPO_CHUNK','E1_EXPERT_CHUNK'):
            env.pop(key)
    return scripts/'run_e1_slppo_a6000.sh',env


def test_two_independent_scales_explicit_method_gpu_and_protocol(tmp_path):
    script,env=fixture(tmp_path)
    subprocess.run(['bash',str(script),'--worker'],env=env,check=True,capture_output=True,text=True)
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len([x for x in calls if x.startswith('launch ')])==2
    for scale,gpu in ((50,0),(15,1)):
        commands=[x for x in calls if f'CVRP{scale}' in x or f'--customers {scale} ' in x]
        prepared=next(x for x in commands if x.startswith('prepare '))
        assert '--seed 3009 --epochs 1500' in prepared
        assert '--method-ppo-chunk slppo=32 --expert-chunk 128' in prepared
        launched=next(x for x in commands if x.startswith('launch '))
        assert f'--server a6000 --method slppo --gpus {gpu} --execute' in launched
        assert any(x.startswith('smoke ') and '--method slppo' in x for x in commands)
    assert not any(x in '\n'.join(calls) for x in ('--after-campaign','native-status','--batch-size','--n-traj','--checkpoint'))


def test_failed_scale_does_not_launch_or_block_other_scale(tmp_path):
    script,env=fixture(tmp_path,True)
    result=subprocess.run(['bash',str(script),'--worker'],env=env,capture_output=True,text=True)
    assert result.returncode!=0
    launched=[x for x in (tmp_path/'calls').read_text().splitlines() if x.startswith('launch ')]
    assert len(launched)==1 and 'CVRP15' in launched[0] and '--gpus 1' in launched[0]


def test_preview_is_read_only_and_rejects_shared_gpu(tmp_path):
    script,env=fixture(tmp_path)
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,check=True)
    assert 'GPU0: CVRP50 SL-PPO' in result.stdout and 'GPU1: CVRP15 SL-PPO' in result.stdout
    assert not (tmp_path/'results').exists() and not (tmp_path/'calls').exists()
    result=subprocess.run(['bash',str(script)],env=dict(env,E1_CVRP15_SLPPO_GPU='0'),capture_output=True,text=True)
    assert result.returncode==2


@pytest.mark.parametrize('n',[15,50])
def test_a6000_chunks_do_not_change_slppo_training_budget(tmp_path,n):
    kwargs=dict(method='slppo',data_root=tmp_path,output_dir=tmp_path/'slppo',expert_pool=tmp_path/'expert.csv',
        distance_unit_km=10.,world_size=1,customers=n)
    before=build_config(**kwargs,ppo_chunk=8,expert_chunk=32)
    after=build_config(**kwargs,ppo_chunk=32,expert_chunk=128)
    assert after['training'].pop('ppo_step_chunk_size')==32
    before['training'].pop('ppo_step_chunk_size')
    assert after['advantage'].pop('sl_expert_logprob_chunk_size')==128
    before['advantage'].pop('sl_expert_logprob_chunk_size')
    assert after['offline'].pop('sl_expert_logprob_chunk_size')==128
    before['offline'].pop('sl_expert_logprob_chunk_size')
    assert before==after
