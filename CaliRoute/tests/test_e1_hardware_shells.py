"""Launcher argument forwarding must preserve per-method execution choices."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from e1.runner import _method_chunk_overrides, parser
from e1.configs import build_config

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('bad',[['rrnco=32'],['ppo_base=0'],['ppo_base=-2'],['ppo_base=3.5'],['ppo_base=8','ppo_base=32']])
def test_reject_invalid_chunks(bad):
    with pytest.raises(ValueError): _method_chunk_overrides(bad)


def test_chunk_override_preserves_training_protocol(tmp_path):
    args=parser().parse_args(['prepare','--audit','a','--campaign','c','--method-ppo-chunk','ppo_base=32','--method-ppo-chunk','ppo_rdi_agda=14'])
    choices=_method_chunk_overrides(args.method_ppo_chunk)
    for scale in [15,50]:
        for method in choices:
            kwargs=dict(data_root=tmp_path,output_dir=tmp_path/'out',distance_unit_km=10.,customers=scale,world_size=1)
            old=build_config(method,ppo_chunk=8,**kwargs)
            new=build_config(method,ppo_chunk=choices[method],**kwargs)
            assert new['training'].pop('ppo_step_chunk_size')==choices[method]
            old['training'].pop('ppo_step_chunk_size')
            assert new==old

@pytest.mark.parametrize('script,scale',[('run_e1_cvrp50.sh','50'),('run_e1_cvrp15.sh','15')])
def test_shell_passes_chunks_to_prepare_without_budget_overrides(tmp_path,script,scale):
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ['run_e1_cvrp50.sh','run_e1_cvrp15.sh','run_e1_batch2.sh','e1_small_scale_hardware.sh']:
        shutil.copy2(ROOT/'scripts'/name,scripts/name)
    (scripts/'run_e1.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$E1_TEST_CALLS"\n')
    calls=tmp_path/'calls'
    env=dict(os.environ,E1_TEST_CALLS=str(calls),E1_BATCH2_ROOT=str(tmp_path/'results'),E1_PYTHON='/usr/bin/python3',E1_BASE_PPO_CHUNK='32',E1_RDI_AGDA_PPO_CHUNK='14',E1_AWBC_PPO_CHUNK='12')
    subprocess.run(['bash',str(scripts/script),'2080ti_b','--execute'],env=env,check=True,capture_output=True,text=True)
    commands=calls.read_text().splitlines()
    prepared=[s for s in commands if s.startswith('prepare ')]
    assert len(prepared)==1 and f'--customers {scale}' in prepared[0]
    for setting in ['ppo_base=32','ppo_rdi_agda=14','awbc=12','dapg=8','slppo=8']:
        assert f'--method-ppo-chunk {setting}' in prepared[0]
    assert '--expert-chunk 32' in prepared[0]
    assert not any(x in prepared[0] for x in ['--batch','--n-traj','--ppo-passes'])
    if scale=='15': assert '--after-campaign' not in '\n'.join(commands)
