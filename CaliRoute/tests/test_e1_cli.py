"""Relative user paths survive the switch into a frozen campaign checkout."""
from argparse import Namespace
from pathlib import Path
import json
import sys
import pytest
from e1 import runner


@pytest.mark.parametrize('equals',[False,True])
def test_frozen_reexecution_resolves_relative_campaign_and_checkpoint(tmp_path,monkeypatch,equals):
    campaign=tmp_path/'results/e1/campaign';source=campaign/'source/CaliRoute';source.mkdir(parents=True)
    (campaign/'manifest.json').write_text(json.dumps({'code_root':str(source)}))
    monkeypatch.chdir(tmp_path)
    values=['resume','--campaign=results/e1/campaign','--checkpoint=backups/last.pt'] if equals else ['resume','--campaign','results/e1/campaign','--checkpoint','backups/last.pt']
    monkeypatch.setattr(sys,'argv',['run_e1.py',*values,'--method','slppo'])
    calls=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda command,**kwargs:calls.append((command,kwargs)))
    runner.main()
    command,kwargs=calls[0]
    assert kwargs['cwd']==source
    assert str(campaign) in ' '.join(command)
    assert str(tmp_path/'backups/last.pt') in ' '.join(command)
    assert kwargs['env']['PYTHONPATH']==str(source)


def test_summary_does_not_reexecute_remote_recorded_source(tmp_path,monkeypatch):
    campaign=tmp_path/'copied';campaign.mkdir()
    (campaign/'manifest.json').write_text(json.dumps({'code_root':'/missing/remote/source'}))
    monkeypatch.setattr(sys,'argv',['run_e1.py','summarize','--campaign',str(campaign)])
    calls=[];monkeypatch.setattr(runner,'summarize',lambda args:calls.append(args.campaign))
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Summary must support relocated campaigns'))
    runner.main();assert calls==[campaign]
