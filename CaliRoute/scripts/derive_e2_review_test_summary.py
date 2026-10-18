#!/usr/bin/env python3
"""Derive a clearly labelled summary from an accepted original E1 test record."""
import argparse,datetime,hashlib,json,math
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def derive(directory):
 manifest=directory/'manifest.json';instances=directory/'instances.jsonl'
 saved=json.loads(manifest.read_text());rows=[json.loads(x) for x in instances.read_text().splitlines()]
 assert saved['accepted'] and digest(instances)==saved['instances_sha256']
 result=saved['result'];costs=[r['recomputed_cost_km'] for r in rows if r['feasible']]
 mean=sum(costs)/len(costs)
 # Read the original summary's metric key; refuse to guess if it changes.
 keys=[k for k in result if 'mean' in k and ('cost' in k or 'distance' in k)]
 matching=[k for k in keys if isinstance(result[k],(int,float)) and math.isclose(result[k],mean,rel_tol=1e-12)]
 assert len(rows)==1000 and len({r['instance_id'] for r in rows})==1000 and matching,(keys,mean)
 assert all(r['split']=='test' and r['actual_K']==50 for r in rows)
 summary=dict(result)
 summary['_archive_provenance']=dict(not_an_original_run_file=True,
  derived_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  source='manifest.json.result, rechecked against original instances.jsonl',
  manifest_sha256=digest(manifest),instances_jsonl_sha256=digest(instances),
  generator='scripts/derive_e2_review_test_summary.py',generator_sha256=digest(Path(__file__)),
  checks=dict(unique_instances=1000,all_test_K50=True,feasible=len(costs),mean_recomputed_cost_km=mean,matching_original_mean_keys=matching))
 (directory/'instances.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary['_archive_provenance']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();print(json.dumps(derive(args.directory),indent=2))
