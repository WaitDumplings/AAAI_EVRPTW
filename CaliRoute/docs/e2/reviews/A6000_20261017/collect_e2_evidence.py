import csv,json,hashlib,subprocess,difflib,shutil,collections
from pathlib import Path
R=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute');O=R/'results/e2/exports/RADAR_RRNCO_E2_review'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
rows=[];allstats={};diffs=[]
for method,campaign in [('radar','results/e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1'),('rrnco','results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2')]:
 base=R/campaign;run=base/'runs'/method;manifest=read(run/'native_manifest.json');stats={}
 for kind in ['source','harness']:
  for item in manifest[kind]['files']:
   rel=Path(item['path']);frozen=run/('native_'+kind)/rel
   current=(R/'benchmarks/native_sources'/method/rel) if kind=='source' else (R/rel)
   frozenhash=sha(frozen);assert frozenhash==item['sha256']
   gitpath=str(current.relative_to(R.parent))
   result=subprocess.run(['git','show',f'HEAD:{gitpath}'],cwd=R,capture_output=True)
   remotehash=hashlib.sha256(result.stdout).hexdigest() if result.returncode==0 else None
   currenthash=sha(current) if current.exists() else None
   rows.append(dict(method=method,kind=kind,path=str(rel),frozen_sha256=frozenhash,head_sha256=remotehash,worktree_sha256=currenthash,frozen_equals_head=frozenhash==remotehash,frozen_equals_worktree=frozenhash==currenthash))
   if kind=='harness' and frozenhash!=remotehash and rel.suffix=='.py':
    diffs.extend(difflib.unified_diff(result.stdout.decode().splitlines(True),frozen.read_text().splitlines(True),fromfile='a/'+gitpath,tofile='b/'+gitpath))
 (O/'patches'/f'{method}_frozen_harness_vs_head.patch').write_text(''.join(diffs));diffs=[]
 train=[json.loads(x) for x in (run/'native_train.jsonl').read_text().splitlines()]
 stats.update(counters=read(run/'native_status.json')['counters'],batch_histogram=dict(collections.Counter(x['batch_instances'] for x in train)),lr_values=sorted(set(x['lr'] for x in train)),train_log_sha256=sha(run/'native_train.jsonl'))
 assert sum(x['batch_instances'] for x in train)==96000 and len(train)==1517
 # decision_steps in each row gets replaced by cumulative counters in worker;
 # take last counter, never sum the cumulative decision_steps column.
 assert train[-1]['decision_steps']==stats['counters']['decision_steps']
 with (O/f'{method}_batch_sequence.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['optimizer_updates','batch_instances','instance_exposures','decision_steps','lr','timestamp']);w.writeheader();w.writerows({k:x[k] for k in w.fieldnames} for x in train)
 stats['datasets']={}
 for split in ['train','val','test']:
  path=run/'native_data'/f'{split}.json';meta=read(path);npz=path.with_suffix('.npz')
  assert sha(npz)==meta['npz_sha256']
  raw=Path(meta['source_path']);assert raw.exists() and sha(raw)==meta['source_sha256']
  identities=[x['instance_id'] for x in meta['instances']]
  identitypath=O/'identities'/f'{method}_{split}.txt';identitypath.parent.mkdir(exist_ok=True);identitypath.write_text('\n'.join(identities)+'\n')
  stats['datasets'][split]={k:v for k,v in meta.items() if k!='instances'}
  stats['datasets'][split].update(metadata_sha256=sha(path),id_list_sha256=sha(identitypath),hashes_verified_now=True)
 for name in ['native_resolved.json','native_architecture.json','native_status.json','native_manifest.json']:
  dest=O/'baseline_metadata'/method/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/name,dest)
 stats['test_manifest']=read(run/'test/manifest.json')
 stats['test_jsonl_sha256']=sha(run/'test/instances.jsonl')
 stats['current_source_diff_count']=sum(not x['frozen_equals_worktree'] for x in rows if x['method']==method and x['kind']=='source')
 stats['head_source_diff_count']=sum(not x['frozen_equals_head'] for x in rows if x['method']==method and x['kind']=='source')
 allstats[method]=stats
with (O/'source_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(O/'baseline_actual_evidence.json',allstats)
print('source rows',len(rows),'mismatches',[(x['method'],x['kind'],x['path']) for x in rows if not x['frozen_equals_worktree']])
print('decision steps',{m:x['counters']['decision_steps'] for m,x in allstats.items()})
# Preserve the original report + exact summary derivation generator.
shutil.copy2('/tmp/build_e1_native_archive.py',O/'derive_e1_archive.py')
shutil.copy2(R/'results/e1/exports/E1_native_baselines_20261017/E1_native_baseline_audit.md',O/'E1_native_baseline_audit.md')
