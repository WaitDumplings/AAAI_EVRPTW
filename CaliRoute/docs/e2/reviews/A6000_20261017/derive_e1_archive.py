from pathlib import Path
import json, hashlib, math, statistics, shutil, tarfile, collections
from datetime import datetime, timezone
import torch

ROOT=Path('/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute')
OUT=ROOT/'results/e1/exports/E1_native_baselines_20261017'
SPECS=[('RADAR_CVRP100','e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1','radar'),
       ('RRNCO_VRPTW100','e1/E1_VRPTW100_NATIVE_S3009_A6000_V2','rrnco')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def verified_copy(src,dst,expected=None):
    actual=sha(src)
    if expected is not None:assert actual==expected,(str(src),actual,expected)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    assert sha(dst)==actual
    return actual

all_evidence=[]
for label,rel,method in SPECS:
    p=ROOT/'results'/rel;q=p/'runs'/method
    package=OUT/label;target=package/'CaliRoute/results'/rel
    m=read(p/'manifest.json');native=read(q/'native_manifest.json');cfg=read(q/'native_resolved.json')
    test=read(q/'test/manifest.json');status=read(q/'native_status.json')
    train=lines(q/'native_train.jsonl');vals=lines(q/'native_validation.jsonl');rows=lines(q/'test/instances.jsonl')
    assert test['accepted'] and test['formal_training_complete']
    assert sha(q/'test/instances.jsonl')==test['instances_sha256']
    assert len(rows)==len({r['instance_id'] for r in rows})==1000
    assert all(r['split']=='test' and r['training_seed']==3009 and r['actual_K']==r['requested_K']==50 and r['checkpoint_id']==test['checkpoint_sha256'] for r in rows)
    good=[r for r in rows if r['feasible']]
    assert len(good)==test['result']['feasible_count']
    assert math.isclose(statistics.mean(r['recomputed_cost_km'] for r in good),test['result']['mean_distance_km'],abs_tol=1e-9)
    assert math.isclose(statistics.mean(r['runtime'] for r in rows),test['result']['mean_runtime_s'],abs_tol=1e-9)
    assert sum(r['batch_instances'] for r in train)==96000
    assert sum(r['completed_rollouts'] for r in train)==4800000
    assert len(train)==1517 and train[-1]['optimizer_updates']==1517
    assert len(vals)==30
    best_val=min(vals,key=lambda x:(-x['feasible_count'],x['mean_distance_km']))
    selected={}
    for name in ('best','last'):
        f=q/f'native_checkpoints/{name}.pt';c=torch.load(f,map_location='cpu',weights_only=False)
        selected[name]={k:c[k] for k in ('schema','method','counters','best_metric','config_sha256','source_sha256','harness_sha256','dataset_hashes','architecture','runtime_versions','saved_at') if k in c}
        selected[name].update(file_sha256=sha(f),file_bytes=f.stat().st_size,
            optimizer_param_groups=[{k:v for k,v in g.items() if k!='params'} for g in c['optimizer']['param_groups']],scheduler=c['scheduler'])
        assert c['config_sha256']==native['config_sha256']
        assert c['source_sha256']==native['source']['sha256'] and c['harness_sha256']==native['harness']['sha256']
    assert selected['best']['file_sha256']==test['checkpoint_sha256']
    assert selected['last']['file_sha256']==status['checkpoint']['sha256']
    assert selected['best']['counters']['instance_exposures']==best_val['instance_exposures']
    assert tuple(selected['best']['best_metric'])==(best_val['feasible_count'],-best_val['mean_distance_km'])
    assert selected['last']['counters']['instance_exposures']==96000
    # Keep campaign and test artifacts byte-identical. Preserve source paths inside JSON.
    for f in p.iterdir():
        if f.is_file() and f.suffix in ('.json','.csv') and not f.name.startswith('gurobi'):
            verified_copy(f,target/f.name)
    for sub in ('assets','scheduler_a6000'):
        if (p/sub).exists():
            for f in (p/sub).rglob('*'):
                if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.log'):
                    verified_copy(f,target/f.relative_to(p))
    for f in q.iterdir():
        if f.is_file() and f.suffix in ('.json','.jsonl','.yaml','.log'):
            verified_copy(f,target/f.relative_to(p))
    for sub in ('test','native_evaluations','native_data'):
        for f in (q/sub).rglob('*'):
            if f.is_file() and f.suffix in ('.json','.jsonl','.log'):
                verified_copy(f,target/f.relative_to(p))
    for group,sub in [('source','native_source'),('harness','native_harness')]:
        for record in native[group]['files']:
            relfile=Path(record['path']);verified_copy(q/sub/relfile,target/'runs'/method/sub/relfile,record['sha256'])
    for relfile,expected in m['source']['files'].items():
        verified_copy(p/'source/CaliRoute'/relfile,target/'source/CaliRoute'/relfile,expected)
    # Record the actual evaluation driver too (RADAR required a separate acceptance-only fix).
    if method=='radar':
        driver=p/'evaluation_driver_metric_fix1';prov=read(driver/'provenance.json')
        for f in driver.iterdir():
            if f.is_file():verified_copy(f,target/f.relative_to(p))
        for relfile,expected in prov['source_files'].items():
            verified_copy(driver/'CaliRoute'/relfile,target/'evaluation_driver_metric_fix1/CaliRoute'/relfile,expected)
        driver_record={'kind':'acceptance_metric_tuple_list_fix','provenance':{k:v for k,v in prov.items() if k!='source_files'}}
    else:
        evaluator=Path(test['evaluator_path'])
        verified_copy(evaluator,package/'CaliRoute'/evaluator.relative_to(ROOT),test['evaluator_sha256'])
        driver_record={'path':str(evaluator),'sha256':test['evaluator_sha256']}
    data={split:read(q/f'native_data/{split}.json') for split in ('train','val','test')}
    assert not ({r['instance_id'] for r in data['test']['instances']} & {r['instance_id'] for split in ('train','val') for r in data[split]['instances']})
    assert data['test']['source_sha256']==test['test_data_sha256']
    summary_path=target/'runs'/method/'test/instances.summary.json'
    missing_summary=not (q/'test/instances.summary.json').exists()
    if missing_summary:dump(summary_path,{**test['result'],'_archive_provenance':{'generated_during_archiving':True,'source':'original test/manifest.json.result; cross-checked against instances.jsonl','not_an_original_run_file':True}})
    evidence={'label':label,'method':method,'run_id':cfg['run_id'],'actual_campaign_path':str(p),
        'recorded_campaign_path':m.get('campaign',str(p)),'package_campaign_path':str(target.relative_to(package)),
        'resolved_config':cfg,'architecture':read(q/'native_architecture.json'),
        'source':{k:v for k,v in m['source'].items() if k!='files'},
        'native_source':{k:v for k,v in native['source'].items() if k not in ('files','upstream_provenance')},
        'upstream_provenance':{k:v for k,v in native['source'].get('upstream_provenance',{}).items() if k!='files'},
        'harness_sha256':native['harness']['sha256'],'resolved_config_sha256':native['config_sha256'],
        'checkpoint_metadata':selected,'best_validation':best_val,'native_status':status,
        'training_log_summary':{'rows':len(train),'batch_histogram':dict(collections.Counter(r['batch_instances'] for r in train)),
          'lr_values':sorted({r['lr'] for r in train}),'first_timestamp':train[0]['timestamp'],'last_timestamp':train[-1]['timestamp'],
          'instance_exposures':sum(r['batch_instances'] for r in train),'complete_rollouts':sum(r['completed_rollouts'] for r in train),
          'nominal_data_passes':19.2,'validation_records':len(vals)},
        'test_manifest':test,'timing_and_decoding':{k:rows[0].get(k) for k in ('decode_mode','num_augment','timing_scope','runtime_kind','timing_batch_size','start_policy','unique_first_customers','implicit_final_return')},
        'datasets':{s:{k:v for k,v in d.items() if k!='instances'} for s,d in data.items()},
        'evaluation_driver':driver_record,'instances_summary_reconstructed':missing_summary,
        'checks':{'frozen_training_source_files_verified':len(m['source']['files']),
          'native_source_files_verified':len(native['source']['files']),'native_harness_files_verified':len(native['harness']['files']),
          'test_rows_and_hash_verified':True,'best_checkpoint_hash_and_selection_verified':True,'test_ids_disjoint_train_val':True},
        'omitted':'Model/optimizer tensors, all .pt/.pth checkpoints and raw .pkl/.npz datasets; metadata and hashes retained'}
    dump(package/'ARCHIVE_EVIDENCE.json',evidence);all_evidence.append(evidence)
    dump(target/'runs'/method/'native_checkpoints/metadata_only.json',{'weights_included':False,**selected})
    hashes={str(f.relative_to(package)):sha(f) for f in sorted(package.rglob('*')) if f.is_file()}
    (package/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    with tarfile.open(OUT/f'{label}.tar.gz','w:gz') as tar:tar.add(package,arcname=label)
    with tarfile.open(OUT/f'{label}.tar.gz','r:gz') as tar:
        for info in tar.getmembers():
            if info.isfile():
                assert not info.name.endswith(('.pt','.pth','.pkl','.npz'))
                relname=str(Path(info.name).relative_to(label))
                if relname in hashes:assert hashlib.sha256(tar.extractfile(info).read()).hexdigest()==hashes[relname]
    print(label,'archive_bytes',(OUT/f'{label}.tar.gz').stat().st_size,'files',len(hashes),flush=True)

dump(OUT/'baseline_evidence.json',all_evidence)
doc=['# E1 原生基线留档：RADAR-CVRP100 与 RRNCO-VRPTW100','',
     '本报告直接读取实际 run 的冻结配置、源码清单、训练/验证 JSONL、best/last checkpoint 元数据和独立 test 输出。未重新训练或重新选择 checkpoint。读者无需访问服务器路径即可查看下述关键事实。',
     '', '## 结论与 E2 匹配边界','',
     '**两组可作为已有 E1 受控预算原生基线的留档，但尚不能据此认定为 E2 的匹配基线。** 目前没有提供 E2 的预算、数据、预处理和推理协议，无法逐项判定等价。',
     '', '- 两组不是 SL-PPO/PPO：配置声明 `native_REINFORCE_POMO`，执行各自原生训练目标，由 E1 外层管理数据、训练曝光预算、验证和保存。不能套用 PPO passes=4。',
     '- 两组都从 scratch、单 training seed=3009、单 GPU、名义 batch64。完成 96,000 次实例曝光和 4,800,000 条训练轨迹。训练池 5,000 条，所以仅相当于 **19.2 个数据集 pass**；“1500”是 96,000/64 的名义 batch 预算，**不是 1500 个原生完整数据集 epoch**。',
     '- 实际优化器更新 **1517 次**。外层在训练数据遍历边界、验证边界和剩余预算处截断 batch；实际 batch 分布为 64×1483，8/16/24/40/48/56 各5次，32×4次。',
     '- 原生源码 provenance 都标记 `includes_uncommitted_adapters=true`。不能仅凭方法名或上游 commit 声称是未经适配的官方原版。冻结源码和运行 harness 已另附归档。',
     '- 如果 E2 要求原作者长程训练、不同 LR 衰减、不同候选/增强预算、不同物理矩阵或预处理，需要先逐项核对；不满足时应重跑对应协议，而不是直接把 E1 数字当作匹配基线。',
     '', '## 实际协议与结果对照','',
     '| 字段 | RADAR-CVRP100 | RRNCO-VRPTW100 |','|---|---|---|',
     '| Training seed / initialization | 3009 / scratch | 3009 / scratch |',
     '| 算法 | native_REINFORCE_POMO | native_REINFORCE_POMO |',
     '| 训练池 / validation / test | 5000 / 1000 / 1000 | 5000 / 1000 / 1000 |',
     '| 配置 batch / n_traj | 64 / 50 | 64 / 50 |',
     '| 实际曝光 / rollouts / updates | 96000 / 4800000 / 1517 | 96000 / 4800000 / 1517 |',
     '| LR / weight decay | 0.0004 / 0.000001 | 0.0004 / 0.000001 |',
     '| LR milestones（data pass）/ gamma | [2001,2101] / 0.1 | [180,195] / 0.1 |',
     '| 实测训练日志 LR | 全部为 0.0004，未衰减 | 全部为 0.0004，未衰减 |',
     '| 验证频率 / 次数 | 每3200曝光 / 30次 | 每3200曝光 / 30次 |',
     '| Validation K / batch | 50 / 8 | 50 / 8 |',
     '| Test K / batch / augmentation | 50 / 1 / 1 | 50 / 1 / 1 |',
     '| 推理 seed / decode | 17003009 / native_greedy_multistart | 17003009 / native_greedy_multistart |',
     '| 选择规则 | 先最大验证可行数，再最小可行实例平均km | 同左 |',
     '| best 所在曝光 / 更新步 | 28800 / 455 | 96000 / 1517 |',
     '| best 验证成本（km） | 204.30174066493848 | 239.08668082952127 |',
     '| test 平均成本（km） | 204.3652851530332 | 239.38207336870022 |',
     '| test 可行数 | 1000/1000 | 1000/1000 |',
     '| test 单实例 latency（s） | 0.07344711947627365 | 0.1986546188122593 |',
     '', '两组 test 的 1000 条唯一实例、候选数、seed、checkpoint ID、逐实例文件 SHA256 和均值均已重新核对。此处没有重新运行物理路线 replay；原始 test 验收和成本重算记录已保留，成本不一致候选数均为0。',
     '', '时间口径：batch1 单实例 latency，包含内存内环境/模型预处理、编码/SVD、全部候选解码、路线验证和最优可行解选择；不含文件IO、原始数据反序列化/适配及 checkpoint 加载。机器常称 A6000，但实卡为 NVIDIA RTX 6000 Ada Generation（49140 MiB）。',
     '', '## 训练时间口径','',
     '`native_status.json.counters.training_wall_time_s` 只累加 train_batch 执行时间，不是完整训练 wall-clock，不能直接除3600填论文的训练小时。RADAR为428.6926997471601秒、验证累计436.280432137195秒；RRNCO为2242.8016848210245秒、验证累计1074.3436151635833秒。初始化、数据转换、保存等开销不包含于上述训练计数。完整 run 的累计 wall-clock 若无覆盖全阶段的起止记录，应保留 TBD。',
     '', '## 适配与评价实现差异','',
     '- RADAR：实际冻结 harness 调用 ACVRPTrainer._train_one_batch；使用 SVD 初始化、embedding256、5层、8头，参数3300264。E1提供有向道路距离、数据转换、曝光预算外循环、独立可行性验证及验证选 best。',
     '- RRNCO：调用 RRNet.shared_step；使用原生 rcvrptw policy/environment 接口及道路时间契约适配，embedding128、6层、8头、instance normalization，参数4290252。启用了 `rrnco_encoder_checkpointing=true`（编码层激活重计算）；这是本机显存相关改动，源码已冻结。',
     '- RRNCO 输入启用坐标、距离和极坐标特征；配置 `task_adapter=rrnco_native_rcvrptw_with_road_time_contract`。时间量按相同 working horizon 归一化。两者坐标均采用 native_per_axis_minmax，需求和容量保留原始量，有向距离单位km。',
     '- RRNCO test 使用50个不同起始客户，记录 `implicit_final_return=true`：原生末点回仓隐式边在验证时显式化。RADAR记录 `implicit_final_return=false`；其原始逐实例文件未单列 start_policy/unique_first_customers，不能凭空补填。',
     '- RADAR test 使用单独的 evaluation_driver_metric_fix1：只修正验收时 JSON list 与 checkpoint tuple 的 best_metric 数值比较。补丁、原始训练 source 和实际评价 driver 均留档；不把这次验收修复记作重新训练。',
     '- 两个 native_status.json 的 test_complete=false 是训练阶段遗留字段；最终 test 状态以 runs/<method>/test/manifest.json 的 accepted=true 为准。',
     '', '## 源码、权重和数据指纹','']
for e in all_evidence:
    s=e['source'];best=e['checkpoint_metadata']['best'];last=e['checkpoint_metadata']['last']
    doc.extend([f'### {e["label"]}','',f'- Run：`{e["run_id"]}`',f'- 主仓库 commit：`{s["git_commit"]}`',f'- 冻结训练源码聚合 SHA256：`{s["content_sha256"]}`',
       f'- 原生代码 provenance 的上游 commit：`{e["upstream_provenance"].get("git_head")}`（包含未提交适配的标记，不能等同于 clean checkout）',
       f'- 实际原生源码 SHA256：`{e["native_source"]["sha256"]}`',f'- 实际 harness SHA256：`{e["harness_sha256"]}`',f'- resolved config SHA256：`{e["resolved_config_sha256"]}`',
       f'- best.pt SHA256：`{best["file_sha256"]}`',f'- last.pt SHA256：`{last["file_sha256"]}`',f'- test instances.jsonl SHA256：`{e["test_manifest"]["instances_sha256"]}`',
       '- 保存时的主仓库未提交状态：','```text',s.get('git_status','').rstrip(),'```','',
       '| 数据 split | 原始 instances.pkl SHA256 | native npz SHA256 |','|---|---|---|'])
    for split,v in e['datasets'].items():doc.append(f'| {split} | `{v["source_sha256"]}` | `{v["npz_sha256"]}` |')
    doc.extend(['','实际 resolved 配置（原样，不按当前默认值补写）：','```json',json.dumps(e['resolved_config'],ensure_ascii=False,indent=2),'```','',
        'best/last checkpoint 提取的运行时版本：','```json',json.dumps(best.get('runtime_versions',{}),indent=2,default=str),'```',''])
doc.extend(['## 文件交付与来源说明','',
 '同目录提供 RADAR_CVRP100.tar.gz 和 RRNCO_VRPTW100.tar.gz，按 CaliRoute/results/... 保留实际目录结构。每包有 ARCHIVE_EVIDENCE.json 和 SHA256SUMS，包含原始配置、manifest、冻结源码/native_source/native_harness、训练日志、30次验证的逐实例记录、评价driver以及独立test文件。权重张量和原始数据集未打包，best/last元数据及文件哈希已经提取。',
 '', '**两组原目录中都没有 instances.summary.json。** 归档中的同名文件是本次从原始 test/manifest.json.result 提取，并用原始 instances.jsonl 核对后生成；文件内有 `_archive_provenance` 标记。原始 manifest/JSONL 不改动，也没有将生成文件伪称为原运行产物。',
 '', 'RADAR-CVRP100 后来由 results/e1 移到 results/e1_cvrp_cus100。原始 JSON 中旧绝对路径保留为来源证据，解包后按包内相对目录查找。此报告的配置/指标/哈希均直接内嵌，可以转发给 E2 核查者；无需能访问那些绝对路径才能阅读。',
 '', '报告只确认实际 E1 运行与证据自洽，不提前认定 E2 匹配；也没有为此另启动训练。'])
(OUT/'E1_native_baseline_audit.md').write_text('\n'.join(doc)+'\n')
(OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
print('REPORT',OUT/'E1_native_baseline_audit.md')
