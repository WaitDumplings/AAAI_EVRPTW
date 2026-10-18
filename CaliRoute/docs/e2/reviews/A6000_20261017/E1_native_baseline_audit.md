# E1 原生基线留档：RADAR-CVRP100 与 RRNCO-VRPTW100

本报告直接读取实际 run 的冻结配置、源码清单、训练/验证 JSONL、best/last checkpoint 元数据和独立 test 输出。未重新训练或重新选择 checkpoint。读者无需访问服务器路径即可查看下述关键事实。

## 结论与 E2 匹配边界

**两组可作为已有 E1 受控预算原生基线的留档，但尚不能据此认定为 E2 的匹配基线。** 目前没有提供 E2 的预算、数据、预处理和推理协议，无法逐项判定等价。

- 两组不是 SL-PPO/PPO：配置声明 `native_REINFORCE_POMO`，执行各自原生训练目标，由 E1 外层管理数据、训练曝光预算、验证和保存。不能套用 PPO passes=4。
- 两组都从 scratch、单 training seed=3009、单 GPU、名义 batch64。完成 96,000 次实例曝光和 4,800,000 条训练轨迹。训练池 5,000 条，所以仅相当于 **19.2 个数据集 pass**；“1500”是 96,000/64 的名义 batch 预算，**不是 1500 个原生完整数据集 epoch**。
- 实际优化器更新 **1517 次**。外层在训练数据遍历边界、验证边界和剩余预算处截断 batch；实际 batch 分布为 64×1483，8/16/24/40/48/56 各5次，32×4次。
- 原生源码 provenance 都标记 `includes_uncommitted_adapters=true`。不能仅凭方法名或上游 commit 声称是未经适配的官方原版。冻结源码和运行 harness 已另附归档。
- 如果 E2 要求原作者长程训练、不同 LR 衰减、不同候选/增强预算、不同物理矩阵或预处理，需要先逐项核对；不满足时应重跑对应协议，而不是直接把 E1 数字当作匹配基线。

## 实际协议与结果对照

| 字段 | RADAR-CVRP100 | RRNCO-VRPTW100 |
|---|---|---|
| Training seed / initialization | 3009 / scratch | 3009 / scratch |
| 算法 | native_REINFORCE_POMO | native_REINFORCE_POMO |
| 训练池 / validation / test | 5000 / 1000 / 1000 | 5000 / 1000 / 1000 |
| 配置 batch / n_traj | 64 / 50 | 64 / 50 |
| 实际曝光 / rollouts / updates | 96000 / 4800000 / 1517 | 96000 / 4800000 / 1517 |
| LR / weight decay | 0.0004 / 0.000001 | 0.0004 / 0.000001 |
| LR milestones（data pass）/ gamma | [2001,2101] / 0.1 | [180,195] / 0.1 |
| 实测训练日志 LR | 全部为 0.0004，未衰减 | 全部为 0.0004，未衰减 |
| 验证频率 / 次数 | 每3200曝光 / 30次 | 每3200曝光 / 30次 |
| Validation K / batch | 50 / 8 | 50 / 8 |
| Test K / batch / augmentation | 50 / 1 / 1 | 50 / 1 / 1 |
| 推理 seed / decode | 17003009 / native_greedy_multistart | 17003009 / native_greedy_multistart |
| 选择规则 | 先最大验证可行数，再最小可行实例平均km | 同左 |
| best 所在曝光 / 更新步 | 28800 / 455 | 96000 / 1517 |
| best 验证成本（km） | 204.30174066493848 | 239.08668082952127 |
| test 平均成本（km） | 204.3652851530332 | 239.38207336870022 |
| test 可行数 | 1000/1000 | 1000/1000 |
| test 单实例 latency（s） | 0.07344711947627365 | 0.1986546188122593 |

两组 test 的 1000 条唯一实例、候选数、seed、checkpoint ID、逐实例文件 SHA256 和均值均已重新核对。此处没有重新运行物理路线 replay；原始 test 验收和成本重算记录已保留，成本不一致候选数均为0。

时间口径：batch1 单实例 latency，包含内存内环境/模型预处理、编码/SVD、全部候选解码、路线验证和最优可行解选择；不含文件IO、原始数据反序列化/适配及 checkpoint 加载。机器常称 A6000，但实卡为 NVIDIA RTX 6000 Ada Generation（49140 MiB）。

## 训练时间口径

`native_status.json.counters.training_wall_time_s` 只累加 train_batch 执行时间，不是完整训练 wall-clock，不能直接除3600填论文的训练小时。RADAR为428.6926997471601秒、验证累计436.280432137195秒；RRNCO为2242.8016848210245秒、验证累计1074.3436151635833秒。初始化、数据转换、保存等开销不包含于上述训练计数。完整 run 的累计 wall-clock 若无覆盖全阶段的起止记录，应保留 TBD。

## 适配与评价实现差异

- RADAR：实际冻结 harness 调用 ACVRPTrainer._train_one_batch；使用 SVD 初始化、embedding256、5层、8头，参数3300264。E1提供有向道路距离、数据转换、曝光预算外循环、独立可行性验证及验证选 best。
- RRNCO：调用 RRNet.shared_step；使用原生 rcvrptw policy/environment 接口及道路时间契约适配，embedding128、6层、8头、instance normalization，参数4290252。启用了 `rrnco_encoder_checkpointing=true`（编码层激活重计算）；这是本机显存相关改动，源码已冻结。
- RRNCO 输入启用坐标、距离和极坐标特征；配置 `task_adapter=rrnco_native_rcvrptw_with_road_time_contract`。时间量按相同 working horizon 归一化。两者坐标均采用 native_per_axis_minmax，需求和容量保留原始量，有向距离单位km。
- RRNCO test 使用50个不同起始客户，记录 `implicit_final_return=true`：原生末点回仓隐式边在验证时显式化。RADAR记录 `implicit_final_return=false`；其原始逐实例文件未单列 start_policy/unique_first_customers，不能凭空补填。
- RADAR test 使用单独的 evaluation_driver_metric_fix1：只修正验收时 JSON list 与 checkpoint tuple 的 best_metric 数值比较。补丁、原始训练 source 和实际评价 driver 均留档；不把这次验收修复记作重新训练。
- 两个 native_status.json 的 test_complete=false 是训练阶段遗留字段；最终 test 状态以 runs/<method>/test/manifest.json 的 accepted=true 为准。

## 源码、权重和数据指纹

### RADAR_CVRP100

- Run：`E1_CVRP100_S3009_V1_a6000_cuda_fix1_radar`
- 主仓库 commit：`3e7357d33ba8d186e4140df14ece9830e4df7c23`
- 冻结训练源码聚合 SHA256：`832b26c3b7b450ca4ca95cc78c36bf36ac814923e09b5a31e9a5f680c020f998`
- 原生代码 provenance 的上游 commit：`74645edb8fce5be2d5889bbb231265526a34de6a`（包含未提交适配的标记，不能等同于 clean checkout）
- 实际原生源码 SHA256：`c3fd4de8ccd1621c3cd69b28870ae3c037c4510f31c6ce2ad87f3a7ec3cf1ecf`
- 实际 harness SHA256：`9ff605705c2557d72af3548200ed5de83af60d09e6c593361801b3b69a325eb3`
- resolved config SHA256：`9ed40a65359a2d1b2704f51627ba7d8219aa6728593e03e608b849a7adbb858f`
- best.pt SHA256：`9a26c61420b27830dab845146ecbf88967987ca6a7a699b1de8d74b10ccb2c17`
- last.pt SHA256：`2b2a64c872ad01757f888c4ac1e11cd6180b175f2840bbe91a005be20bdcb2fe`
- test instances.jsonl SHA256：`620fe3785cb025e2985cc06807ff5d0fa5a679a8a476133e35829f33c8f708de`
- 保存时的主仓库未提交状态：
```text
 M CaliRoute/e1/native_worker.py
```

| 数据 split | 原始 instances.pkl SHA256 | native npz SHA256 |
|---|---|---|
| train | `2e381c2c1d4508180b634cc76587e09e4437764d5cb24f5b47562a3e30b755b5` | `10fe88aa977b70f8cf81d6cc34b35d0efd0ee204131fafca618e0d43b83afbb2` |
| val | `f05de226bf129f361c6bf974795077f007daaacf47bdd3df5e88d133bd48478a` | `e680547227b8991dc833002a4801f0a4575e2fff92489a6db1e3022c34cda4a1` |
| test | `a024aaa7c5c3945c03e8fc0e4ee3eca24604956c4ac7df6a2a17246381362a65` | `b13d3bfd8d22c2de97b2a0965bf475f41e8cd69732e4e77ffb212ad23eccf589` |

实际 resolved 配置（原样，不按当前默认值补写）：
```json
{
  "backup_dir": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/checkpoint_backups/E1_CVRP100_S3009_V1_a6000_cuda_fix1/radar",
  "batch_size": 64,
  "checkpoint_backup_dir": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/checkpoint_backups/E1_CVRP100_S3009_V1_a6000_cuda_fix1/radar",
  "checkpoint_interval_updates": 50,
  "cpu_threads": 1,
  "decode_cap": 201,
  "epoch_semantics": "one optimizer update per native batch; data_pass is floor(train instance exposures / train pool size)",
  "eval_batch_size": 8,
  "eval_interval_exposures": 3200,
  "eval_k": 50,
  "eval_seed": 17003009,
  "experiment_protocol": {
    "checkpoint_backup_required": true,
    "controlled_epoch_meaning": "one sampled global instance batch followed by PPO/loss updates, NOT a dataset pass",
    "customer_count": 100,
    "data_audit_sha256": "75b84119fa8d1567aaf36e9711389f8b506efd8d4c82ca578ef7477095a703cb",
    "decode_cap": 201,
    "directed_road_metric": true,
    "evaluation_interval_exposures": 3200,
    "expert_pool_sha256": null,
    "fixed_vehicle_cost": 0,
    "global_batch": 64,
    "hardware_timing_comparable_only_on_same_hardware": true,
    "include_depot_returns": true,
    "independent_search": false,
    "inference_seed": 17003009,
    "initialization": "scratch",
    "method": "radar",
    "requested_K": 50,
    "schema": "aaai_e1_cvrp100_v1",
    "selection_rule": "highest validation feasibility; then lowest mean feasible independently validated km",
    "self_trajectory_replay": false,
    "source_content_sha256": "832b26c3b7b450ca4ca95cc78c36bf36ac814923e09b5a31e9a5f680c020f998",
    "task": "cvrp100",
    "test_used_for_training_or_selection": false,
    "total_instance_exposures": 96000,
    "training_seed": 3009,
    "training_seed_count": 1,
    "validation_instances": 1000,
    "world_size": 1
  },
  "inference_seed": 17003009,
  "initialization": "scratch",
  "instance_exposures": 96000,
  "max_steps": 201,
  "method": "radar",
  "n_traj": 50,
  "native_algorithm": "native_REINFORCE_POMO",
  "native_learning_rate": 0.0004,
  "native_lr_milestones_data_passes": [
    2001,
    2101
  ],
  "native_python": "/data/Maojie/AAAI/.venvs/e1-radar/bin/python",
  "native_source_root": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/E1_CVRP100_S3009_V1_a6000_cuda_fix1/source/CaliRoute/benchmarks/native_sources/radar",
  "native_weight_decay": 1e-06,
  "num_augment": 1,
  "num_customers": 100,
  "run_id": "E1_CVRP100_S3009_V1_a6000_cuda_fix1_radar",
  "schema": "aaai_e1_cvrp100_v1",
  "train_path": "/data/Maojie/AAAI/AAAI_Dataset/dataset/cvrp/train/Cus100",
  "training_seed": 3009,
  "val_path": "/data/Maojie/AAAI/AAAI_Dataset/dataset/cvrp/val/Cus100",
  "world_size": 1
}
```

best/last checkpoint 提取的运行时版本：
```json
{
  "torch": "2.7.0+cu126",
  "numpy": "2.2.5"
}
```

### RRNCO_VRPTW100

- Run：`E1_VRPTW100_NATIVE_S3009_A6000_V2_rrnco`
- 主仓库 commit：`f45d579b08415a7ae13ae98374ac948680de0619`
- 冻结训练源码聚合 SHA256：`652f05435d52b9861373468d0fc7a29bd4e1475eb1ab124d6f2497f4c862a728`
- 原生代码 provenance 的上游 commit：`823d510dadf4dd711730ec4fbf337c356a0de6ae`（包含未提交适配的标记，不能等同于 clean checkout）
- 实际原生源码 SHA256：`3f43e3b6d24e0945d5c93297ceacf1518d14e396225ae63db965eb177a960172`
- 实际 harness SHA256：`f98758bbb42a9184c8c9225dff3b3b3c559a05634829d578fc555b99999ce089`
- resolved config SHA256：`9b17849a683f16bf37003f548f9eeb0a1929cd8a2845f5676bb033981eaaa4a4`
- best.pt SHA256：`31202343e857e387c0900ec3e1b9296dea031dc0bc3ae501002de827d59f4c94`
- last.pt SHA256：`fba07877c4b1ebe912bc514bfb4fb7f08a93c9e4818c1103cd077b64782ede6c`
- test instances.jsonl SHA256：`d8a21c3b94c16a10fd8e248970e7f7ee17a19be0330a5d39e80054a4a27e89bd`
- 保存时的主仓库未提交状态：
```text
 M CaliRoute/e1/lifecycle.py
 M CaliRoute/e1/native_vrptw.py
 M CaliRoute/e1/native_worker.py
 M CaliRoute/scripts/run_e1_vrptw_native.py
 M CaliRoute/tests/test_e1_lifecycle.py
 M CaliRoute/tests/test_e1_native_vrptw.py
?? CaliRoute/cvrp50_a6000_launch.log
?? CaliRoute/cvrp50_a6000_launch.log.before_20261012T230940Z
?? CaliRoute/cvrp50_a6000_launch.pid
```

| 数据 split | 原始 instances.pkl SHA256 | native npz SHA256 |
|---|---|---|
| train | `bc09aa2112a3827f98cb1ee564135b054ff3d2a14f8ff6e89d6e30fe0e2fc81f` | `078096065c69b6d22b0fb67d4cf83712b739faebde45637845aef5347f69099b` |
| val | `41abd42dbf67877061bd4886c080536832edcbc2debdd0bf55735d6256837193` | `acb940b7fc366f573369aec8ed4946a5a1fb251ae90031232fcd1660c6ba7adc` |
| test | `10f08f0f518f12dc433b977f8633d26d1d6097ffb2b27ca187e52f6c37757214` | `60ed303694e7f489e662247bf3100cd696ba04e040c5a3b382e298c20eea9d37` |

实际 resolved 配置（原样，不按当前默认值补写）：
```json
{
  "backup_dir": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/checkpoint_backups/E1_VRPTW100_NATIVE_S3009_A6000_V2/rrnco",
  "batch_size": 64,
  "checkpoint_interval_updates": 50,
  "cpu_threads": 1,
  "decode_cap": 201,
  "epoch_semantics": "one optimizer update per native batch; data_pass is floor(train instance exposures / train pool size)",
  "eval_batch_size": 8,
  "eval_interval_exposures": 3200,
  "eval_k": 50,
  "experiment_protocol": {
    "distance_objective": "directed_road_km",
    "expert_data_used": false,
    "global_batch": 64,
    "initialization": "scratch",
    "method": "rrnco",
    "n_traj": 50,
    "nominal_online_loops": 1500,
    "requested_K": 50,
    "schema": "aaai_e1_vrptw100_native_v1",
    "selection_rule": "feasibility_then_mean_physical_km",
    "test_used_for_training_or_selection": false,
    "time_unit": "all_time_quantities_divided_by_same_working_horizon",
    "total_instance_exposures": 96000,
    "training_seed": 3009,
    "validation_instances": 1000
  },
  "inference_seed": 17003009,
  "initialization": "scratch",
  "instance_exposures": 96000,
  "method": "rrnco",
  "n_traj": 50,
  "native_algorithm": "native_REINFORCE_POMO",
  "native_learning_rate": 0.0004,
  "native_lr_milestones_data_passes": [
    180,
    195
  ],
  "native_python": "/data/Maojie/AAAI/.venvs/e1-rrnco/bin/python",
  "native_source_root": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2/source/CaliRoute/benchmarks/native_sources/rrnco",
  "native_weight_decay": 1e-06,
  "num_augment": 1,
  "num_customers": 100,
  "problem_type": "vrptw",
  "rrnco_encoder_checkpointing": true,
  "run_id": "E1_VRPTW100_NATIVE_S3009_A6000_V2_rrnco",
  "task_adapter": "rrnco_native_rcvrptw_with_road_time_contract",
  "train_path": "/data/Maojie/AAAI/AAAI_Dataset/dataset/vrptw/train/Cus100",
  "training_seed": 3009,
  "val_path": "/data/Maojie/AAAI/AAAI_Dataset/dataset/vrptw/val/Cus100",
  "world_size": 1
}
```

best/last checkpoint 提取的运行时版本：
```json
{
  "torch": "2.7.0+cu126",
  "numpy": "2.5.1",
  "rl4co": "0.6.0",
  "tensordict": "0.13.0",
  "torchrl": "0.13.3",
  "lightning": "2.6.5"
}
```

## 文件交付与来源说明

同目录提供 RADAR_CVRP100.tar.gz 和 RRNCO_VRPTW100.tar.gz，按 CaliRoute/results/... 保留实际目录结构。每包有 ARCHIVE_EVIDENCE.json 和 SHA256SUMS，包含原始配置、manifest、冻结源码/native_source/native_harness、训练日志、30次验证的逐实例记录、评价driver以及独立test文件。权重张量和原始数据集未打包，best/last元数据及文件哈希已经提取。

**两组原目录中都没有 instances.summary.json。** 归档中的同名文件是本次从原始 test/manifest.json.result 提取，并用原始 instances.jsonl 核对后生成；文件内有 `_archive_provenance` 标记。原始 manifest/JSONL 不改动，也没有将生成文件伪称为原运行产物。

RADAR-CVRP100 后来由 results/e1 移到 results/e1_cvrp_cus100。原始 JSON 中旧绝对路径保留为来源证据，解包后按包内相对目录查找。此报告的配置/指标/哈希均直接内嵌，可以转发给 E2 核查者；无需能访问那些绝对路径才能阅读。

报告只确认实际 E1 运行与证据自洽，不提前认定 E2 匹配；也没有为此另启动训练。
