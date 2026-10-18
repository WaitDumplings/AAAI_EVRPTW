# A6000：E2 RADAR / RRNCO 核查报告

核查起点：`ca71bd3df61205992803b0f8309e6826df5f2aae`，分支 `fix/e1-no-road-base-20261012`。本机原有修复已保存并恢复；本次核查代码另发独立 review 分支。**没有启动 E1 或 E2 正式训练。** 本次运行的是准备队列、原生首 batch 复现及小样本接口测试。

## 四个直接结论

1. **RADAR E1 → E2-A：conditional。** 原生 12 个源码文件与冻结 E1 完全相同；原规格首 batch 的 loss、成本、decision steps 和完整轨迹数精确复现。若增强组沿用本报告固定的数据、训练预算、随机初始化、评估和选 best 协议，可复用 E1 原生基线；当前不能把尚未冻结的 E2 配置自动认定为匹配。不同训练预算/采样/预处理/候选策略必须补跑匹配基线。
2. **RRNCO E1 → E2-A：conditional。** 124 个原生源码文件相同，首 batch 同样精确复现。任务适配和编码器 checkpointing 必须保持；RRNCO＋AGDA 的完整 decoder/context 接入仍为 **not_tested**，下文提供具体插入位置。
3. **RADAR＋AGDA 原生接入：verified（短程范围）。** 已补齐 set_kv/set_q1、真实 CVRP 状态、visited 记录、参数注册、随机数保护、缓存生命周期；真实 Cus100 小 batch 完整路线、原生三次更新、物理 validator、保存恢复全部通过。**修正原型 bias 顺序**，与冻结 E1 一样放在 tanh clipping 后。新增正式 campaign 接线和 batch64×50 显存测试仍未验收。
4. **RRNCO 逐动作 replay：verified；PPO 四组：conditional。** 固定编码邻居后，FP32/BF16 的 old/replay log-prob 最大差为 0，逐动作 ratio=1、近似 KL=0。已完成同一 actor/critic 的两次 PPO 更新和两次 SL-PPO 核心损失更新、train 专家 teacher forcing。完整 SL-PPO 调度/门控/成本记忆持久化、AWBC/DAPG、四组正式 worker 和长程恢复均 **not_tested**。E1 REINFORCE **不能**作为 E2-B PPO 基线。

## P0：实际 E1 协议与核查

| 项目 | RADAR-CVRP100 | RRNCO-VRPTW100 |
|---|---:|---:|
| Training seed / 初始化 | 3009 / scratch | 3009 / scratch |
| 常规 batch / n_traj / 单卡 | 64 / 50 / 是 | 64 / 50 / 是 |
| optimizer updates | 1517 | 1517 |
| 实例曝光 / 完整轨迹 | 96000 / 4800000 | 96000 / 4800000 |
| 累计 decision steps | 496507190 | 489933517 |
| 训练池 / 名义数据遍历量 | 5000 / 19.2 | 5000 / 19.2 |
| Adam LR / weight decay | 4e-4 / 1e-6 | 4e-4 / 1e-6 |
| MultiStepLR milestones / gamma | [2001,2101] / 0.1 | [180,195] / 0.1 |
| 训练日志实际 LR | 全部 4e-4，未衰减 | 全部 4e-4，未衰减 |
| best 曝光 / optimizer update | 28800 / 455 | 96000 / 1517 |
| best validation km | 204.30174066493848 | 239.08668082952127 |
| Test km / 可行数 | 204.3652851530332 / 1000 | 239.38207336870022 / 1000 |

`decision_steps` 日志列是累计计数，不能把1517行再求和。批次分布实测：64×1483，8/16/24/40/48/56各5次，32×4次。逐批完整记录在 `radar_batch_sequence.csv`、`rrnco_batch_sequence.csv`。19.2不是1500个完整数据集 epoch，RRNCO也不是每条实例恰好出现19.2次。

**Sampler 与 scheduler：** RADAR 每个数据 pass 无放回，`ACVRPEnv.start_fixed_epoch` 使用独立 `np.random.default_rng(3009 + data_pass)`；正式首 pass=1，因此首轮 permutation seed=3010。到数据尾部/验证边界截断 batch；新 pass 开始时先调用 scheduler.step 再训练，保留其原生调用顺序（会产生 PyTorch warning）。RRNCO `FixedRMTVRPGenerator._sample_indices` 使用全局 `np.random.randint` **有放回**；每满5000曝光后调用 scheduler.step，余量只控制计数边界，不改变有放回规则。两个模型的 rollout/SVD/邻居抽样还使用 PyTorch RNG。原日志未逐次记录训练 instance ID 抽样序列，不将其伪称为完整逐样本审计；本次首 batch 复现是额外可重复性证据。

**规范化与物理契约：** 原始 D 是有向道路 km；坐标采用逐轴 min-max，原始 demand/capacity 保留。RADAR 环境用需求/容量，模型对 D 做实例均值/标准差归一化后随机 SVD(q=10)，SVD需求特征还除总需求；Sinkhorn/encoder/glimpse未改变。RRNCO环境对 D 做实例 min-max，actor/原生 alpha 使用该 D_norm，原生 REINFORCE使用 normalized reward；报告和 validator 用物理 km。VRPTW T 是权威道路秒数/同一实例 working horizon，时间窗减 depot opening 后除相同 horizon，服务时间同尺度，**不是从 D 或欧氏距离推导**。mask检查服务开始≤TW关闭、服务完成+返仓≤depot关闭和容量；step实现提前到达等待、服务后时钟、返仓重置。RRNCO最后一段返仓由reward隐式计价，交独立validator时显式化。数据 D 对角线为0的本任务下成本重算已通过；不将这外推到任意非零距离最小值的数据。

**验证/选 best：** 每3200曝光评1000条 val，共30次；K=50、batch8、augment1；先最大可行数，再最小独立重算平均km。test只用选择好的best，1000条独立实例，K=50、batch1、inference seed=17003009、augment1、cap201。模型时间含编码/SVD、全部候选和验证选择；不含文件IO/数据转换/checkpoint载入。

**实际 decode：** RADAR `eval_type='softma'` 不等于代码比较的 `'softmax'`：model.eval时走argmax，训练时由model.training强制sampling。RRNCO明确指定`multistart_greedy`。两者起始客户均按原生1…50分配、没有重复起点；RRNCO用(S B)展开。不同起点不保证不存在等价路线集合；原test只保存选中路线，没有全部50条候选路线，无法事后量化等价解去重比例，不能补写为“50个独立随机采样解”。RADAR显式终止返仓；RRNCO传给native policy的max_steps=200，但该实现使用step>max_steps才break，代码理论上可到201个循环动作再加forced start；本次不悄悄修正该原生边界。E2-B新接口用严格cap201并对截断直接报错，四组需一致。

**源码与数据：** `source_comparison.csv` 共149项（136个原生源码+13个harness条目）逐文件对照冻结清单、ca71bd3和当前worktree。原生宿主无差异。RADAR harness主要新增VRPTW分支/小规模起点适配，RRNCO还依赖本机已有的encoder checkpointing与TensorDict device修复；E1 validator新增EVRPTW分支不改变这两个任务路径。完整差异保存在`patches/*frozen_harness_vs_head.patch`（方向HEAD→历史冻结，仅审计用途，**不要整包覆盖当前E1**）。

六个split的原始pkl、转换npz及元数据哈希已重新读取核验，ID列表及SHA在`identities/`与`baseline_actual_evidence.json`。历史主仓库commit、上游commit、源码聚合SHA、best/last SHA、runtime版本在`E1_native_baseline_audit.md`和证据包。它们与本次review HEAD不同。

**随机初始化附加证据：** 两组冻结代码首batch64×50精确复现。RADAR reconstructed初始权重SHA=`237f292a787d19593e2a34a0bea0dbc03b512762630c042b8bfdd056cc77bc8f`；RRNCO=`5da833c4b2d595a918166f79937d9798bb947c56130988e0aa0c47f5dac43acf`。这些是本次重建的初始哈希，历史没有单独保存初始化checkpoint，不能冒充历史产物。RADAR增强组共享宿主初始参数和CPU/CUDA RNG与无插件组均逐项一致；插件初始化使用fork_rng，不能靠加载已训练best获得预算优势。

## P1：RADAR真实接入及工程测试

补丁：`e2/adapters/radar.py::install_radar_agda`。在原生 `ACVRPTrainer` 创建optimizer之前替换Model factory，原生 `_train_one_batch`不改。所有注册参数恰好被optimizer覆盖一次。原SVD、Sinkhorn、K/V、glimpse保持；每次pre_forward重建编码/KV并清空动态上下文，不跨更新复用参数相关缓存。

采用明确的CVRP任务特征：candidate `[B,S,N,9]`=当前→候选、候选→当前、候选→depot、depot→候选（四者除实例最大道路km）、需求/容量、剩余load、visited、可行mask、depot标志。system `[B,S,4]`=load、未服务总需求/容量、未服务比例、当前到depot距离比。四个token为当前节点、depot、全图均值、未服务客户均值。visited通过实际action记录，不能把容量禁用mask误认成visited。**无虚构时间或电量输入；这是任务专用schema，并非宣称复现E1 30/15维输入。**

残差顺序：`(glimpse·native_key + glimpse·delta_key)/sqrt(D)` → 原生`tanh*clip` → `+action_bias` → hard mask → softmax。ca71bd3原型把bias加在tanh内，本次已修正；对应冻结E1 `graph_model/decoder.py::calc_logits`与`multi_head_attention.py::AttentionScore.forward`的实际顺序。

真实 train CVRP100，B=2、S=3，三次原生更新；这些小样本数字不代表正式64×50的峰值。

| 验收项 | 实测 |
|---|---:|
| disabled / zero概率最大差 | 0.0 / 0.0 |
| zero固定动作log-prob / 宿主梯度最大差 | 0.0 / 0.0 |
| zero成本最大差 | 0.0 |
| 新插件初始非零梯度L1 | 38.18328761776729 |
| mask动作概率最大值 | 0.0 |
| 保存加载后概率最大差 | 0.0 |
| 额外注册参数量 | 1919811 |
| CUDA峰值allocated / 时间 | 1.095 GiB / 6.099s |

六条完整路线全部通过独立物理validator；更新后策略概率确实改变。额外参数数含AGDA核心里关闭分支的注册参数，不代表全部分支都有非零梯度。已测的是启用action-key/bias的训练信号，不把关闭K/V分支包装成已训练功能。

## P1：RRNCO逐动作actor、AGDA建议与PPO边界

`e2/adapters/rrnco_step.py`实现实验接口：

- `encode_static(policy,state,auxiliary_randomness)`：记录或复用每实例每节点的25个邻居索引；每个更新都重算当前encoder、投影和cache，绝不复用旧detach embedding。
- `step_logits(...)`：保留原生 `_compute_q/_compute_kvl/pointer`、alpha/beta及`log(exp(pointer-alpha*D-beta*T)+1e-6)`；再用原生process_logits处理tanh、mask和temperature。
- `run_actions(...,actions=...)`：完整rollout或teacher forcing，输出逐动作logp/entropy/policy_valid/value/reward；保持(S B)顺序，显式检查mask及动作合法性。forced首客户logp=0且不参与policy loss；终端后padding不参与policy或value loss。first forced动作的道路reward仍进入GAE；终端转移加隐式返仓。无合法动作、截断、专家非法动作直接失败，不修复路线。
- `StepCritic`：当前节点128维+均值128维+时间+剩余容量，四组共用；value梯度也流入共享actor encoder。gamma=1、lambda=.95，统一固定100km reward尺度，物理成本输出仍为km。

原 `DistanceExpert._sample_indices(sample_type='prob')` 每次按逆距离multinomial重新抽25个邻居，train/eval都随机。未更新参数但不固定索引时，实测log-prob最大差 **13.382738**；固定索引后FP32和BF16最大差均 **0.0 / 0.0**，FP32 ratio最大差=0.0、近似KL=0.0。与原生greedy的动作和逐步log-prob一致。真实配置是instance normalization，无Dropout/BatchNorm；rollout/replay均固定train phase，不能暗中切换模式或候选顺序。

**RRNCO＋AGDA建议（not_tested）：** 保持AAFM/NAB和其道路/时间偏置；在`RRNet_PointerAttention.forward`原生`glimpse = ffn(heads+query)+(heads+query)`之后，把delta_action_key加到logit_key，保留sqrt(D)缩放。然后完整保留decoder alpha/beta和log-exp非线性；action-bias应在`DecodingStrategy.step/process_logits`原生tanh之后、mask/temperature之前加入。需要新增decoder→decode-strategy动态bias上下文，不能只替换pointer就宣称接入完成；也不能把bias塞入原生exp或删除alpha/beta。当前没有改变这些原生运算的数值写法。

真实 train VRPTW100 B=2、S=3；变长padding、forced start、完整路线/物理时间、非法动作及截断拒绝均已测。外部专家只来自train的前两条，并先验证完整路线及成本；首客户作为外部给定条件排除可学习logp，之后所有动作与on-policy使用同一mask/概率精度。用于三个经验方法的完整专家pool哈希已记录，但本次没有全量专家回放验收。

**短更新范围：** 同一初始actor/critic分别执行两次PPO/GAE和两次PPO＋现有portable SL-PPO group/expert route-surrogate更新。专家成本用于reference，自己的轨迹仅用于本次on-policy PPO更新，没有新增self-replay。完整E1 SL-PPO的权重门控、cost-incumbent跨批持久化和调度尚未迁移，不能把此工程测试命名为“正式SL-PPO实现已验收”。

最终短测actor LR=1e-06, critic LR=0.0001。PPO更新后近似KL=0.00108266，SL-PPO核心更新后近似KL=0.00321649。较高actor LR=1e-4的诊断结果另存`rrnco_high_lr_diagnostic.json`，KL过大；不能照抄native LR=4e-4当PPO配置。四组建议参数在`proposed_protocols/rrnco_vrptw100_step_ppo_quartet.json`，**只是提案，未获批准也未启动**。正式worker必须实现相同KL guard，并测试完整batch64×50和四passes。

RRNCO小样本峰值allocated=0.441 GiB，耗时11.551s；机器实际为NVIDIA RTX 6000 Ada Generation，俗称A6000。此数不是完整训练显存需求。

## 复现、交付与校验

本次使用原有两个Python3.12环境，torch=2.7.0+cu126；完整pip freeze在`*_environment.lock.txt`，没有升级环境。`prototype_tests.log`：19 passed，涵盖原型协议/服务器计划、RADAR接口、GAE/forced/padding和VRPTW契约。独立GPU测试明细在`smoke_results.json`和各smoke日志。

仓库根目录检出review分支，进入CaliRoute；先核对报告与文件SHA，再在有E1数据/冻结目录的服务器执行：

```bash
E2_PYTHON=/home/exx/anaconda3/envs/maojie/bin/python E2_DATA_ROOT=/data/Maojie/AAAI/AAAI_Dataset bash scripts/run_e2_a6000.sh --prepare
# 两个命令是短测，不是正式训练；可按接收机环境替换 Python 路径。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /data/Maojie/AAAI/.venvs/e1-radar/bin/python scripts/review_e2_native_smoke.py --method radar --output results/e2/review_reproduction
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 /data/Maojie/AAAI/.venvs/e1-rrnco/bin/python scripts/review_e2_native_smoke.py --method rrnco --output results/e2/review_reproduction
# 首batch历史复现：相同参数，脚本替换为review_e2_native_first_batch.py。
```

准备结果位于`results/e2/E2_CUS100_S3009_V1_a6000/status.json`，状态`prepared_not_training`。现有prepare脚本始终保留执行阻塞；本次未把它改成训练入口。

交付：本报告、`baseline_reuse_decision.json`、149行`source_comparison.csv`、`interface_map.csv`、`smoke_results.json`、三份协议提案、`acceptance_checklist.csv`、补丁与文件哈希；原始训练/验证日志和test文件在`baseline_metadata/`；完整E1冻结证据另附`RADAR_CVRP100.tar.gz`、`RRNCO_VRPTW100.tar.gz`，不含大型checkpoint或原始数据数组。`checksums.sha256`可用`sha256sum -c checksums.sha256`校验。

原始两组都缺`instances.summary.json`，两份历史E1压缩包中的同名文件由上轮派生，仅有来源标记；本轮`baseline_metadata/*/test/instances.summary.json`已重新派生，补齐UTC派生时间、原始manifest/JSONL SHA、生成脚本SHA和一致性检查。生成代码为`scripts/derive_e2_review_test_summary.py`，上轮原生成代码也附为`derive_e1_archive.py`。原始run未改动。这些说明和证据在GitHub review分支可直接阅读/下载，无需协调机器访问A6000绝对路径。
