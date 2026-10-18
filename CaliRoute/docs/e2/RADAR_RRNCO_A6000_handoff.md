# E2：RADAR / RRNCO 核查任务（交给 A6000 agent）

## 1. 本次目的与范围

请帮助核查两套已经在本机完成 E1 的原生实现，确定它们如何可靠地接入 E2。
本次优先交回**实际代码证据、接口建议和短程验证结果**，不需要重新启动 E1，也不要把 E2 的准备 shell 当作正式训练入口。

E2 区分两类实验：

| 面板 | 任务与宿主 | 对照 | 必须保持不变 |
|---|---|---|---|
| E2-A | CVRP100 / RADAR | 原版 vs 原版＋AGDA | 原生 SVD、Sinkhorn、encoder、glimpse、训练目标与训练预算 |
| E2-A | VRPTW100 / RRNCO | 原版 vs 原版＋AGDA | 原生 AAFM/NAB、距离/时间偏置、原生 trainer 与训练预算 |
| E2-B | VRPTW100 / RRNCO actor | PPO / AWBC / DAPG / SL-PPO | 同一 actor、critic、环境、逐动作 PPO 协议；不额外加入我们的 RDI/AGDA |

E2-A 不把宿主改成 PPO。E2-B 的 PPO 是逐动作 PPO/GAE，不用原生 REINFORCE 结果代替 PPO 基线，也不直接用将整条构造轨迹视为 single-step MDP 的 PPO。

方法边界：外部专家记录可以提供成本和完整轨迹；模型自己的 incumbent 只存成本，不新增 self-replay。三个经验方法应使用相同训练专家，不能使用 val/test 的路线或成本进行训练。

## 2. 已收到的信息，不需要重新人工填写

以下是远端 agent 已回报的结果；当前协调机器尚未读取证据压缩包，因此应区分“远端已核实”与“协调端已复核”。

| 项目 | RADAR-CVRP100 | RRNCO-VRPTW100 |
|---|---:|---:|
| Training seed | 3009 | 3009 |
| 常规 batch / 训练轨迹数 | 64 / 50 | 64 / 50 |
| 实际 optimizer updates | 1,517 | 1,517 |
| 实例曝光量 | 96,000 | 96,000 |
| 训练池大小 | 5,000 | 5,000 |
| LR | 0.0004，未衰减 | 0.0004，未衰减 |
| best optimizer update | 455 | 1,517 |
| Test K / 可行数 | 50 / 1000 | 50 / 1000 |
| Test 平均 km | 204.365285 | 239.382073 |

注意：`96000 / 5000 = 19.2` 是**名义数据遍历量**，不是 1500 次完整数据遍历。若 sampler 有放回采样，也不代表每条实例实际出现了 19.2 次。

协调端当前 `e1/native_worker.py` 按数据池尾部和验证曝光边界切分 batch。假设训练池 5000、曝光预算 96000、常规 batch64、每3200曝光验证一次，可以重现 1517 次更新：1483 个 batch64，另有 34 个不足64的小 batch。该推算不能代替远端实际日志，请从冻结日志确认切分与采样规则。

已报告的证据目录：

```text
CaliRoute/results/e1/exports/E1_native_baselines_20261017/
  E1_native_baseline_audit.md
  [两个证据压缩包：冻结源码、配置、日志、逐实例 test；不含大型 checkpoint]
```

协调机器尚无这个目录。协调机根目录是 `/data/Maojie/AAAI_EVRPTW`，A6000 可能是 `/data/Maojie/AAAI/AAAI_EVRPTW`；下面均使用相对 `CaliRoute/` 的路径。

## 3. 当前协调端实际已有代码

| 项目 | 路径 | 当前状态 |
|---|---|---|
| RADAR 宿主 | `benchmarks/native_sources/radar/acvrp/` | E1 vendored 代码，包含已有适配 |
| RRNCO 宿主 | `benchmarks/native_sources/rrnco/rrnco/` | E1 vendored 代码，包含已有适配 |
| 原生训练/评估适配 | `e1/native.py`、`e1/native_worker.py`、`e1/native_vrptw.py` | 需要与远端运行快照逐文件核对 |
| AGDA 核心 | `caliroute/plugins/agda.py::AdaptiveGraphAttention` | E1 核心配置为 action-key＋action-bias，K/V 修正关闭 |
| RADAR decoder 接口原型 | `e2/adapters/radar.py::RadarAGDADecoder` | 仅张量接口测试通过，尚未完成原生训练器接入 |
| 原型测试 | `tests/test_e2_radar_adapter.py` | 关闭/零初始化一致性、梯度和 mask 测试通过 |
| E2 协议检查 | `e2/protocol.py` | 初版；依赖真实协议元数据填充 |
| 三台准备 shell | `scripts/run_e2_*.sh` | 仅准备/检查；正式 E2 训练未启动 |

以上 E2 新文件随准备版代码提交到 `fix/e1-no-road-base-20261012` 分支。A6000 请先拉取该分支，并在报告中记录实际 `git rev-parse HEAD`；具体发布 commit 以协调端交接消息为准。若拉取后仍缺文件，请在报告中标明，不能将“未取得文件”写成“验证通过”。不要求为核查升级或覆盖现有训练环境。

协调端 vendored manifest 标识（不能仅用 commit 断言等同实际冻结源码）：

- RADAR upstream commit：`74645edb8fce5be2d5889bbb231265526a34de6a`。
- RADAR source manifest aggregate：`c3fd4de8ccd1621c3cd69b28870ae3c037c4510f31c6ce2ad87f3a7ec3cf1ecf`，**请以实际 `SOURCE_MANIFEST.json` 字段为准**。
- RRNCO upstream commit：`823d510dadf4dd711730ec4fbf337c356a0de6ae`。
- RRNCO source manifest aggregate：`3f43e3b6d24e0945d5c93297ceacf1518d14e396225ae63db965eb177a960172`。
- 两套 manifest 都标注了适配改动；回传逐文件哈希比只提供 git HEAD 更重要。

## 4. P0：先判定 E1 原生基线是否可复用

请直接利用已整理的证据，不必重跑训练。

### 4.1 实际配置与实现

请检查并回传：

- 原 run 路径、正式 config、manifest、冻结源码清单和哈希；区分 upstream commit、当前仓库 commit、实际运行源码。
- 转换后的 `native_data/*.npz` 及元数据哈希、原始 train/val/test 数据哈希、各 split 的 instance ID 列表或其哈希。
- 训练 sampler：有放回/无放回、shuffle、跨数据边界处理和 RNG 来源。协调端看到 RADAR 的固定顺序 shuffle，而 RRNCO generator 配置 `sample_with_replacement=True`；请以冻结源码为准确认。
- 实际 batch 序列、optimizer 更新数、曝光数、完整轨迹数、decision steps、LR 轨迹、weight decay、scheduler 调用顺序及里程碑。
- 原生 normalization、道路 D / 时间 T 的单位与转换、编码器邻域采样和数据增强设置。
- best checkpoint 的选择规则、验证频率及候选预算；不是只检查 best 的步数。
- 推理 decode mode、inference seed、test batch、candidate/start ID、是否有候选重复、decode cap、终止规则和隐式最后返仓的处理。
- 派生生成的 `instances.summary.json` 应保留派生时间、输入 JSONL 哈希、生成代码和与原记录一致性的检查结果。

特别注意：K=50 不等于采样策略一致。当前 RRNCO E1 使用 `multistart_greedy`；RADAR E1 的模型参数出现 `eval_type='softma'`，它与原生代码比较的 `'softmax'` 不同，可能实际进入 greedy 分支。请追踪实际调用确认，不能因字段名称看起来相似就记为 stochastic sampling。

### 4.2 复用结论要求

请分别给出：

1. **可直接作为 E2-A 匹配基线**：列出已满足条件和增强组必须沿用的字段。
2. **暂不能确认**：列出缺失文件或证据。
3. **需要补跑匹配基线**：指出哪个协议差异使旧基线不能复用。

不能只因为 seed、最终均值或总 epoch 数相同就判定可复用。增强组不能加载 E1 已训练 checkpoint 后追加训练，再与原 E1 预算比较。可复用时应从相同随机初始化规则开始，尽可能验证共享宿主参数的初始哈希；插件初始化消耗的 RNG 不能改变宿主参数初始化或采样序列。

## 5. P1：RADAR＋AGDA 接口具体需要核查什么

### 5.1 保留哪些宿主运算

`ACVRPModel.pre_forward` 对距离矩阵标准化后执行 `torch.svd_lowrank`，拼接 demand 再经过原 encoder。
`ACVRPModel_LIB.MixedScore_MultiHeadAttention` 内含 Sinkhorn。两者应原样保留。

原 decoder 保存 `k`、`v`、`single_head_key`，执行原生 glimpse，再计算：

```text
score = glimpse @ single_head_key
scaled = score / sqrt(embedding_dim)
clipped = logit_clipping * tanh(scaled)
probs = softmax(clipped + ninf_mask)
```

当前接口原型仅在该边界加入候选 action-key residual 和 action-bias，不动 native K/V/glimpse。

### 5.2 未确认项

- **特征构造**：从哪个 reset/step state 得到 directed distance row、反向边、depot distance、remaining load、remaining demand、visited 和 mask？逐项给出字段、shape、单位、归一化来源。
- **不活动资源**：CVRP 不应通过填假的时间、电量特征制造状态含义。可定义 task-specific feature schema，但要明确与 E1 AGDA 核心的对应关系。
- **注入顺序**：原型在 `score/sqrt(D)` 后、tanh clipping 前加 action-bias。请与冻结 E1 AGDA 的顺序/缩放对照；若改变，要说明是接口适配还是不同方法。
- **API 完整性**：原型尚未实现给 native decoder 的全部 `set_kv` / `set_q1` 代理，也未接 `model.pre_forward/forward` 的动态上下文传递。不能直接替换 decoder 后宣称原生 trainer 可运行。
- **参数注册**：应在 optimizer 创建前装配插件；每个插件参数被 optimizer 覆盖且只出现一次。
- **缓存失效**：SVD特征、宿主 node embeddings、AGDA projections 的生命周期是什么？权重更新后不能继续使用陈旧的参数相关缓存。
- **随机 SVD**：`torch.svd_lowrank` 消耗随机数。新模块初始化/计算不能无意改变基线随机序列。若用于 PPO 重算，需要在相同外生随机投影条件下重算；不能简单复用 detach 的旧 embeddings 代替当前策略。

### 5.3 请补的短测试

用真实 CVRP100 小 batch（例如2～4实例）测试，不需要跑1500轮：

- 插件关闭时，与实际原版的概率、固定动作 log-prob、mask、成本完全一致；报告最大误差。
- 插件零初始化时同上；检查共享宿主参数梯度一致。
- 插件打开且进行少量更新后，新参数有有限非零梯度，效果不是始终为零；被 mask 动作仍概率为零。
- 从 reset 到完整路线至少跑一次，并用独立 validator 重算物理道路成本。
- 保存/加载包含新模块的 checkpoint 后行为一致；报告额外参数量、显存峰值和耗时，仅作为工程预检。

## 6. P1：RRNCO＋AGDA 与外部 actor PPO 接口

### 6.1 原生结构和额外随机性

协调端 `RRNetEncoder` 实际调用 `AttnFreeNet`，包含 AAFM/NAB，不能按普通 attention encoder 强行插入 RDI。

当前 E1 的 policy 配置还包括：

```text
embed_dim=128, num_heads=8, num_encoder_layers=6
normalization='instance', use_graph_context=False, nab_type='gating'
init_embedding_kwargs:
  use_coords=True, use_polar_feats=True, use_dist=True
  use_matnet_init=False, sample_type='prob', sample_size=25
```

请检查 `sample_type='prob'` 的实际邻域/特征采样实现。它和 RADAR 的 SVD 一样，可能使同一状态多次编码不一致；PPO 不能把这类外生随机变化错误计成策略 KL。

### 6.2 Decoder 插入位置

当前 `RRNetDecoder` 有 `PrecomputedCache`、`_compute_q`、`_compute_kvl` 与 `RRNet_PointerAttention`。
VRPTW 还保留原生偏置：

```text
inductive_bias = alpha * distance_row + beta * duration_row
logits = log(exp(pointer_logits - inductive_bias) + 1e-6)
```

请给出 AGDA action-key/bias 应在哪个具体函数、哪一步插入。保留原生 alpha/beta 与其非线性，不能为方便适配而删除。若考虑数值稳定写法，也应单独验证等价性，避免混入未隔离的改进。

核查 `batchify/unbatchify` 顺序：当前代码会将 `[B,S,N]` 重排为 `[(S B),N]`。错误展平可能不报 shape 错误，却让不同实例/轨迹共享错误状态。

### 6.3 逐动作 PPO 的关键未确认项

请给出最小 actor 接口建议及代码位置：

```text
encode_static(instance/state) -> cache（随参数更新重建）
step_logits(state, cache, auxiliary_randomness) -> logits, feasible_mask
log_prob_given_actions(trajectory) -> 每动作 log_prob / entropy / valid-step mask
```

需要回答：

- 强制 depot/POMO start 的动作概率怎样记录？无参数选择不应伪造可学习 log-prob；PPO/GAE 的有效步规则如何一致处理？
- 不更新参数时，采样轨迹 old log-prob 与 replay log-prob 是否一致？逐步 ratio 是否接近1、KL是否接近0？请报告 FP32/实际 AMP 设置下误差，而不只报平均 loss。
- 动态 mask、归一化状态、采样邻域、候选顺序、浮点精度和随机数是否在 rollout/replay 一致？
- 是否有 dropout、BatchNorm 或其他与 train/eval 模式相关的操作？以真实运行配置为准，不假定论文默认配置。
- 一次 optimizer 更新后，重算是否确实依赖当前参数，而不是缓存旧 logits/embeddings？
- 轨迹变长、done 后 padding、首次 forced action、无合法动作或截断怎样处理？
- 复用 E1 逐动作 PPO trainer 所需的接口差异是什么？critic 如何读取同一 actor 的状态，并在四组中保持相同？
- AWBC/DAPG/SL-PPO 的完整专家轨迹如何 teacher-force；专家采样、概率精度、mask 与 rollout 一致性如何检查？

至少完成：同一 actor 的短程 PPO 更新与 SL-PPO 更新；之后才加入 AWBC/DAPG 并准备正式四组。当前尚未冻结 E2-B 的 PPO passes、GAE、LR、batch 和训练曝光预算，请提出一套四组匹配的配置，不能直接把 native 的1517步当作 PPO 已批准预算。

### 6.4 VRPTW 物理契约

对照 `e1/native_vrptw.py` 与 `e1/validator.py` 核查：

- D 用物理道路 km；T 用同一实例的道路行驶时间，不允许悄悄改为欧氏距离/固定比例。
- 时间窗与服务时间归一化在 state、mask 和 validator 中一致；服务开始和完成时刻的定义一致。
- 客户容量、depot 工作窗口、提前到达等待、最后一段返仓均正确。
- 原生 RRNCO 的隐式最后返仓如何纳入路线输出、道路成本与时窗验证。
- 使用与基线一致的 decode cap 和失败记录；不能为增强组额外增加预算或修复失败路线。

## 7. 建议回传形式

请在本机新建独立目录，例如：

```text
CaliRoute/results/e2/exports/RADAR_RRNCO_E2_review/
  review.md
  baseline_reuse_decision.json
  source_comparison.csv
  interface_map.csv
  smoke_results.json
  proposed_protocols/
    radar_cvrp100_native_agda.json
    rrnco_vrptw100_native_agda.json
    rrnco_vrptw100_step_ppo_quartet.json
  patches/                    # 若有建议修复，附独立补丁及对应源码哈希
  checksums.sha256
```

`review.md` 开头请直接回答四个结论：

1. RADAR E1 是否可复用为 RADAR＋AGDA 的 E2-A 基线？
2. RRNCO E1 是否可复用为 RRNCO＋AGDA 的 E2-A 基线？
3. RADAR 的 AGDA 原生训练接入还缺什么、哪些已实测？
4. RRNCO 能否可靠重算逐动作概率，PPO 四组还缺什么？

每项标为 `verified` / `conditional` / `not_tested` / `failed`，附实际文件、函数、哈希或测试日志路径。缺证据保持缺失，不用推测值补齐。

证据包可以不含大型 checkpoint。若需要由协调端复现，请附新模块/适配补丁、运行命令、Python/torch/rl4co/tensordict版本及环境锁定文件；checkpoint 哈希和选择记录即可先用于核查。

最后请将报告、证据包及 SHA256 清单同步到协调端能读取的位置，或回传可访问下载地址。仅给 A6000 的绝对路径，协调端无法读取。
