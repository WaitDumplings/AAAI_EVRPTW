# E3 第一批：五组 CVRP-Cus100 双卡实验

| 服务器 | GPU | 实验 |
|---|---|---|
| 2080ti_a（本机） | 0,1 | A0_no_bias |
| 2080ti_a（本机） | 2,3 | AE_euclidean_bias |
| 2080ti_3 | 0,1 | AR_road_bias |
| 2080ti_b | 0,1 | B0_static_candidates |
| 2080ti_b | 2,3 | B1_feature_matched_logit |

2080ti_3 的 GPU2 不使用。远端由用户启动，本机入口不会 SSH 或停止其他任务。

在仓库的 `CaliRoute` 目录执行，对应机器三选一：

```bash
bash scripts/run_e3_initial.sh 2080ti_a --execute
bash scripts/run_e3_initial.sh 2080ti_3 --execute
bash scripts/run_e3_initial.sh 2080ti_b --execute
```

不加 `--execute` 只显示分配。正式流程是 nohup + setsid，关闭终端后继续。
默认优先用仓库 `.venv/bin/python`，不存在则用当前 `python`；可设置 `E3_PYTHON`、`E3_DATA_ROOT`、`E3_RUN`。
已有目录不覆盖；中断后检查日志再用 `--resume`，只恢复冻结协议的 latest checkpoint。

## 共同协议

- seed3009、从头训练，1500 个在线采样轮次，不是1500遍训练集。
- 每组两进程、每卡32实例，全局batch64；每实例50条轨迹。
- 每轮4 PPO passes ×4 minibatches；FP32、LR=1e-4 constant、gamma=.99、GAE lambda=.95，与E1 PPO预算一致。
- 最多96,000实例曝光、4,800,000轨迹尝试、24,000次optimizer更新尝试；实际计数以日志为准。
- PPO step chunk默认16（2080Ti实测B1、A0、AE在chunk32时发生OOM，正式训练均未开始）。chunk仅控制重算内存块，不改变有效batch、候选数或更新数。
- 验证每50轮、1000实例、K50；按可行率优先、可行成本次之选择best。
- 完成后自动在独立1000条test上跑best、K50、batch8；覆盖/预算/成本校验后写验收标记并更新E3表。
- 预检单独跑2轮，全batch双卡，验证4实例K50；正式训练不加载预检权重。
- 输出源文件快照及哈希、配置哈希、数据哈希/ID、初始化哈希、GPU监控、训练/验证日志；checkpoint有独立备份目录。

## 明确的对照边界

本批采用原始 scalar road-distance bias RDI，常规graph attention encoder（256维、2层），不使用 joint graph encoder、物理摘要adapter、RDI-v2、额外探索或专家数据。

A0/AE/AR 完全保留同一 decoder distance-row 分支、query、mask；只有encoder bias不同。AE由公里坐标减去depot，再除同一固定标量后计算欧式距离；不得修改环境/decoder的道路矩阵。单位仅用train拟合，跨split冻结。

B0/B1及后续匹配B2使用同一个road-bias encoder和query。B系列全部关闭独立distance-row分支，因为它自身也改变候选action-key，不能出现在严格的static-candidates B0。B1/B2共享原始AGDA的30维candidate descriptors、state/decision token构造；B1直接MLP输出logit修正，B2修改候选投影。B1的head以实际使用的B2候选分支参数量匹配，256维下误差小于1%；共同token网络相同。

因此 **AR不能当作B0，E1 graph版不能直接当作本批B2**。本次按用户分配分别运行五组；后续B2需新增匹配训练。旧E1结果保留，不改名复用。

历史模型含未参与forward的参数。初始化清单同时报告注册参数、requires_grad参数和B1/B2有效候选head参数；不会用填充的无效参数宣称计算量相等。E3开关写入checkpoint模型签名，禁止跨消融配置resume。

E3-A近邻失配诊断提前固定：客户出边top10集合不一致率，按[0,1/3)、[1/3,2/3)、[2/3,1]分组。该分析不用于训练或checkpoint选择，待三组test齐备后执行。

## 查看进度

默认目录：`results/e3/E3_CVRP100_S3009_E1500_<server>_V1/`。本机实际为 `..._2080ti_a_V3/`，前两次分别在数据路径审计和显存预检阶段退出，没有启动正式训练。

- 同级 `<目录名>.supervisor.log` / `.supervisor.pid`：后台总控。
- `status.json`：各组阶段、最新训练/验证行。
- `<variant>/train.log` / `config.yaml` / `status.json`。
- `<variant>/preflight/accepted.json`：预检是否通过。
- `<variant>/test/instances.jsonl`、`instances.summary.json`、`manifest.json`：独立test与验收。
- `source/CaliRoute/`：冻结源码；`artifacts/`：日志与checkpoint。

显存占用随采样/重算阶段变化。保持有效batch一致，仅允许在新campaign中调整执行chunk；不为显存百分比增加某组训练预算。

远端启动前拉取 `fix/e1-no-road-base-20261012` 分支的 E3 提交。详细证据：`initialization_audit.json`、`B1_dual_preflight.json`；机器实测结果与正式训练指标分开保存。
