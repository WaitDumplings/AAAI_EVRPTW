# E3 实验清单说明

来源为用户本轮提供的执行说明，原文保存于 [user_execution_specification.txt](user_execution_specification.txt)。初版为结果占位表；当前已安排首批五组 CVRP100 双卡任务，执行配置及对照边界以 [第一批启动说明](E3_first_five_launch.md) 为准。

查看：[PNG 表](E3_experiment_table.png) · [HTML 表](E3_experiment_table.html) · [完整 CSV](E3_experiment_table.csv)。

## 范围与数量

统一使用自有 backbone、Cus100、单 training seed。表中暂列 seed3009、test1000、K50，启动前与拟复用 E1 run 的实际配置核对。首批五组已冻结1500在线轮次、全局batch64、K50、4次PPO pass、FP32、不使用专家。

| 部分 | CVRP | VRPTW | EVRPTW | 确定计划新增数（复用成立时） |
|---|---|---|---|---:|
| A：道路关系 | A0 / AE / AR | — | — | 2 |
| B：信息利用 | B0 / B1 / B2 | B0 / B1 / B2 | B1 / B2 | 4 |
| C：完整解学习 | C0–C5 | — | C2 / C3 / C4 | 6 |

原计划在复用全部成立时是12个新增。本轮核查后A0、B0需要匹配重跑，且本批B系列不能复用E1 graph版B2；新增数量随协议核查更新，不再固定为12。对照位置不是独立run计数：CVRP-B2和C0可指向同一个合格E1结果；CVRP-AR仅在接口匹配时兼作B0。

原始建议首批为7个配置；用户当前实际安排为A0、AE、AR、B0、B1五组，C系列尚未安排。之后扩展VRPTW的B0/B1，以及EVRPTW的B1/C2/C3，共5项。12项是所有复用成立时的新增数；匹配失败需增加对应基线或对照，不能强行按名称复用。

## A：道路关系

同一真实道路环境、reward、可行性、最终道路成本和PPO训练协议；AGDA、SL-PPO关闭，仅改变encoder的距离bias。

- A0无encoder bias；AE采用物理几何Euclidean bias；AR采用原始RDI road-distance bias。
- A0仍保留匹配的decoder/query/mask。此前E1的“无道路输入PPO base”可能移除了decoder距离信息，因此不能未经核查直接复用。
- Euclidean距离从物理几何构建，明确单位和标量归一化；不能用逐轴min-max拉伸后的坐标当作真实Euclidean几何。
- 需提前固定近邻失配指标、邻居数、并列处理、分组阈值规则，再对固定test计算低/中/高失配组的AR相对AE收益；不能看到收益后重新分组，也不预设趋势一定单调。

## B：同信息量对照

RDI开启，SL-PPO关闭。B0动态query+静态候选；B1用与B2相同信息的MLP给出logit correction；B2用原始AGDA候选修正。关键比较为B2对B1。

B1/B2须匹配candidate descriptors、车辆状态、全局上下文、query、mask及非候选路径；尽量匹配可训练参数量并报告实际差异。B1不能只给少量原始特征，而B2额外读取上下文。

AR不自动等于B0。现有E1可能采用graph RDI或后来改进的AGDA，需先确定本轮的具体模块版本；不能将不同版本仅凭名称视为同一机制。B2复用必须同时满足actor、训练预算、初始化与评估协议匹配。

EVRPTW先统计真实充电访问及电量余量。若能量约束少有作用，仅报告该数据上的结果，不能据此声称验证了充电决策机制。energy-stress数据属于另行补充，不替换原test。

## C：成本和轨迹的增量

所有组保留同一RDI+AGDA actor及PPO基础，只改变解级目标与信息使用。

| 配置 | Policy solution objective | 自身incumbent成本 | 外部成本 | 外部轨迹 |
|---|---|---|---|---|
| C0 | 无 | 不参与训练 | 无 | 无 |
| C1 | 仅group comparison | 关闭 | 无 | 无 |
| C2 | group + 可用reference | 开启 | 无 | 无 |
| C3 | 与完整版本一致 | 开启 | 开启 | 关闭 |
| C4 | 完整版本 | 开启 | 开启 | 原expert更新形式 |
| C5 | 同C4 | 开启 | 开启 | 同轨迹/门控/权重，改为长度归一化weighted BC |

C1–C0看解级目标；C2–C1看自身最好成本；C3–C2看外部成本；C4–C3看对应轨迹的增量；C4–C5区分经验选择/权重与expert更新形式。C5不是E1 AWBC或DAPG的替代品，也不修改正式SL-PPO。

信息开关必须贯穿整个数据流：group_only不把incumbent纳入group statistics；no_external不读专家成本；external_cost_only不读取专家轨迹计算训练损失。自身incumbent始终只存成本，不新增self-replay。专家仅来自train。

输出当前policy的validation曲线，并记录reference来源、incumbent更新、replay可用率、门控通过率和实际权重；不能只画best-so-far缓存曲线。

## 表格更新

所有结果和服务器当前保留TBD，未将E1数字提前当作已验收E3结果。CSV预留参数量、推理时间、训练时间、run/config/checkpoint/逐实例路径。正式结果按验证集选出的best checkpoint独立test，不用test做开发或模型选择。

修改CSV后，在CaliRoute运行：

```bash
../.venv/bin/python scripts/render_e3_table.py
```

首轮不扩展Cus15/50、多seed、所有K/V分支、全部factorial或历史专家质量扫描。关键结论后续再补其他seeds；无需重生成test或重跑整套Gurobi。
