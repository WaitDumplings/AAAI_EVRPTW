"""Render the E3 planned contrasts; no training or evidence reuse is implied."""
from pathlib import Path
import csv,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'docs/e3/E3_experiment_table.csv'
with source.open() as f:rows=list(csv.DictReader(f))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
bold=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
fig,ax=plt.subplots(figsize=(18,17),dpi=180)
fig.patch.set_facecolor('#f3f6fb');ax.set_position([0,0,1,1]);ax.set_xlim(0,1800);ax.set_ylim(0,1700);ax.axis('off')
texts=[]
def txt(x,y,s,size=12,color='#18344b',strong=False):
 t=ax.text(x,y,s,fontsize=size,color=color,fontproperties=bold if strong else font,va='center');texts.append(t);return t

txt(45,1653,'E3 · 机制消融实验与结果占位表',26,strong=True)
txt(45,1608,'自有 backbone  |  统一 Cus100  |  单 training seed（计划 3009，与 E1 核对）  |  Test 1,000 · K=50',13,color='#526c80')
txt(45,1576,'当前安排：CVRP-Cus100 的 A0 / AE / AR / B0 / B1，五组各双卡；其他配置待后续安排。',13,color='#526c80')
txt(45,1544,'E3-A 保留 decoder road-row；E3-B 使用严格静态候选对照，B2 需补跑匹配版本。',12,color='#526c80')
xs=[55,165,510,990,1195,1370,1535,1660]
labels=['任务','配置','改变什么 / 要回答的问题','训练安排','批次','服务器','可行数','成本 km ↓']
y=1493
section={
 'A':('E3-A · 道路关系：为什么需要 RDI？','AGDA / SL-PPO 关闭；同一 PPO 与真实道路环境，只改变 encoder 距离表示。'),
 'B':('E3-B · 动态信息：AGDA 是否优于同信息量 MLP？','RDI 开启、SL-PPO 关闭；关键比较 B2 vs B1。EVRPTW 首轮至少 B1 / B2。'),
 'C':('E3-C · 完整解学习：成本、轨迹和 incumbent 的贡献','固定 RDI+AGDA actor；自身 incumbent 仅存成本。关键比较 C4 vs C3，以及 C4 vs C5。')}
for panel in 'ABC':
 title,subtitle=section[panel]
 txt(45,y,title,18,strong=True);y-=30;txt(45,y,subtitle,12,color='#526c80');y-=38
 ax.add_patch(Rectangle((40,y-19),1720,38,color='#193b60',linewidth=0))
 for x,label in zip(xs,labels):txt(x,y,label,11,'white',True)
 for i,r in enumerate(row for row in rows if row['panel']==panel):
  y-=40;ax.add_patch(Rectangle((40,y-20),1720,40,color='white' if i%2==0 else '#e8eff7',linewidth=0))
  vals=[r['task'],r['config_id'],r['comparison_purpose'],r['training_plan'],r['batch'],r['server'],r['feasible_count'],r['mean_cost_km']]
  for j,(x,value) in enumerate(zip(xs,vals)):
   color='#168064' if value=='新增' else '#96621a' if value=='条件复用' else '#8996a5' if value=='TBD' else '#18344b'
   txt(x,y,value,11,color,j==0)
 y-=52
notes=[
 '复用核查：A0 的 decoder 信息必须一致；AR 只有在 query/context 等非候选路径一致时才可兼作 CVRP-B0。',
 '复用失败需补跑匹配基线；当前 E1 的具体 RDI / AGDA 结构、预算与评估协议均需核对，不按名称直接复用。',
 '配套诊断：A 做近邻失配分组与收益图；B 报参数量/推理耗时并检查能量压力；C 画当前 policy 的 validation 曲线。',
 '本轮不新增 VRPTW-C 系列、EVRPTW-B0 或完整 factorial；不重生成 test，不重新跑整套 Gurobi。',
 'TBD 为未验收结果；完整实验定义、复用来源、证据路径和结果字段见同目录 CSV / 说明。']
for note in notes:txt(45,y,note,11,color='#526c80');y-=27
fig.canvas.draw()
renderer=fig.canvas.get_renderer();canvas=fig.bbox
for t in texts:
 box=t.get_window_extent(renderer)
 if box.x0<0 or box.y0<0 or box.x1>canvas.width or box.y1>canvas.height:raise RuntimeError('Text outside canvas: '+t.get_text())
fig.savefig(source.with_suffix('.png'),dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
columns=[('panel','面板'),('task','任务'),('customers','规模'),('config_id','配置'),('comparison_purpose','目的'),('training_plan','安排'),('batch','批次'),('reuse_candidate','复用候选'),('server','服务器'),('status','状态'),('feasible_count','可行数'),('mean_cost_km','成本 km'),('parameter_count','参数量'),('inference_s_per_instance','推理 s/实例')]
body='<tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr>'
for r in rows:body+='<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k,_ in columns)+'</tr>'
source.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E3 实验表</title><style>body{font:15px system-ui;background:#f3f6fb;color:#18344b;padding:28px}table{border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #dde5ef;white-space:nowrap}th{background:#193b60;color:white}tr:nth-child(even){background:#e8eff7}</style><h1>E3 · 机制消融实验</h1><p>自有 backbone · Cus100 · 单 seed；当前首批五组各双卡；其他复用均需协议核查。</p><p><a href="E3_experiment_table.png">查看 PNG</a> · <a href="E3_experiment_table.csv">CSV</a> · <a href="E3_experiment_notes.md">实验说明</a></p><div style="overflow:auto"><table>'+body+'</table></div><p>20个对照位置不等于20次训练；服务器与阶段见 CSV；TBD 为尚未验收结果。</p></html>')
print(source.with_suffix('.png'))
