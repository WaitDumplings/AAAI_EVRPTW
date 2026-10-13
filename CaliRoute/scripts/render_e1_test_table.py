from pathlib import Path
import csv,html,hashlib,argparse
parser=argparse.ArgumentParser()
parser.add_argument("--problem",choices=("CVRP","VRPTW"),default="CVRP")
problem=parser.parse_args().problem
prefix=problem.lower()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from matplotlib.font_manager import FontProperties

out=Path(__file__).resolve().parents[1]/'results/e1/reports/visualizations'
out.mkdir(parents=True,exist_ok=True)
source=out.parent/'CVRP_VRPTW_test_overview_20261013.csv'
rows=[r for r in csv.DictReader(source.open()) if r['problem']==problem]
methods=list(dict.fromkeys(r['method'] for r in rows));scales=[15,50,100]
data={(r['method'],int(r['customers'])):r for r in rows}
values=lambda n:{m:float(data[m,n]['all_feasible_mean_km']) for m in methods[:-1] if data[m,n]['all_feasible_mean_km']!='TBD'}
winners={n:min(values(n),key=values(n).get) if values(n) else None for n in scales}
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
bold=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
ink='#142a3a';muted='#627588';teal='#087e78';remote_color='#4275a6'
def cell(m,n):
 r=data[m,n];v=r['all_feasible_mean_km'];remote='remote' in r['status'].lower()
 if v=='TBD':return ('训练中' if 'training' in r['status'].lower() and 'completed' not in r['status'].lower() else '待 test'),False,False
 return f'{float(v):.4f}'+(' *' if remote else ''),remote,m==winners[n]

fig=plt.figure(figsize=(12,8.8),dpi=180,facecolor='#f4f7fa');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1200);ax.set_ylim(0,880);ax.axis('off')
def text(x,y,s,size=13,color=ink,weight=False,ha='left'):
 ax.text(x,y,s,fontproperties=bold if weight else font,fontsize=size,color=color,ha=ha,va='center')
text(60,828,f'{problem} · 独立 Test 结果',24,weight=True)
text(60,787,'平均成本（km）↓  ·  数值越低越好',12,muted)
text(1140,817,'2026-10-13',12,muted,ha='right')
ax.add_patch(FancyBboxPatch((50,166),1100,570,boxstyle='round,pad=0,rounding_size=14',facecolor='white',edgecolor='#dfe7ee'))
left=70; width=1060; top=717; header_height=51;row_h=59
centers=[590,805,1020]
text(90,top-10,'方法',13,weight=True)
for x,n in zip(centers,scales):text(x,top-10,f'Cus{n}',15,weight=True,ha='center')
ax.plot([75,1125],[676,676],color='#dfe7ee',lw=1)
for i,m in enumerate(methods):
 y=645-i*row_h
 if m.startswith('Gurobi'):
  ax.add_patch(Rectangle((71,y-28),1058,55,facecolor='#edf2f6',edgecolor='none'))
  label='Gurobi（2 h，1 thread）'
 else:label=m.replace('PPO (base)','PPO base（无路网输入）').replace('PPO (RDI+AGDA)','PPO + RDI + AGDA')
 text(90,y,label,13,weight=m=='PPO (RDI+AGDA)')
 for x,n in zip(centers,scales):
  s,remote,win=cell(m,n)
  if win:
   ax.add_patch(FancyBboxPatch((x-87,y-22),174,44,boxstyle='round,pad=0,rounding_size=8',facecolor='#e3f4ef',edgecolor='none'))
  color=teal if win else remote_color if remote else muted if data[m,n]['all_feasible_mean_km']=='TBD' else ink
  text(x,y,s,15 if data[m,n]['all_feasible_mean_km']!='TBD' else 12,color,weight=win,ha='center')
 if i<6:ax.plot([90,1110],[y-29,y-29],color='#f0f3f7',lw=.7)
text(65,124,'绿色：该规模已收集模型结果中的最低成本（不含 Gurobi）。',11,teal)
text(65,94,'* 远端回报，逐实例文件尚未在本机核验；未带 * 的已知数值已有本机核验记录。',10,muted)
text(65,66,'Gurobi 为 2 小时参考解，包含超时 incumbent；并非全部已证明最优。',10,muted)
text(65,38,'表中仅展示 test；未完成结果不参与排名，不计算共同可行子集 Gap。',10,muted)
fig.savefig(out/f'{prefix}_test_table_20261013.png',dpi=180)
fig.savefig(out/f'{prefix}_test_table_20261013.pdf')
plt.close(fig)
trs=[]
for m in methods:
 label=m.replace('PPO (base)','PPO base（无路网输入）').replace('PPO (RDI+AGDA)','PPO + RDI + AGDA')
 tds=[]
 for n in scales:
  s,remote,win=cell(m,n);r=data[m,n]
  css='best' if win else 'remote' if remote else 'pending' if r['all_feasible_mean_km']=='TBD' else ''
  tds.append(f'<td class="{css}" title="{html.escape(r["status"],quote=True)}">{html.escape(s)}</td>')
 trs.append('<tr'+(' class="reference"' if m.startswith('Gurobi') else '')+'><th scope="row">'+html.escape(label)+'</th>'+''.join(tds)+'</tr>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CVRP Test 结果</title><style>
body{font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:#f4f7fa;color:#142a3a;margin:0;padding:40px}main{max-width:1080px;margin:auto}h1{margin-bottom:10px}p,footer{color:#627588;line-height:1.9}.table-wrap{overflow-x:auto;background:white;border:1px solid #dfe7ee;border-radius:14px;margin-top:28px}table{border-collapse:collapse;width:100%;min-width:680px}th,td{padding:20px;text-align:center;font-variant-numeric:tabular-nums}thead{background:#edf2f6}tbody th{text-align:left;font-weight:500}tbody tr{border-top:1px solid #edf2f6}td{font-size:19px}.best{color:#087e78;background:#e3f4ef;font-weight:750}.remote{color:#4275a6}.pending{font-size:15px;color:#627588}.reference{background:#edf2f6}footer{font-size:13px;margin-top:22px}a{color:#087e78}@media print{body{background:white;padding:0}main{max-width:none}td,th{padding:14px}} </style><main><h1>CVRP · 独立 Test 结果</h1><p>平均成本（km）↓ · 数值越低越好 · 2026-10-13</p><div class="table-wrap"><table><thead><tr><th>方法</th><th>Cus15</th><th>Cus50</th><th>Cus100</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div><footer>绿色：该规模已收集模型结果中的最低成本，不含 Gurobi。<br>* 远端回报，逐实例文件尚未在本机核验。悬停单元格可查看详细状态。<br>Gurobi 为 2 小时参考解，包含超时 incumbent，并非全部已证明最优。<br>仅包含 test；未完成结果不参与排名，共同可行子集 Gap 暂不计算。<br><a href="cvrp_test_table_20261013.png">PNG 图片</a> · <a href="cvrp_test_table_20261013.pdf">PDF</a> · <a href="../CVRP_VRPTW_test_overview_20261013.csv">来源 CSV</a></footer></main></html>'''
page=page.replace('CVRP ·',problem+' ·').replace('CVRP Test',problem+' Test').replace('cvrp_test_table',prefix+'_test_table')
(out/f'{prefix}_test_table_20261013.html').write_text(page)
print('Winners',winners)
print(out/f'{prefix}_test_table_20261013.png')
