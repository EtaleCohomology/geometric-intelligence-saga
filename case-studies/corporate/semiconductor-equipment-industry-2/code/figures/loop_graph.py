# 日本語の文字を描くため、Noto Sans CJK のフォントを使う。
# フォントのファイルの場所は、環境に合わせて書き換えること（下の fp と fpb）。
# 図は、このスクリプトを実行した場所に、jpgで保存される。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp); plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
W,H=16,9
fig,ax=plt.subplots(figsize=(W,H),dpi=150); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
fig.patch.set_facecolor('white')
c=np.array([8.0,5.15]); R=2.9
ang={'府中':90,'台湾':-30,'米国':210}
pos={k:c+R*np.array([np.cos(np.radians(a)),np.sin(np.radians(a))]) for k,a in ang.items()}
blue='#2E74B5'; teal='#1F9E89'; gray='#555555'
# 矢印（円に沿って時計回り）
def arc(a0,a1):
    t=np.radians(np.linspace(a0,a1,60)); return c[0]+R*np.cos(t), c[1]+R*np.sin(t)
edges=[('府中','台湾',90,-30,'＋0.10 nm'),('台湾','米国',-30,-150,'−0.15 nm'),('米国','府中',210,90,'＋0.05 nm')]
node_r=0.78
for a,b,a0,a1,lab in edges:
    # ノードの円の外側から外側まで
    dA=np.degrees(node_r/R)*1.15
    s=a0-dA; e=a1+dA
    x,y=arc(s,e)
    ax.plot(x[:-3],y[:-3],color=teal,lw=2.6,solid_capstyle='round',zorder=1)
    ax.add_patch(FancyArrowPatch((x[-4],y[-4]),(x[-1],y[-1]),arrowstyle='-|>',mutation_scale=26,color=teal,lw=2.6,zorder=2))
    m=np.radians((a0+a1)/2); lx,ly=c+ (R+0.95)*np.array([np.cos(m),np.sin(m)])
    ax.add_patch(FancyBboxPatch((lx-1.05,ly-0.33),2.1,0.66,boxstyle='round,pad=0.05,rounding_size=0.18',fc='white',ec=gray,lw=1.2,zorder=4))
    ax.text(lx,ly,lab,ha='center',va='center',fontsize=21,color='#222222',zorder=5)
for k,p in pos.items():
    ax.add_patch(Circle(p,node_r,fc='#EAF2FB',ec=blue,lw=2.2,zorder=3))
    ax.text(p[0],p[1],f'{k}の\n計測装置',ha='center',va='center',fontsize=18,color='#1B3A5C',linespacing=1.25,zorder=4)
ax.text(W/2,0.22,'線の上の差は、測った数値から求める',ha='center',va='center',fontsize=22,color='#222222')
fig.savefig('koryo2_04b_loop_graph.jpg',dpi=150,facecolor='white',bbox_inches=None,pil_kwargs={'quality':92})
print('ok')
