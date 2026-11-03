# 日本語の文字を描くため、Noto Sans CJK のフォントを使う。
# フォントのファイルの場所は、環境に合わせて書き換えること（下の fp と fpb）。
# 図は、このスクリプトを実行した場所に、jpgで保存される。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fpb='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fm.fontManager.addfont(fp); fm.fontManager.addfont(fpb)
plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
B=fm.FontProperties(fname=fpb)
W,H=16,9
fig,ax=plt.subplots(figsize=(W,H),dpi=150); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); fig.patch.set_facecolor('white')
P={'府中':np.array([8.0,4.6]),'上海':np.array([3.2,7.0]),'台湾':np.array([3.2,2.2]),'米国':np.array([12.8,4.6])}
r=0.62
col={'W1':'#E07B39','W2':'#2E74B5','W3':'#C0392B','W4':'#3A9D5D'}
def lane(x,y,c,off):
    """府中とXの組ごとに決まった向きの垂線で、車線のように平行にずらす"""
    F=P['府中']; X=P[x if x!='府中' else y]
    d=X-F; n=np.array([-d[1],d[0]])/np.linalg.norm(d); u=d/np.linalg.norm(d)
    a=P[x]+n*off; b=P[y]+n*off
    s=1 if x=='府中' else -1
    a=a+s*u*(r+0.12); b=b-s*u*(r+0.12)
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=20,color=c,lw=2.8,zorder=3))
# 府中と台湾：W1の行き、W1の帰り、W3の行き
lane('府中','台湾',col['W1'],0.40); lane('台湾','府中',col['W1'],0.10); lane('府中','台湾',col['W3'],-0.22)
# 府中と米国：W2の行き、W2の帰り、W3の帰り
lane('府中','米国',col['W2'],0.32); lane('米国','府中',col['W2'],0.0); lane('米国','府中',col['W3'],-0.32)
# 府中と上海：W4の行き、W4の帰り
lane('府中','上海',col['W4'],0.16); lane('上海','府中',col['W4'],-0.16)
# W3：台湾から米国（下を大きく回る）
ax.add_patch(FancyArrowPatch(P['台湾']+np.array([0.58,-0.25]),P['米国']+np.array([-0.30,-0.56]),connectionstyle='arc3,rad=0.28',arrowstyle='-|>',mutation_scale=20,color=col['W3'],lw=2.8,zorder=3))
for k,p in P.items():
    ax.add_patch(Circle(p,r,fc='#EAF2FB',ec='#2E74B5',lw=2.2,zorder=5))
    ax.text(p[0],p[1],k,ha='center',va='center',fontsize=19,color='#1B3A5C',zorder=6,fontproperties=B)
ax.text(W/2,8.45,'４枚の基準ウェハーの経路（案B）',ha='center',va='center',fontsize=24,color='#222222',fontproperties=B)
# 凡例
L=[('W1','府中 → 台湾 → 府中'),('W2','府中 → 米国 → 府中'),('W3','府中 → 台湾 → 米国 → 府中'),('W4','府中 → 上海 → 府中')]
x0,y0=10.4,7.55
for i,(w,t) in enumerate(L):
    y=y0-0.55*i
    ax.plot([x0,x0+0.7],[y,y],color=col[w],lw=4,solid_capstyle='round')
    ax.text(x0+0.9,y,f'{w}：{t}',ha='left',va='center',fontsize=16,color='#222222')
fig.savefig('koryo2_05b_wafer_routes.jpg',dpi=150,facecolor='white',pil_kwargs={'quality':92})
print('ok')
