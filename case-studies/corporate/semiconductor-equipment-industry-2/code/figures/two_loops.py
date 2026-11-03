# 日本語の文字を描くため、Noto Sans CJK のフォントを使う。
# フォントのファイルの場所は、環境に合わせて書き換えること（下の fp と fpb）。
# 図は、このスクリプトを実行した場所に、jpgで保存される。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fpb='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fm.fontManager.addfont(fp); fm.fontManager.addfont(fpb)
plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
W,H=16,9
fig,ax=plt.subplots(figsize=(W,H),dpi=150); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); fig.patch.set_facecolor('white')
blue='#2E74B5'; teal='#1F9E89'; gray='#8A8A8A'; orange='#E07B39'; green='#3A9D5D'; dark='#222222'
r=0.62
def node(p,lab):
    ax.add_patch(Circle(p,r,fc='#EAF2FB',ec=blue,lw=2.0,zorder=5))
    ax.text(p[0],p[1],lab,ha='center',va='center',fontsize=17,color='#1B3A5C',zorder=6)
def edge(p,q,color,rad=0.0,lw=3.0,ls='-'):
    a=FancyArrowPatch(p,q,connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',mutation_scale=22,color=color,lw=lw,ls=ls,shrinkA=r*72*1.05/ (ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0])*0+46,shrinkB=46,zorder=3)
    ax.add_patch(a)
def label(x,y,t,color=dark,fs=15,box=True,fc='white',ec='#999999',weight='normal'):
    if box:
        ax.text(x,y,t,ha='center',va='center',fontsize=fs,color=color,fontweight=weight,zorder=8,
                bbox=dict(boxstyle='round,pad=0.35',fc=fc,ec=ec,lw=1.1))
    else:
        ax.text(x,y,t,ha='center',va='center',fontsize=fs,color=color,fontweight=weight,zorder=8)
# 区切り線
ax.plot([8,8],[0.6,8.4],color='#DDDDDD',lw=1.5)
# ---------- 左：基準ウェハー１枚 ----------
ax.text(4,8.35,'基準ウェハーが１枚：輪は１つ',ha='center',va='center',fontsize=22,color=dark,fontproperties=fm.FontProperties(fname=fpb))
F=np.array([4,6.6]); T=np.array([6.0,3.4]); U=np.array([2.0,3.4])
edge(F,T,teal,rad=-0.15); edge(T,U,teal,rad=-0.15); edge(U,F,teal,rad=-0.15)
node(F,'府中'); node(T,'台湾'); node(U,'米国')
label(4,4.85,'一周の差が\n0に戻らない',color=orange,fs=16,ec=orange)
label(4,1.75,'３つの区間は、どれも同じ１枚の基準ウェハーしか通らない。\nどの区間で変わったのかは、区別できない',fs=15,box=False)
# ---------- 右：基準ウェハー２枚 ----------
ax.text(12,8.35,'基準ウェハーがもう１枚：輪が２つ',ha='center',va='center',fontsize=22,color=dark,fontproperties=fm.FontProperties(fname=fpb))
F2=np.array([12,6.6]); T2=np.array([14.0,3.4]); U2=np.array([10.0,3.4])
# 大きな輪（１枚目）
edge(F2,T2,green,rad=-0.15,lw=3.0); edge(T2,U2,orange,rad=-0.15,lw=3.4); edge(U2,F2,orange,rad=-0.15,lw=3.4)
# 往復の輪（２枚目）：府中と台湾の間を、外側にずらした２本の矢印で往復
d=T2-F2; n=np.array([-d[1],d[0]])/np.linalg.norm(d)   # 外向き（右上）の垂線
o1=n*0.55; o2=n*1.05
edge(F2+o1,T2+o1,green,rad=0.0,lw=2.4,ls='--'); edge(T2+o2,F2+o2,green,rad=0.0,lw=2.4,ls='--')
node(F2,'府中'); node(T2,'台湾'); node(U2,'米国')
label(12,4.85,'１枚目の輪：\n一周の差が\n0に戻らない',color=orange,fs=13,ec=orange)
label(14.4,7.35,'２枚目：府中と台湾を往復する輪\n一周の差が0に戻った',color=green,fs=14,ec=green)
label(15.45,3.85,'府中と台湾の間では\n変わっていない',color=green,fs=13,ec=green)
label(12,1.75,'変わった区間は、台湾→米国、米国→府中の\nどちらかに絞り込める（オレンジの矢印）',fs=15,box=False)
fig.savefig('koryo2_04c_two_loops.jpg',dpi=150,facecolor='white',pil_kwargs={'quality':92})
print('ok')
