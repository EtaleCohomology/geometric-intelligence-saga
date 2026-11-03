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
plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); B=fm.FontProperties(fname=fpb)
W,H=16,9
fig,ax=plt.subplots(figsize=(W,H),dpi=150); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); fig.patch.set_facecolor('white')
dark='#222222'; blue='#2E74B5'; orange='#E07B39'; red='#C0392B'; gray='#9A9A9A'
def wafer(x,y,lab,grown=False):
    ax.add_patch(Circle((x,y),0.42 if grown else 0.36,fc='#F3F3F3',ec=orange if grown else gray,lw=2.6 if grown else 1.8,zorder=4))
    ax.text(x,y,lab,ha='center',va='center',fontsize=14,color=dark,zorder=5,fontproperties=B)
def box(x,y,w,h,t,ec=blue,fc='#EAF2FB',fs=15,bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.06,rounding_size=0.15',fc=fc,ec=ec,lw=2,zorder=3))
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,color=dark,zorder=6,fontproperties=B if bold else None,linespacing=1.3)
def arr(p,q,c=gray,lw=2.4):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=18,color=c,lw=lw,zorder=2))
def panel(cx,title,grown,habit):
    ax.text(cx,8.35,title,ha='center',va='center',fontsize=17,color=dark,fontproperties=B,linespacing=1.3)
    # 府中
    box(cx-2.2,6.2,1.6,0.8,'府中')
    wafer(cx-2.55,5.1,'W1'); wafer(cx-1.85,5.1,'W3')
    # 輸送
    arr((cx-1.25,5.1),(cx+1.05,5.1))
    ax.text(cx-0.1,5.55,'運ぶ',ha='center',va='center',fontsize=14,color=dark)
    if grown: ax.text(cx-0.1,4.55,'途中で２枚とも\n0.1 nm 太くなる',ha='center',va='center',fontsize=13,color=orange,fontproperties=B)
    else: ax.text(cx-0.1,4.6,'ウェハーは\n変わらない',ha='center',va='center',fontsize=13,color=gray)
    # 台湾
    box(cx+2.2,6.2,1.6,0.8,'台湾')
    wafer(cx+1.85,5.1,'W1',grown); wafer(cx+2.55,5.1,'W3',grown)
    box(cx+2.2,3.55,2.6,0.95,'台湾の装置\n'+('クセ：＋0.1 nm' if habit else 'クセ：なし'),ec=red if habit else blue,fc='#FDECEA' if habit else '#EAF2FB',fs=14)
    arr((cx+2.2,4.6),(cx+2.2,4.1),c=gray,lw=2)
    # 測った値
    box(cx+0.0,1.95,5.6,0.95,'測った値：W1も W3も ＋0.1 nm 大きく出る',ec='#555555',fc='white',fs=15,bold=True)
    arr((cx+2.2,3.05),(cx+1.4,2.45),c='#555555',lw=2)
panel(4.0,'場合１：運ぶ途中で、W1もW3も\nそろって 0.1 nm 太くなった',True,False)
panel(12.0,'場合２：ウェハーは変わらない。\n台湾の装置が 0.1 nm 大きく測るクセを持つ',False,True)
ax.plot([8,8],[1.3,8.8],color='#DDDDDD',lw=1.5)
ax.text(8,1.95,'＝',ha='center',va='center',fontsize=34,color=dark,fontproperties=B,
        bbox=dict(boxstyle='circle,pad=0.15',fc='white',ec='white'))
ax.text(8,0.6,'測った値は、どちらの場合もまったく同じ。だから、ウェハーがそろって変わったのか、台湾の装置のクセなのかは、見分けられない',ha='center',va='center',fontsize=15,color=dark)
fig.savefig('koryo2_05c_cannot_tell.jpg',dpi=150,facecolor='white',pil_kwargs={'quality':92})
print('ok')
