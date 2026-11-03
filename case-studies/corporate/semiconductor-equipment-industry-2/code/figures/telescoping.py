# 日本語の文字を描くため、Noto Sans CJK のフォントを使う。
# フォントのファイルの場所は、環境に合わせて書き換えること（下の fp と fpb）。
# 図は、このスクリプトを実行した場所に、jpgで保存される。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fpb='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fm.fontManager.addfont(fp); fm.fontManager.addfont(fpb)
plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); B=fm.FontProperties(fname=fpb)
W,H=16,9
fig,ax=plt.subplots(figsize=(W,H),dpi=150); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); fig.patch.set_facecolor('white')
dark='#222222'; F='#2E74B5'; T='#E07B39'; U='#3A9D5D'; gray='#888888'
def seq(y,items,x0=0.6,fs=17):
    """items: (文字, 色, 打ち消し線) を左から並べる"""
    x=x0
    for s,c,strike in items:
        t=ax.text(x,y,s,ha='left',va='center',fontsize=fs,color=c,fontproperties=B if c!=dark else None)
        fig.canvas.draw(); bb=t.get_window_extent(); inv=ax.transData.inverted()
        (x1,_),(x2,_)=inv.transform([(bb.x0,bb.y0),(bb.x1,bb.y1)])
        if strike: ax.plot([x1,x2],[y-0.02,y+0.05],color='#C0392B',lw=2.2)
        x=x2+0.05
    return x
ax.text(W/2,8.4,'一周の差を足し合わせると、台湾と米国の値が打ち消し合う',ha='center',va='center',fontsize=22,color=dark,fontproperties=B)
ax.text(0.6,7.45,'① 府中 → 台湾 → 米国 → 府中 と運び、線ごとの差を足し合わせる',ha='left',va='center',fontsize=16,color=dark)
seq(6.75,[('（',dark,0),('台湾の値',T,0),(' − ',dark,0),('府中（送り出す前）の値',F,0),('）',dark,0)])
seq(6.1,[('＋（',dark,0),('米国の値',U,0),(' − ',dark,0),('台湾の値',T,0),('）',dark,0)])
seq(5.45,[('＋（',dark,0),('府中（戻ってきた後）の値',F,0),(' − ',dark,0),('米国の値',U,0),('）',dark,0)])
ax.text(0.6,4.45,'② 括弧を外して、同じ拠点の値どうしを並べ替える',ha='left',va='center',fontsize=16,color=dark)
xe=seq(3.7,[('− ',dark,0),('府中（送り出す前）の値',F,0),('　＋（',dark,0),('台湾の値',T,1),(' − ',dark,0),('台湾の値',T,1),('）',dark,0),
            ('　＋（',dark,0),('米国の値',U,1),(' − ',dark,0),('米国の値',U,1),('）',dark,0),('　＋ ',dark,0),('府中（戻ってきた後）の値',F,0)],fs=15)
ax.text(6.7,3.05,'＝ 0',ha='center',va='center',fontsize=16,color='#C0392B',fontproperties=B)
ax.text(9.65,3.05,'＝ 0',ha='center',va='center',fontsize=16,color='#C0392B',fontproperties=B)
ax.add_patch(FancyBboxPatch((0.6,0.75),14.8,1.45,boxstyle='round,pad=0.08,rounding_size=0.2',fc='#EAF2FB',ec=F,lw=2))
ax.text(8,1.75,'③ 残るのは「府中（戻ってきた後）の値 − 府中（送り出す前）の値」＝ 一周の差',ha='center',va='center',fontsize=17,color=dark,fontproperties=B)
ax.text(8,1.15,'台湾と米国の値は、プラスとマイナスで１回ずつ入るので消える。計測装置のクセが、どれだけ大きくても同じ',ha='center',va='center',fontsize=15,color=dark)
fig.savefig('koryo2_03b_telescoping.jpg',dpi=150,facecolor='white',pil_kwargs={'quality':92})
print('ok')
