"""光稜精機（架空）の計測装置の合わせ込み：答えを仕込んだデータでの確認
装置：府中F1,F2 台湾T1,T2 米国U1,U2 上海S1,S2（F1を基準=0とする）
基準ウェハー：W1 府中→台湾→府中, W2 府中→米国→府中, W3 府中→台湾→米国→府中, W4 府中→上海→府中
仕込んだ答え：(a) W3は台湾→米国の輸送で +0.25nm 変化  (b) F2は40日目の部品交換で +0.18nm クセが変化
各拠点で、届いたウェハーをその拠点の2台で測る（1回の測定は20点の平均。ばらつき 0.02nm）"""
import numpy as np
rng=np.random.default_rng(7)
tools=['F1','F2','T1','T2','U1','U2','S1','S2']; ti={t:i for i,t in enumerate(tools)}
off=dict(F1=0.0,F2=0.12,T1=-0.08,T2=0.05,U1=0.20,U2=-0.10,S1=0.15,S2=-0.05)
site={'F':['F1','F2'],'T':['T1','T2'],'U':['U1','U2'],'S':['S1','S2']}
routes={'W1':['F','T','F'],'W2':['F','U','F'],'W3':['F','T','U','F'],'W4':['F','S','F']}
true_cd={'W1':32.10,'W2':31.85,'W3':32.40,'W4':31.95}
change={('W3','T','U'):0.25}
sd=0.02
# 測定の記録：(ウェハー, 区間番号, 拠点, 装置, 日, 値)
rec=[]; day=0
for w,r in routes.items():
    level=0.0
    for k,s in enumerate(r):
        if k>0: level+=change.get((w,r[k-1],s),0.0)
        for t in site[s]:
            o=off[t]+(0.18 if (t=='F2' and day>=40) else 0.0)
            rec.append((w,k,s,t,day,true_cd[w]+level+o+rng.normal(0,sd)))
        day+=6
# ---------- いまのやり方：ウェハーごとに、測った全装置の中央値との差を補正値とし、装置ごとに平均 ----------
med={}
for w in routes:
    vals=[v for (ww,k,s,t,d,v) in rec if ww==w]; m=np.median(vals)
    for (ww,k,s,t,d,v) in rec:
        if ww==w: med.setdefault(t,[]).append(v-m)
med_off={t:np.mean(v) for t,v in med.items()}; base=med_off['F1']
print('■ いまのやり方（中央値との差）による補正値（F1基準, nm）')
for t in tools: print(f'  {t}: 推定 {med_off[t]-base:+.3f}   正解 {off[t]:+.3f}   誤差 {med_off[t]-base-off[t]:+.3f}')
# ---------- 一周の差：同じウェハーを府中で出発前と帰着後に、同じ装置F1で測った差 ----------
print('\n■ 一周の差（府中F1で、出発前と帰着後の差, nm）')
for w,r in routes.items():
    a=[v for (ww,k,s,t,d,v) in rec if ww==w and k==0 and t=='F1'][0]
    b=[v for (ww,k,s,t,d,v) in rec if ww==w and k==len(r)-1 and t=='F1'][0]
    z=(b-a)/(sd*np.sqrt(2)); print(f'  {w} {"→".join(r)}: {b-a:+.3f}  （z={z:+.1f}）')
# ---------- 一周の差が有意なW3について、どの区間で変わったかを、ほかのウェハーと組み合わせて推定 ----------
# モデル：値 = ウェハーの真値 + 区間ごとのウェハーの変化(W3の各区間) + 装置のクセ(+F2の段差) + ばらつき
cols=[f'cd_{w}' for w in routes]+[f'off_{t}' for t in tools[1:]]+['F2_step','W3_FT','W3_TU','W3_UF']
X=[];y=[]
for (w,k,s,t,d,v) in rec:
    row=np.zeros(len(cols)); row[cols.index(f'cd_{w}')]=1
    if t!='F1': row[cols.index(f'off_{t}')]=1
    if t=='F2' and d>=40: row[cols.index('F2_step')]=1
    if w=='W3':
        r=routes['W3']
        for kk in range(1,k+1):
            row[cols.index({('F','T'):'W3_FT',('T','U'):'W3_TU',('U','F'):'W3_UF'}[(r[kk-1],r[kk])])]=1
    X.append(row); y.append(v)
X=np.array(X); y=np.array(y)
rank=np.linalg.matrix_rank(X); print(f'\n■ 推定できるか：未知数 {len(cols)}, 行列の階数 {rank}')
# W3の区間の変化は3区間の和しか決まらない（ほかのウェハーが同じ区間を通らない）→ 和を1つの区間に割り振る候補を比べる
best=None
for keep in ['W3_FT','W3_TU','W3_UF']:
    cc=[c for c in cols if not c.startswith('W3_')]+[keep]
    Xs=X[:,[cols.index(c) for c in cc]]
    beta,res,rk,_=np.linalg.lstsq(Xs,y,rcond=None); rss=np.sum((y-Xs@beta)**2)
    print(f'  変化を {keep} に置いた場合：残差平方和 {rss:.5f}')
    if best is None or rss<best[0]: best=(rss,keep,cc,beta)
rss,keep,cc,beta=best
print(f'  → 最もよく説明するのは {keep}（変化量 {beta[cc.index(keep)]:+.3f} nm, 正解 +0.250）')
print('■ 推定した装置のクセ（F1基準, nm）')
for t in tools[1:]:
    print(f'  {t}: 推定 {beta[cc.index("off_"+t)]:+.3f}   正解 {off[t]:+.3f}')
print(f'  F2の部品交換後の段差: 推定 {beta[cc.index("F2_step")]:+.3f}   正解 +0.180')
