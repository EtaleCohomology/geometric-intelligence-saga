"""第4章：測り方の計画で、装置のクセとウェハーの変化を区別できるか（行列の階数とD基準）
第5章：府中の2台（F1, F2）を毎日同じ監視用ウェハーで測った差から、F2の段差がいつ起きたかを見つける（CUSUM、カルマンフィルタ）"""
import numpy as np
tools=['F1','F2','T1','T2','U1','U2','S1','S2']; site={'F':['F1','F2'],'T':['T1','T2'],'U':['U1','U2'],'S':['S1','S2']}
def design_matrix(routes):
    cols=[f'cd_{w}' for w in routes]+[f'off_{t}' for t in tools[1:]]
    legs=[]
    for w,r in routes.items():
        for k in range(1,len(r)): legs.append((w,k))
    cols+= [f'leg_{w}_{k}' for (w,k) in legs]
    X=[]
    for w,r in routes.items():
        for k,s in enumerate(r):
            for t in site[s]:
                row=np.zeros(len(cols)); row[cols.index(f'cd_{w}')]=1
                if t!='F1': row[cols.index(f'off_{t}')]=1
                for kk in range(1,k+1): row[cols.index(f'leg_{w}_{kk}')]=1
                X.append(row)
    return np.array(X),cols
designs={
 '案A：1枚のウェハーで一周（府中→台湾→米国→上海→府中）':{'W1':['F','T','U','S','F']},
 '案B：4枚（W1 府中→台湾→府中, W2 府中→米国→府中, W3 府中→台湾→米国→府中, W4 府中→上海→府中）':{'W1':['F','T','F'],'W2':['F','U','F'],'W3':['F','T','U','F'],'W4':['F','S','F']},
 '案C：4枚（すべて往復のみ。W3も府中→台湾→府中）':{'W1':['F','T','F'],'W2':['F','U','F'],'W3':['F','T','F'],'W4':['F','S','F']},
}
print('■ 未知数（ウェハーの真値、装置のクセ、輸送区間ごとのウェハーの変化）を、すべて区別できるか')
for name,r in designs.items():
    X,c=design_matrix(r); rk=np.linalg.matrix_rank(X); n=len(c)
    ok=rk==n
    D=np.linalg.slogdet(X.T@X)[1]/n if ok else float('nan')
    print(f'  {name}\n     未知数 {n}, 階数 {rk} → {"区別できる" if ok else "区別できない（自由度 "+str(n-rk)+" が残る）"}', f'D基準（1未知数あたり）{D:.3f}' if ok else '')
# ---- 第5章：F2-F1の差の時系列（毎日、監視用ウェハーを両方で測る） ----
rng=np.random.default_rng(11)
days=np.arange(80); true=0.12+0.18*(days>=40); sd=0.02*np.sqrt(2)
x=true+rng.normal(0,sd,len(days))
mu0=np.mean(x[:20]); k=0.045; h=0.15   # 基準期間の平均、許容幅、判定の閾値（nm）
S=0; alarm=None
for d,v in zip(days,x):
    S=max(0,S+(v-mu0-k))
    if S>h and alarm is None: alarm=d
print(f'\n■ CUSUM：基準期間の平均 {mu0:+.3f} nm、警報の日 = {alarm}日目（段差を仕込んだ日 = 40日目）')
# 変化点の推定（警報の前で、累積和が最後に0だった日）
S=0; last0=0
for d,v in zip(days,x):
    S=max(0,S+(v-mu0-k))
    if S==0: last0=d
    if d==alarm: break
print(f'  変化が始まった日の推定 = {last0+1}日目')
# カルマンフィルタ（ゆっくり変わる水準を追う）
q=0.003**2; r=sd**2; m=mu0; P=0.01; est=[]
for v in x:
    P+=q; K=P/(P+r); m=m+K*(v-m); P=(1-K)*P; est.append(m)
print(f'  カルマンフィルタの推定：39日目 {est[39]:+.3f}、45日目 {est[45]:+.3f}、79日目 {est[79]:+.3f}（正解：段差前 +0.120、段差後 +0.300）')
