"""第6章：両立の判定（架空の数値と、物語の中の法務の判断）  第7章：分けるか待つか（実物オプション）"""
import numpy as np, z3
from scipy.optimize import linprog
# ---- 数値の条件：中国の先端の工場の顧客向けの保守（月あたりの技術者の時間） ----
# 変数 x=[上海の非米国人の技術者の時間, 府中から派遣する技術者の時間, 米国人の技術者の時間]
rows=['顧客との契約：必要な保守の時間  x1+x2+x3 ≥ 1800','上海の非米国人の技術者の上限  x1 ≤ 1200','府中からの派遣の上限  x2 ≤ 400','米国の規制（許可なし）：米国人の時間  x3 ≤ 0']
A=np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]],float); b=np.array([-1800,1200,400,0],float)
def check(A,b,rows,title):
    r=linprog(np.zeros(3),A_ub=A,b_ub=b,bounds=[(0,None)]*3,method='highs')
    print('■',title)
    if r.status==0: print('  両立する。例：',np.round(r.x)); return
    m=len(b); Aeq=np.vstack([A.T,b[None,:]])
    # x>=0 なので、証明書は y>=0, A^T y >= 0, b^T y < 0
    c=linprog(np.ones(m),A_ub=-A.T,b_ub=np.zeros(3),A_eq=b[None,:],b_eq=[-1],bounds=[(0,None)]*m,method='highs')
    y=c.x/np.max(c.x)
    print('  両立しない。証明書（各条件に掛ける重み）：')
    for yi,row in zip(y,rows):
        if yi>1e-9: print(f'    {yi:.2f} × [{row}]')
    print(f'  重みをつけて足すと：0 ≤ {y@b:.0f}（矛盾）')
check(A,b,rows,'先端の工場の顧客（月1,800時間の保守が必要）')
b2=b.copy(); b2[0]=-900
check(A,b2,['顧客との契約：必要な保守の時間 ≥ 900']+rows[1:],'先端でない工場の顧客（月900時間）')
# ---- 数値でない条件 ----
v={k:z3.Bool(k) for k in ['raw_export','by_central','central_has_us','us_support_adv','by_china_team','diff_only']}
R={
 'C1 中国：上海の生の計測データは、安全評価が済むまで国外へ出さない':z3.Not(v['raw_export']),
 'K1 社内規程：中国の先端の工場向けの補正値も、府中の全社チームが計算する':v['by_central'],
 'K2 府中の全社チームには、米国子会社の米国人の技術者が含まれる':v['central_has_us'],
 'L1 法務の判断（物語の設定）：先端の工場の装置の補正値を米国人が計算すると、保守の支援に当たる':z3.Implies(z3.And(v['by_central'],v['central_has_us']),v['us_support_adv']),
 'U1 米国の規制：許可がない限り、米国人は先端の工場の保守を支援しない':z3.Not(v['us_support_adv']),
 'M1 補正値の計算には、生のデータか、差の集計のどちらかが必要':z3.Or(v['raw_export'],v['diff_only']),
 'M2 補正値は、府中の全社チームか、中国向けの専任チームのどちらかが計算する':z3.Or(v['by_central'],v['by_china_team']),
}
def sat(rs,title):
    s=z3.Solver(); tags={}
    for k,e in rs.items():
        t=z3.Bool('t:'+k); tags[str(t)]=k; s.assert_and_track(e,t)
    print('\n■',title)
    if s.check()==z3.sat:
        m=s.model(); print('  両立する：',{str(d):m[d] for d in m.decls() if not str(d).startswith('t:')}); return
    core=[tags[str(c)] for c in s.unsat_core()]
    for k in list(core):
        t=[c for c in core if c!=k]; s2=z3.Solver(); [s2.add(rs[c]) for c in t]
        if s2.check()==z3.unsat: core=t
    print('  両立しない。原因になっている最小の組：'); [print('   -',c) for c in core]
sat(R,'いまの体制')
R2={k:e for k,e in R.items() if not k.startswith('K1')}
R2["K1' 社内規程の見直し：中国の先端の工場向けの補正値は、米国人を含まない専任チームが、差の集計だけから計算する"]=z3.And(z3.Not(v['by_central']),v['by_china_team'],v['diff_only'])
sat(R2,'専任チームに分けた場合')
# ---- 第7章 実物オプション（単位：億円、割引率5%） ----
r=0.05; d=1/(1+r); cC,cR,K,s=12.0,2.0,40.0,3.0
pv=lambda a:a*(1+r)/r; VR=pv(cR); Vs=K+pv(s)
def solve(q):
    V=Vs
    for _ in range(3000):
        w=cC+d*(q*VR+(1-q)*V); V=min(Vs,w)
    return V,w
lo,hi=0,1
for _ in range(60):
    mid=(lo+hi)/2; _,w=solve(mid)
    lo,hi=(mid,hi) if w>Vs else (lo,mid)
print(f'\n■ 実物オプション：いま分ける {Vs:.1f}、解消後も統合のまま {VR:.1f}、境目の確率 q* ≈ {hi:.3f}')
for q in [0.10,0.20,0.30]:
    V,w=solve(q); print(f'  q={q:.2f}：待つ場合の費用 {w:.1f}、分ける場合 {Vs:.1f}、待つことの価値 {max(0,Vs-w):.1f}')
