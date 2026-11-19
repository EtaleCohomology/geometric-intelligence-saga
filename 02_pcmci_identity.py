"""02 恒等式で決まる勘定科目を外して、遅れのある因果を PCMCI で推定する（第3章・第4章）。

架空の月次データ（120か月）。本当のつながり（データを作った仕組み）は次のとおり。
  売上 S_t       : 自分の過去に依存する（季節を除いた動き）
  売掛金 R_t     : 2か月前の売上 S_{t-2} に依存する（回収までの遅れ）
  在庫 I_t       : 1か月前の売上 S_{t-1} に依存する
  現金 C_t       : 1か月前の売掛金 R_{t-1} に依存する（回収されて現金になる）
  資産の合計 A_t  : R_t + I_t + C_t（会計の恒等式。定義で決まる）
PCMCI（Runge ほか、2019年）は、条件付き独立性の検定を積み重ねる。
資産の合計のように、ほかの勘定科目の和で完全に決まる変数を入れると、
それを条件に入れた検定で残差が 0 になり、検定そのものが成り立たなくなる。
"""
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

rng = np.random.default_rng(0)
T = 120
S = np.zeros(T); R = np.zeros(T); I = np.zeros(T); C = np.zeros(T)
for t in range(T):
    S[t] = (0.5 * S[t-1] if t >= 1 else 0) + rng.normal()
    R[t] = (0.8 * S[t-2] if t >= 2 else 0) + 0.5 * rng.normal()
    I[t] = (0.6 * S[t-1] if t >= 1 else 0) + 0.5 * rng.normal()
    C[t] = (0.7 * R[t-1] if t >= 1 else 0) + 0.5 * rng.normal()
A = R + I + C                                  # 恒等式

def run(data, names):
    df = pp.DataFrame(data, var_names=names)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_min=1, tau_max=3, pc_alpha=0.05)
    links = []
    for j in range(len(names)):
        for i in range(len(names)):
            for tau in range(1, 4):
                if res["p_matrix"][i, j, tau] < 0.01:
                    links.append(f"{names[i]}(t-{tau}) → {names[j]}")
    return links

names4 = ["売上", "売掛金", "在庫", "現金"]
print("恒等式で決まる科目を外した場合（独立な科目の組だけ）")
for l in run(np.column_stack([S, R, I, C]), names4):
    print("  ", l)

# 恒等式で決まる科目を入れた場合：同じ時点の A_t は R_t, I_t, C_t で完全に決まる
X = np.column_stack([R, I, C])
coef, *_ = np.linalg.lstsq(X, A, rcond=None)
resid = A - X @ coef
print("\n資産の合計を、売掛金・在庫・現金で回帰した残差の分散:", float(np.var(resid)))
print("→ 0 なので、これらを条件に入れた条件付き独立性の検定は、分母が 0 になり定義できない。")

names5 = names4 + ["資産の合計"]
print("\n恒等式で決まる科目を入れたまま走らせた場合（遅れ1〜3か月の検定だけでも）")
links5 = run(np.column_stack([S, R, I, C, A]), names5)
for l in links5:
    print("  ", l)
print("→ 『売上 → 資産の合計』の矢印が、遅れ1〜3か月で加わる。資産の合計は、"
      "売掛金・在庫・現金の和として定義で決まるので、これは事業の仕組みの因果ではなく、"
      "定義を通した見かけの矢印である。どの勘定科目が何に響くかを読む図には、入れてはならない。")
