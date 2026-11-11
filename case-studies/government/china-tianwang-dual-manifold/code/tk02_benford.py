"""
敵の多様体 補助文書03 コード2：ベンフォードの法則
自然に生まれる数値（桁が大きく違うものが混ざった数値）と、人の手で作られた数値の、先頭の桁の分布を比べる。
数値は架空のもの。外れていることは、不正の証拠ではなく、調べる場所の手がかりにすぎない。
"""
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
benford = np.log10(1 + 1 / np.arange(1, 10))
def first_digit(x): return (x / 10 ** np.floor(np.log10(x))).astype(int)
natural = rng.lognormal(mean=10, sigma=2.5, size=3000)          # 何桁にもわたって散らばる数値
made = rng.uniform(4500, 6500, size=3000)                        # 目標の近くに作られた数値（範囲が狭い）
for name, x in [("自然に生まれた数値", natural), ("人の手で作られた数値", made)]:
    obs = np.bincount(first_digit(x), minlength=10)[1:]
    chi2, pval = stats.chisquare(obs, benford * obs.sum())
    print(f"{name}: 先頭が1の割合 {obs[0]/obs.sum():.3f}（ベンフォードでは {benford[0]:.3f}）  カイ二乗 {chi2:.1f}  p値 {pval:.2g}")
print("注意：範囲の狭い数値や目標値の決まった数値には、もともとベンフォードの法則は当てはまらない。")
