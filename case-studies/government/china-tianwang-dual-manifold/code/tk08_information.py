"""
敵の多様体 補助文書03 コード8：相互情報量とデータ処理不等式、KL 情報量とワッサースタイン距離
(1) 現実 X を観測した Y から、都合の悪い値を「平穏」に書き換えた Y'（検閲）を作ると、I(X;Y') ≤ I(X;Y)。
(2) 現実の分布とモデルの分布が重ならない部分を持つと、KL 情報量は発散するが、ワッサースタイン距離は有限のまま。
数値は架空のもの。
"""
import numpy as np
from scipy.stats import wasserstein_distance
rng = np.random.default_rng(10)
def mutual_info(a, b):
    ja = np.unique(a); jb = np.unique(b); p = np.zeros((len(ja), len(jb)))
    for i, u in enumerate(ja):
        for j, v in enumerate(jb): p[i, j] = np.mean((a == u) & (b == v))
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())
X = rng.integers(0, 4, 20000)                          # 現実の状態（0=平穏 … 3=危機）
Y = np.where(rng.random(20000) < 0.9, X, rng.integers(0, 4, 20000))   # 正直な観測（1割は誤り）
Yc = np.where(Y >= 2, 0, Y)                            # 検閲：危うい値を「平穏」に書き換える
print(f"I(X;Y)  正直な観測 = {mutual_info(X, Y):.3f} ビット")
print(f"I(X;Y') 検閲した観測 = {mutual_info(X, Yc):.3f} ビット（増えることはない）")
# KL とワッサースタイン：モデル（検閲された分布）は、現実の遠くの裾（隠れ債務）に質量を持たない
bins = np.arange(0, 11)
real = np.r_[rng.normal(3, 1, 9000), rng.normal(9, 0.5, 1000)]          # 1割が遠くの裾
model = rng.normal(3, 1, 10000)
P, _ = np.histogram(real, bins=bins, density=True); Q, _ = np.histogram(model, bins=bins, density=True)
with np.errstate(divide="ignore", invalid="ignore"):
    kl = np.sum(np.where(P > 0, P * np.log(P / Q), 0.0))
print(f"KL(現実‖モデル) = {kl}（モデルが確率0とする場所に現実の質量があると発散する）")
print(f"ワッサースタイン距離 = {wasserstein_distance(real, model):.3f}（遠くの質量も、動かす手間として有限に測れる）")
