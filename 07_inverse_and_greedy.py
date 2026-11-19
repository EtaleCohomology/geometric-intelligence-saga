"""07 目的の逆算（逆最適化）と、欲張り法の癖（第5章・第9章）。

1. 跳ね返りの足し合わせ：影響の行列 M のスペクトル半径 ρ が 1 より小さければ、
   I + M + M² + … = (I − M)^{-1} は収束する。ρ が 0.4 なら約1.7倍、0.9 なら約10倍。
2. 点が k 個の図の最大固有値は k−1 以下で、等号は完全グラフ（クリーク）のときだけ。
3. 架空の 6 社の図で、欲張り法の答えと、すべての組み合わせを調べた最適が、4社目で分かれる例。
4. 逆最適化の答えが 1 つに決まらない例：目的の大きさは、手からは分からない。
数字と会社はすべて架空。相手の手順を計算するためのコードではなく、守る側が指紋を読むための例である。
"""
import itertools
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---- 1. 跳ね返りの足し合わせ ---------------------------------------
for rho in (0.4, 0.9):
    M = rho * jnp.eye(1)
    print(f"スペクトル半径 {rho}: 跳ね返りを全部足すと、最初の傷の {float(jnp.linalg.inv(jnp.eye(1) - M)[0,0]):.2f} 倍")

# ---- 2. 最大固有値と完全グラフ -------------------------------------
rng = np.random.default_rng(3)
k = 4
K4 = np.ones((k, k)) - np.eye(k)
print("\n完全グラフ K4 の最大固有値:", round(float(np.linalg.eigvalsh(K4)[-1]), 6), "（k−1 = 3）")
worst = 0.0
for _ in range(2000):
    A = np.triu((rng.random((k, k)) < 0.6).astype(float), 1); A = A + A.T
    if A.sum() < k * (k - 1):
        worst = max(worst, float(np.linalg.eigvalsh(A)[-1]))
print("完全グラフでない 4 点の図の最大固有値の最大値:", round(worst, 4), "（3 より小さい）")

# ---- 3. 欲張り法と、すべての組み合わせ ------------------------------
names = ["カルテンブリュック", "玄冬", "藍潮", "凪浦", "Q社", "R社"]
w = np.array([0.38, 0.34, 0.17, 0.11, 0.0, 0.0])     # 目的 C の重み（狙う会社の価値の下落の重み）
# 影響の強さ M[i, j]：会社 j の傷が、会社 i にどれだけ伝わるか（架空。跳ね返りの計算に使う）
M = np.zeros((6, 6))
# 取引の結びつきの強さ L[i, j]：公開の情報から見える、2社の取引の太さ（架空。欲張り法が使う）
L = np.zeros((6, 6))
def link(i, j, m, l):
    M[i, j] = M[j, i] = m
    L[i, j] = L[j, i] = l
link(0, 1, 0.30, 0.30)   # カルテンブリュック—玄冬（圧縮機の納入と保守）
link(0, 2, 0.20, 0.25)   # カルテンブリュック—藍潮（倉庫に水産物を預ける）
link(2, 3, 0.10, 0.20)   # 藍潮—凪浦（売掛債権を売る）
link(1, 3, 0.10, 0.20)   # 玄冬—凪浦（顧客へのリース）
link(5, 2, 0.05, 0.05)   # R社—藍潮
# Q社 → カルテンブリュック：冷凍設備の部品をほぼ一手に納める。取引の額は小さいが、
# Q社が傷めば、カルテンブリュックの倉庫の設備が止まる（片向きの、強い影響）
M[0, 4] = 0.51; M[4, 0] = 0.02
L[0, 4] = L[4, 0] = 0.22


def objective(S):
    """選んだ会社に最初の傷を与えたとき、跳ね返りを全部足した、重みつきの価値の下落。"""
    S = list(S)
    shock = np.zeros(6); shock[S] = 1.0
    total = np.linalg.solve(np.eye(6) - M, shock)       # (I − M)^{-1} × 最初の傷
    return float(w @ total)


def greedy(k):
    S = [int(np.argmax(w))]                             # 最初に、重みの最も大きい会社
    while len(S) < k:
        rest = [j for j in range(6) if j not in S]
        # すでに選んだ会社との、つながりの強さの和が最も大きい会社を足す（実務の近道）
        S.append(max(rest, key=lambda j: sum(L[i, j] for i in S)))
    return S


g = greedy(4)
best = max(itertools.combinations(range(6), 4), key=objective)
print("\n欲張り法の道筋     :", " → ".join(names[i] for i in g), f"  目的の値 {objective(g):.3f}")
print("すべてを調べた最適 :", "、".join(names[i] for i in best), f"  目的の値 {objective(best):.3f}")
print(f"最適は、欲張り法より {objective(best)/objective(g) - 1:.1%} 大きい")

# ---- 4. 目的の大きさは、手からは分からない ---------------------------
# 相手は、目的の値が大きい手ほど選びやすい（ボルツマン型）、と確率で書く。
cands = list(itertools.combinations(range(6), 4))
vals = jnp.array([objective(c) for c in cands])
chosen = cands.index(tuple(sorted(g)))


def loglik(beta, scale):
    logits = beta * scale * vals
    return float(logits[chosen] - jax.nn.logsumexp(logits))


print("\n目的を c 倍し、選び方の鋭さ β を 1/c 倍しても、手の起こりやすさは同じ")
for c in (1.0, 2.0, 10.0):
    print(f"  c = {c:4.1f}: 対数尤度 {loglik(20.0 / c, c):.6f}")
print("→ だから、重みの合計を 1 にそろえる、という決まりを先に置く。")
