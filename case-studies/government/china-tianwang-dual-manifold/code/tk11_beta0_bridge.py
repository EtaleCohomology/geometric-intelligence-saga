"""
敵の多様体 補助文書03 コード11：パーシステント・ホモロジーの β0 ── 別々だった２つの塊がつながる
２つの点の塊（中国の動き、ロシアの関与が疑われる動き）に、あとから両者をつなぐ点の列（取っ手）が加わる。
半径 r の球を各点に置いたとき、いくつのかたまりに分かれるか（β0）を、ユニオン・ファインドで数える。数値は架空のもの。
"""
import numpy as np
rng = np.random.default_rng(1)
A = rng.normal([0, 0], 0.4, (40, 2)); B = rng.normal([5, 0], 0.4, (40, 2))
handle = np.c_[np.linspace(0.8, 4.2, 12), 0.15 * rng.standard_normal(12)]
def beta0(P, r):
    n = len(P); parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= 2 * r: parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})
for r in [0.3, 0.5, 0.8]:
    print(f"r={r}: 取っ手の前 β0={beta0(np.vstack([A, B]), r)}   取っ手の後 β0={beta0(np.vstack([A, B, handle]), r)}")
print("→ 取っ手が加わると、同じ半径で、２つの塊が１つにつながる（β0 が 2 → 1）。")
