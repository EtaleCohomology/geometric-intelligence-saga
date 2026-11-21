"""08 危うさで測った距離が短い道（測地線）（本編第5章のたとえ話）。

平面の上に、危うさの山 ρ(x, y) を置く（架空）。危うさの大きい場所ほど距離を長く測る定規
  ds = (1 + λ ρ(x, y)) · （ふつうの長さ）
を考える。この定規は、ふつうの距離の定規に正の関数を掛けたもの（共形な計量）で、リーマン計量の一種である。
1. 平面を細かい格子にして、隣り合う点（斜めを含む）を結び、線の長さを上の定規で測る
2. ダイクストラの方法で、出発から行き先までの最短の道（測地線の近似）を求める
3. まっすぐな道と比べ、ふつうの長さ、危うさで測った長さ、道の上の危うさの最大を並べる
4. λ（危うさをどれだけ重く見るか）を変えると、道が変わることを確かめる
格子の道は、斜め45度の刻みでしか曲がれないので、本当の測地線より少し長く出る（格子を細かくすると近づく）。
"""
import numpy as np
import networkx as nx

n = 61
xs = np.linspace(0, 1, n)
X, Y = np.meshgrid(xs, xs, indexing="ij")
bumps = [(0.5, 0.5, 0.12, 1.0), (0.3, 0.65, 0.08, 0.8), (0.72, 0.35, 0.09, 0.9)]
rho = sum(a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w)) for cx, cy, w, a in bumps)

def build(lam):
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n:
                    L = np.hypot(xs[a] - xs[i], xs[b] - xs[j])
                    w = L * (1 + lam * 0.5 * (rho[i, j] + rho[a, b]))
                    G.add_edge((i, j), (a, b), weight=w, length=L)
    return G

def measure(G, path, lam):
    plain = sum(G[u][v]["length"] for u, v in zip(path, path[1:]))
    risk_len = sum(G[u][v]["weight"] for u, v in zip(path, path[1:]))
    peak = max(rho[p] for p in path)
    return plain, risk_len, peak

start, goal = (0, 30), (60, 30)
straight = [(i, 30) for i in range(n)]
print("λ      道             ふつうの長さ  危うさで測った長さ  道の上の危うさの最大")
for lam in [0, 2, 8, 30]:
    G = build(lam)
    geo = nx.dijkstra_path(G, start, goal, weight="weight")
    for name, P in [("まっすぐな道", straight), ("測地線の近似", geo)]:
        a, b, c = measure(G, P, lam)
        print(f"{lam:3d}    {name:12s}  {a:.3f}        {b:.3f}              {c:.3f}")
print("λ = 0 では、危うさを測らないので、まっすぐな道が最も短い。λ を大きくすると、道は危うさの山を避けて遠回りする。")
print("λ は、危うさをどれだけ重く見るかの重みで、数学ではなく、依頼する側の価値で決まる。")

# 測地線の性質の確かめ：道の途中の短い区間を取り出しても、その区間の両端の最短の道になっている
G = build(8)
geo = nx.dijkstra_path(G, start, goal, weight="weight")
bad = 0
for k in range(0, len(geo) - 6, 3):
    seg = geo[k:k + 6]
    seg_len = sum(G[u][v]["weight"] for u, v in zip(seg, seg[1:]))
    best = nx.dijkstra_path_length(G, seg[0], seg[-1], weight="weight")
    bad += seg_len > best + 1e-12
print(f"\nλ = 8 の道で、短い区間（5歩）を取り出したとき、もっと近い道があった区間の数：{bad}")
