# s02 グラフラプラシアン、定数の層、木と輪、響き方、六角形と三角形２つ、フィードラー値
# （第5章「ラプラシアン」「同じ門」「響き方の点検」「近くだけ見る AI」「照らし合わせの強さ」、第6章）
import numpy as np
import networkx as nx
from common import plt, savefig, NAVY, CYAN, AMBER, GREY


def coboundary(G, stalk=1, maps=None):
    """セルラー層のコバウンダリ δ。maps[(u,v)] = (F_u, F_v) は点から線への制限写像（行列）。
    maps を省くと定数の層（どの制限写像も「値をそのまま渡す」）。"""
    nodes = list(G.nodes()); pos = {n: k for k, n in enumerate(nodes)}
    E = list(G.edges())
    d = np.zeros((len(E) * stalk, len(nodes) * stalk))
    for k, (u, v) in enumerate(E):
        Fu, Fv = (np.eye(stalk), np.eye(stalk)) if maps is None else maps[(u, v)]
        d[k*stalk:(k+1)*stalk, pos[u]*stalk:(pos[u]+1)*stalk] = Fu
        d[k*stalk:(k+1)*stalk, pos[v]*stalk:(pos[v]+1)*stalk] = -Fv
    return d


# 1. 点Ｂでのグラフラプラシアン：Ａ=2, Ｂ=5, Ｃ=3 を１列につなぐ
P3 = nx.path_graph(["A", "B", "C"])
L = nx.laplacian_matrix(P3, nodelist=["A", "B", "C"]).toarray()
x = np.array([2, 5, 3])
print("Ｄ − Ａ =\n", L)
print("(Ｄ − Ａ)x =", L @ x, " ← 点Ｂの値は (5−2)+(5−3) = 5")
print("線の両端の差の２乗の合計 xᵀLx =", x @ L @ x, "=", (5-2)**2 + (5-3)**2)

# 2. 同じ門：定数の層の層ラプラシアン δᵀδ はグラフラプラシアンに一致する
for G, name in [(nx.cycle_graph(6), "輪6"), (nx.path_graph(6), "1列6"), (nx.petersen_graph(), "ピーターセン")]:
    d = coboundary(G)
    same = np.allclose(d.T @ d, nx.laplacian_matrix(G, nodelist=list(G.nodes())).toarray())
    print(f"{name}: δᵀδ と Ｄ − Ａ は一致するか → {same}")

# 3. 木と輪：H⁰ と H¹ の次元を、δ の階数から計算する
for G, name in [(nx.path_graph(3), "木（３点１列）"), (nx.cycle_graph(3), "輪（三角形）")]:
    d = coboundary(G)
    r = np.linalg.matrix_rank(d)
    h0 = d.shape[1] - r          # 核の次元 ＝ 大域切断の次元
    h1 = d.shape[0] - r          # 線の側で、点の側から作れない方向の数
    print(f"{name}: H⁰ の次元 = {h0}, H¹ の次元 = {h1}")
# 木では、どんな線の値も点の値から作れる（δ が上への写像）。だから食い違いは吸い込まれる
d = coboundary(nx.path_graph(3)); yv = np.array([0.7, -1.3])
xs, *_ = np.linalg.lstsq(d, yv, rcond=None)
print("木の上の勝手な食い違い", yv, "→ 残り", np.round(yv - d @ xs, 12))
d = coboundary(nx.cycle_graph(3)); yv = np.array([0.7, -1.3, 0.9])
xs, *_ = np.linalg.lstsq(d, yv, rcond=None)
print("三角形の上の食い違い", yv, "→ 残り", np.round(yv - d @ xs, 4), "（１周の合計",
      round(float(yv.sum()), 4), "の分は消えない）")

# 4. 響き方：輪と１列の固有値
ring = np.round(np.linalg.eigvalsh(nx.laplacian_matrix(nx.cycle_graph(6)).toarray()), 2) + 0.0
ring[np.abs(ring) < 1e-9] = 0.0
line = np.round(np.linalg.eigvalsh(nx.laplacian_matrix(nx.path_graph(6)).toarray()), 2)
print("輪　（６点）の固有値:", ring)
print("１列（６点）の固有値:", line)

# 5. 六角形と、三角形２つ：１次元のヴァイスファイラー・レーマン検定では区別できない
hexagon = nx.cycle_graph(6)
two_tri = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))


def wl_colors(G, rounds=5):
    col = {v: 0 for v in G}
    for _ in range(rounds):
        sig = {v: (col[v], tuple(sorted(col[u] for u in G[v]))) for v in G}
        table = {s: k for k, s in enumerate(sorted(set(sig.values())))}
        col = {v: table[sig[v]] for v in G}
    return sorted(col.values())


print("WL の色の並び：六角形", wl_colors(hexagon), " 三角形２つ", wl_colors(two_tri))
for G, name in [(hexagon, "六角形"), (two_tri, "三角形２つ")]:
    b0 = nx.number_connected_components(G)
    b1 = G.number_of_edges() - G.number_of_nodes() + b0
    print(f"{name}: 塊の数（０次のベッチ数） {b0}, 独立な輪の数（１次のベッチ数） {b1}")

# 6. フィードラー値（２番目に小さい固有値）：つながりの強さの目安
for G, name in [(nx.path_graph(6), "１列"), (nx.cycle_graph(6), "輪"),
                (nx.barbell_graph(3, 0), "三角形２つを１本の線でつなぐ"), (nx.complete_graph(6), "全部つなぐ")]:
    ev = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray()))
    print(f"{name}: フィードラー値 {ev[1]:.3f}")

# --- 図：輪と１列の響き方 ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, vals, title, col in [(axes[0], ring, "輪（６点）", CYAN), (axes[1], line, "１列（６点）", AMBER)]:
    ax.bar(range(1, 7), vals, color=col)
    for k, v in enumerate(vals):
        ax.text(k + 1, v + 0.08, f"{v:g}", ha="center", fontsize=9)
    ax.set_title(f"{title}の固有値（響き方）", fontsize=10)
    ax.set_xticks(range(1, 7)); ax.set_ylim(0, 4.6)
    ax.set_xlabel("小さいほうから何番目か")
savefig(fig, "s02_ring_vs_path.png")

# --- 図：六角形と三角形２つ ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, G, title in [(axes[0], hexagon, "六角形：塊１、輪１"), (axes[1], two_tri, "三角形２つ：塊２、輪２")]:
    pos = nx.circular_layout(hexagon) if G is hexagon else {
        0: (-1.2, 0.5), 1: (-0.4, 0.5), 2: (-0.8, -0.2), 3: (0.4, 0.5), 4: (1.2, 0.5), 5: (0.8, -0.2)}
    nx.draw(G, pos, ax=ax, node_color=CYAN, edge_color=NAVY, node_size=380, width=2)
    ax.set_title(title, fontsize=10)
fig.suptitle("どの点も隣は２つ。近くだけ見ると同じに見える", fontsize=11)
savefig(fig, "s02_hexagon_two_triangles.png")
