# s01 記録の出どころの図の上のセルラー層で、時計を貼り合わせる（第5章、第6章）
#
# 点：記録を残した機械。茎は (a, b) の２つの数（ずれ a 秒、進み方の違い b 秒/時）。
# 線：同じ出来事を２つの機械が記録した組。観測は「２つの記録の時刻の差」。
# 制限写像：(a, b) を a + b*t に移す。t は出来事の時刻なので、出来事ごとに変わる。
# すべて架空のデータである。
import numpy as np
import networkx as nx
from common import plt, savefig, NAVY, CYAN, AMBER, RED, GREY

rng = np.random.default_rng(20381018)

# --- 1. 記録の出どころの図（架空）------------------------------------------
names = ["親機", "サーバー1", "サーバー2", "ネットワーク機器", "クラウド", "AIエージェント",
         "中継装置", "制御装置", "自動搬送車", "端末の基本ソフト", "作業記録アプリ", "電話交換機", "仕分けライン"]
N = len(names)
idx = {n: i for i, n in enumerate(names)}
edges = [("親機", "サーバー1"), ("親機", "サーバー2"), ("サーバー1", "ネットワーク機器"),
         ("サーバー2", "クラウド"), ("ネットワーク機器", "AIエージェント"), ("クラウド", "AIエージェント"),
         ("ネットワーク機器", "中継装置"), ("中継装置", "制御装置"), ("制御装置", "仕分けライン"),
         ("中継装置", "自動搬送車"), ("自動搬送車", "仕分けライン"),          # 輪Ａ（倉庫の機械の側）
         ("サーバー1", "端末の基本ソフト"), ("端末の基本ソフト", "作業記録アプリ"),
         ("作業記録アプリ", "電話交換機"), ("電話交換機", "仕分けライン"),
         ("仕分けライン", "端末の基本ソフト")]                                  # 輪Ｂ（作業の記録を含む）

# 本当のずれと進み方の違い（親機を基準 0 にする）
a_true = rng.normal(0, 1.5, N); a_true[0] = 0.0
b_true = rng.normal(0, 0.02, N); b_true[0] = 0.0
# 出どころごとの時刻の揺れ（秒）。制御装置だけ１秒刻みなので 1/√12 ≒ 0.29 秒
sigma = np.full(N, 0.004)
sigma[idx["制御装置"]] = 1 / np.sqrt(12)

# --- 2. 観測（同じ出来事の２つの記録の差）を作る ----------------------------
rows = []  # (i, j, t, y)
for (u, v) in edges:
    i, j = idx[u], idx[v]
    for t in np.sort(rng.uniform(0, 6, 60)):          # 夜の 0〜6 時の出来事
        ti = t + a_true[i] + b_true[i] * t
        tj = t + a_true[j] + b_true[j] * t
        if sigma[i] > 0.1: ti = np.floor(ti)            # １秒刻みに切り捨てる機械
        else: ti += rng.normal(0, sigma[i])
        if sigma[j] > 0.1: tj = np.floor(tj)
        else: tj += rng.normal(0, sigma[j])
        rows.append((i, j, t, ti - tj))

# 作業記録アプリの、ある区間（2:19〜2:31）の時刻だけを、あとから早い側へ書き直す
TAMPER = 0.3  # 秒（小さく、静かな書き直し）
rows = [(i, j, t, y - TAMPER if (names[i] == "作業記録アプリ" and 2.32 <= t <= 2.52) else
         (y + TAMPER if (names[j] == "作業記録アプリ" and 2.32 <= t <= 2.52) else y))
        for (i, j, t, y) in rows]


def solve(rows, weights):
    """コバウンダリ δ を組み立てて、重み付き最小二乗で茎の値を合わせる。
    親機（基準）の (a, b) は 0 に固定する（H⁰ の自由を消す）。"""
    M = len(rows)
    delta = np.zeros((M, 2 * N))
    y = np.zeros(M)
    for k, (i, j, t, val) in enumerate(rows):
        delta[k, 2 * i], delta[k, 2 * i + 1] = 1, t        # 点 i から線への制限写像 (a,b) -> a + b t
        delta[k, 2 * j], delta[k, 2 * j + 1] = -1, -t
        y[k] = val
    keep = [c for c in range(2 * N) if c not in (0, 1)]    # 基準の時計を外す
    W = np.sqrt(weights)
    x, *_ = np.linalg.lstsq(delta[:, keep] * W[:, None], y * W, rcond=None)
    full = np.zeros(2 * N); full[keep] = x
    resid = y - delta @ full
    return full, resid, delta


# --- 3. １回目の計算：全部の線を同じ重みで比べる（カイトの誤り）------------
w1 = np.ones(len(rows))
_, r1, _ = solve(rows, w1)
# --- 4. 雑音の大きさで割る（頑健な標準化）-----------------------------------
sig_edge = np.array([np.hypot(sigma[i], sigma[j]) for (i, j, t, y) in rows])
_, r2, delta = solve(rows, 1 / sig_edge**2)
z = r2 / sig_edge
# 外れた値に引っぱられない尺度：中央値と、中央値からの離れ方の中央値（MAD）
med = np.median(z); mad = np.median(np.abs(z - med)) * 1.4826
zr = (z - med) / mad


def by_edge(vals):
    out = {}
    for (i, j, t, y), v in zip(rows, vals):
        out.setdefault((names[i], names[j]), []).append(abs(v))
    return {e: max(v) for e, v in out.items()}


e1 = by_edge(r1)
e2 = by_edge(zr)
print("１回目（同じ重み）で食い違いの大きい線：")
for e, v in sorted(e1.items(), key=lambda kv: -kv[1])[:4]:
    print(f"  {e[0]} - {e[1]}: {v:.2f} 秒")
print("雑音の大きさで割ったあと、食い違いの大きい線（頑健な z 値）：")
for e, v in sorted(e2.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {e[0]} - {e[1]}: {v:.1f}")

# --- 5. 部分ごとの整合性の半径（輪ごとの食い違いのいちばん大きな値）---------
G = nx.Graph(); G.add_nodes_from(names); G.add_edges_from(edges)
basis = nx.cycle_basis(G)
print("独立な輪の数（＝定数の層の H¹ の次元）:", len(basis),
      "＝ 線の数 − 点の数 + 塊の数 =", G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))
loops = {"輪Ａ（倉庫の機械の側）": ["中継装置", "制御装置", "仕分けライン", "自動搬送車"],
         "輪Ｂ（作業の記録を含む）": ["端末の基本ソフト", "作業記録アプリ", "電話交換機", "仕分けライン"]}
for name, cyc in loops.items():
    cyc_edges = {frozenset((cyc[k], cyc[(k + 1) % len(cyc)])) for k in range(len(cyc))}
    vals = [v for e, v in e2.items() if frozenset(e) in cyc_edges]
    print(f"  {name}: 整合性の半径 {max(vals):.1f}")
rest = [v for e, v in e2.items() if not any(frozenset(e) <= set(c) for c in loops.values())]
print(f"  そのほかの線: 整合性の半径 {max(rest):.1f}")

# --- 6. 図 ------------------------------------------------------------------
pos = nx.kamada_kawai_layout(G)
fig, ax = plt.subplots(figsize=(8, 5.2))
zz = [e2.get((u, v), e2.get((v, u))) for (u, v) in G.edges()]
col = [RED if z_ > 6 else (AMBER if z_ > 2.5 else CYAN) for z_ in zz]
wid = [3.2 if c != CYAN else 1.6 for c in col]
nx.draw_networkx_edges(G, pos, edge_color=col, width=wid, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=NAVY, node_size=260, ax=ax)
nx.draw_networkx_labels(G, {k: (p[0], p[1] + 0.09) for k, p in pos.items()}, font_size=8,
                        font_family="Noto Sans CJK JP", ax=ax)
ax.set_title("雑音の大きさで割ったあとに残る食い違い：作業記録アプリを含む輪の上に集まる（赤と橙）", fontsize=10)
ax.axis("off")
savefig(fig, "s01_residual_graph.png")
