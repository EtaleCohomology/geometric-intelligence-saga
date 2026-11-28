# s09 メンガーの定理：輪のある記録の設計（第11章）
# 出来事が起きた場所 s から、記録が最後に届く場所 t まで、途中で同じ点を通らない道の最大の本数と、
# s と t を切り離すためにふさぐ点の最小の数は等しい。
import networkx as nx
from networkx.algorithms.connectivity import node_disjoint_paths, minimum_st_node_cut
from common import plt, savefig, NAVY, CYAN, AMBER, RED

G = nx.Graph()
G.add_edges_from([
    ("出来事", "作業記録アプリ"), ("作業記録アプリ", "記録サーバー"), ("記録サーバー", "保管庫"),
    ("出来事", "端末の基本ソフト"), ("端末の基本ソフト", "中継装置"), ("中継装置", "保管庫"),
    ("出来事", "電話交換機"), ("電話交換機", "クラウド"), ("クラウド", "保管庫"),
    ("作業記録アプリ", "端末の基本ソフト"),
])
s, t = "出来事", "保管庫"
paths = list(node_disjoint_paths(G, s, t))
cut = minimum_st_node_cut(G, s, t)
print("途中で同じ点を通らない道:", len(paths))
for p in paths:
    print("  ", " → ".join(p))
print("切り離すためにふさぐ点の最小の数:", len(cut), sorted(cut))

H = G.copy(); H.remove_edges_from([("電話交換機", "クラウド")])
print("電話交換機からの道がなくなると:", len(list(node_disjoint_paths(H, s, t))), "本")

pos = {"出来事": (0, 0), "作業記録アプリ": (1, 1), "端末の基本ソフト": (1, 0), "電話交換機": (1, -1),
       "記録サーバー": (2, 1), "中継装置": (2, 0), "クラウド": (2, -1), "保管庫": (3, 0)}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=1.5, ax=ax)
for p, c in zip(paths, [CYAN, AMBER, RED]):
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(p, p[1:])), edge_color=c, width=3.5, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=NAVY, node_size=300, ax=ax)
nx.draw_networkx_labels(G, {k: (v[0], v[1] + 0.2) for k, v in pos.items()}, font_size=8,
                        font_family="Noto Sans CJK JP", ax=ax)
ax.set_title("途中で同じ点を通らない道が３本 ⇔ 切り離すには３点をふさぐ必要がある", fontsize=10)
ax.set_ylim(-1.35, 1.45); ax.set_xlim(-0.3, 3.3)
ax.axis("off")
savefig(fig, "s09_menger.png")
