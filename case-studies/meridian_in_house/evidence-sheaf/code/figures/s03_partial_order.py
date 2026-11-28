# s03 半順序と、それと食い違わない並べ方の数（第4章「時計を信じずに言えること」）
# ａ → ｂ, ａ → ｃ, ｂ → ｄ, ｃ → ｄ。ｅ はどれとも矢印でつながっていない。
from itertools import permutations
import networkx as nx
from common import plt, savefig, NAVY, CYAN, AMBER

events = ["a", "b", "c", "d", "e"]
before = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]


def respects(order):
    pos = {x: k for k, x in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in before)


ext = [o for o in permutations(events) if respects(o)]
print("半順序と食い違わない並べ方の数:", len(ext))
for o in ext:
    print("  ", " ".join(o))

# ネットワーク機器の時計が親機より 2.6 秒遅れていた例（第6章）。時刻は「2:06:00 からの秒」で書く
cmd_ai = 60.0           # AI エージェントの命令 2:07:00（AI エージェントの時計は親機と合っていた）
cut_net = 58.0          # ネットワーク機器の記録で、つながりが切れた 2:06:58
print("直す前：切れた時刻 − 命令の時刻 =", cut_net - cmd_ai, "秒（命令より先に切れたように見える）")
print("直した後：", round(cut_net + 2.6 - cmd_ai, 1), "秒（命令のほうが 0.6 秒先）")

# 図：ハッセ図
G = nx.DiGraph(before); G.add_node("e")
pos = {"a": (0, 0), "b": (-0.7, 1), "c": (0.7, 1), "d": (0, 2), "e": (1.8, 1)}
fig, ax = plt.subplots(figsize=(4.6, 3.8))
nx.draw_networkx_nodes(G, pos, node_color=[CYAN] * 4 + [AMBER], node_size=700, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=13, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color=NAVY, width=2, arrowsize=18, ax=ax, node_size=700)
ax.set_title("矢印は「確かに前」。ｅは、どれとも前後が決まらない", fontsize=10)
ax.set_xlim(-1.1, 2.2); ax.set_ylim(-0.4, 2.4)
ax.axis("off")
savefig(fig, "s03_poset.png")
