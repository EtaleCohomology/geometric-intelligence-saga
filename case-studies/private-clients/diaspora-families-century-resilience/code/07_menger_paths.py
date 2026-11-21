"""07 互いに交わらない道の数と、切り離すためにふさぐ場所の数（メンガーの定理）（本編第5章）。

一族から「次の世代」までの道を、町（点）と道（線）の図にする（すべて架空）。
点の名前は「拠点」「拠点１」「拠点２」とし、実在の国や都市の危うさを表すものではない。
メンガーの定理（点の版）：隣り合わない２点 s, t について、
  s と t を結ぶ、途中の点を共有しない道の最大の本数 ＝ s と t を切り離すためにふさぐ点の最小の数
1. 道が１本に集まる「ひとつの籠」の図と、２本に分けた図で、両方の数を計算して一致を確かめる
2. 最小の数だけ点をふさいで、実際に切り離されることを確かめる
3. 無作為にいくつかの点がふさがれたとき、s と t がつながったままでいる確率を比べる
"""
import itertools
import numpy as np
import networkx as nx

rng = np.random.default_rng(7)

def one_basket():
    G = nx.Graph()
    G.add_edges_from([("一族", "a1"), ("一族", "a2"), ("a1", "拠点"), ("a2", "拠点"),
                      ("拠点", "b1"), ("拠点", "b2"), ("b1", "次の世代"), ("b2", "次の世代")])
    return G

def two_paths():
    G = nx.Graph()
    G.add_edges_from([("一族", "a1"), ("一族", "a2"), ("a1", "拠点１"), ("a2", "拠点２"),
                      ("拠点１", "b1"), ("拠点２", "b2"), ("b1", "次の世代"), ("b2", "次の世代"),
                      ("a1", "a2")])
    return G

def check(G, name):
    s, t = "一族", "次の世代"
    paths = list(nx.node_disjoint_paths(G, s, t))
    cut = nx.minimum_node_cut(G, s, t)
    print(f"{name}")
    print(f"   途中の点を共有しない道の最大の本数 {len(paths)}")
    for P in paths:
        print("     " + " → ".join(P))
    print(f"   切り離すためにふさぐ点の最小の数 {len(cut)}  ふさぐ点の例 {sorted(cut)}")
    H = G.copy(); H.remove_nodes_from(cut)
    print(f"   その点をふさぐと、つながっているか：{nx.has_path(H, s, t)}")
    return G

def survive_prob(G, q, trials=20000):
    """途中の点が、それぞれ確率 q でふさがれるとき、s と t がつながったままでいる確率。"""
    inner = [v for v in G if v not in ("一族", "次の世代")]
    ok = 0
    for _ in range(trials):
        blocked = [v for v in inner if rng.random() < q]
        H = G.copy(); H.remove_nodes_from(blocked)
        ok += nx.has_path(H, "一族", "次の世代")
    return ok / trials

G1 = check(one_basket(), "1. 道が１か所に集まる図")
G2 = check(two_paths(), "\n2. 道を２本に分けた図")

# メンガーの定理の一致を、小さな無作為の図で確かめる
agree, total = 0, 0
for _ in range(300):
    G = nx.gnp_random_graph(9, 0.35, seed=int(rng.integers(1e9)))
    s, t = 0, 8
    if G.has_edge(s, t) or not nx.has_path(G, s, t):
        continue
    a = len(list(nx.node_disjoint_paths(G, s, t)))
    b = len(nx.minimum_node_cut(G, s, t))
    agree += (a == b); total += 1
print(f"\n3. 無作為の小さな図で、２つの数が一致した回数：{total}回のうち {agree}回（つながっていて隣り合わない組だけを数えた）")

print("\n4. 途中の点が、それぞれ確率 q でふさがれるとき、つながったままでいる確率")
for q in [0.05, 0.1, 0.2]:
    print(f"   q = {q:.2f}  １か所に集まる図 {survive_prob(G1, q):.3f}   ２本に分けた図 {survive_prob(G2, q):.3f}")
print("   注意：点が別々にふさがれる、という仮定は、ふつう成り立たない。２つの拠点が同じ出来事で同時にふさがれる（共通の原因）なら、")
print("   ２本の道の強みは小さくなる。道が本当に独立かどうかは、数学ではなく、それぞれの拠点の制度と歴史の知識で確かめる。")
