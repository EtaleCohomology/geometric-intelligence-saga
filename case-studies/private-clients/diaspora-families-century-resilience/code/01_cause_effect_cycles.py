"""01 原因と結果の輪（閉路）を見つけ、輪の強さを測る（本編第3章）。

本編の「輪１〜輪３」を、原因から結果へ矢印を引いた図（有向グラフ）にする。
矢印の重みは「原因が１だけ強まったとき、結果がどれだけ強まるか」の仮の値（すべて架空）。
1. 閉路（矢印をたどって元に戻る道）をすべて列挙する（ジョンソンの方法。networkx の simple_cycles）
2. 閉路ごとに、重みの積（一周したときに、最初の強まりが何倍になって戻るか）を計算する
3. 図全体の跳ね返りの強さとして、重みの行列のスペクトル半径（固有値の絶対値の最大）を計算する
   スペクトル半径が 1 以上なら、跳ね返りの足し合わせ I + W + W^2 + ... は収まらない（どこまでも強まる）
4. 矢印の重みを年ごとに変えて、輪がいつ「自分自身を強める」水準に近づくかを見る
"""
import numpy as np
import networkx as nx

nodes = ["暮らしの悪化", "人々の不満", "少数派を責める言葉", "財産の没収や暴力",
         "少数派の出国", "国と国の緊張", "『内なる敵』の名指し", "監視と差別"]
idx = {n: i for i, n in enumerate(nodes)}

def build(scale):
    """scale: 年ごとの、矢印の強さの倍率（架空）。"""
    e = {  # 輪１
        ("暮らしの悪化", "人々の不満"): 0.6,
        ("人々の不満", "少数派を責める言葉"): 0.5,
        ("少数派を責める言葉", "財産の没収や暴力"): 0.4,
        ("財産の没収や暴力", "少数派の出国"): 0.7,
        ("少数派の出国", "暮らしの悪化"): 0.3,
        # 輪２
        ("国と国の緊張", "『内なる敵』の名指し"): 0.5,
        ("『内なる敵』の名指し", "監視と差別"): 0.6,
        ("監視と差別", "国と国の緊張"): 0.2,
        # ２つの輪をつなぐ矢印
        ("『内なる敵』の名指し", "少数派を責める言葉"): 0.4,
    }
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (a, b), w in e.items():
        G.add_edge(a, b, weight=w * scale)
    return G

G = build(1.0)
print("閉路の一覧（重みの積 = 一周したときの倍率）")
for c in nx.simple_cycles(G):
    gain = np.prod([G[c[i]][c[(i + 1) % len(c)]]["weight"] for i in range(len(c))])
    print(f"  長さ {len(c)}  倍率 {gain:.4f}  : " + " → ".join(c) + " → " + c[0])

def spectral_radius(G):
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    return max(abs(np.linalg.eigvals(W)))

print("\n閉路の長さを k、倍率を g とすると、その閉路だけの図のスペクトル半径は g の k 乗根になる")
print("  輪１：0.0252 の 5 乗根 =", f"{0.0252 ** (1 / 5):.3f}", "／ 輪２：0.0600 の 3 乗根 =", f"{0.06 ** (1 / 3):.3f}")
print("\n年ごとに矢印が強まる場合（年と倍率は架空）")
for year, s in zip(range(2048, 2058), np.linspace(1.0, 2.6, 10)):
    r = spectral_radius(build(s))
    note = "  ← 1 以上：跳ね返りの足し合わせが収まらない" if r >= 1 else ""
    print(f"  {year}年  倍率 {s:.2f}  スペクトル半径 {r:.3f}{note}")

# 跳ね返りの足し合わせ (I - W)^{-1}。W[i, j] は i から j への矢印の重みなので、
# (I - W)^{-1} の (i, j) 成分は、i から j へのすべての道の重みの積の合計になる。
#「暮らしの悪化」に 1 の打撃を与えたとき、最終的に各所がどれだけ強まるか
W = nx.to_numpy_array(build(1.6), nodelist=nodes, weight="weight")
L = np.linalg.inv(np.eye(len(nodes)) - W)
print("\n倍率 1.6 の年に『暮らしの悪化』へ 1 の打撃を与えたときの、最終的な強まり（直接 + 跳ね返り）")
for n in nodes:
    print(f"  {n:14s} {L[idx['暮らしの悪化'], idx[n]]:.3f}")
