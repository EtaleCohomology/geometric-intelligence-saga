"""第５章・第８章：専門家の座標の上の定規（フィッシャー計量）の階数。

専門家が付けた４つの軸の座標 x = (o, I, h, K) を、店の結果の確率モデルにつなぐ。
- 結果が「閉店した／しなかった」の二値だけ（ロジスティック・モデル）なら、
  フィッシャー計量 G(x) = p(1-p) w w^T は、どの点でも階数 1。
  「閉店する確率が変わる向き」にしか、長さを持たない。
- 選べる結果が K 通り（多項ロジット）なら、階数は高々 K-1。
- 向きの違う結果（口コミの点数、再来店までの間隔、１日の客数）を足すと、階数が上がる。

あわせて、第８章でリンが検算した、指数の項の大きさも確認する。
数値は、物語の中の架空の値である。
"""
import numpy as np
import matplotlib.pyplot as plt
import jpfont

rng = np.random.default_rng(4)
AXES = ["o（相補性）", "I（相互作用の強さ）", "h（ハイブリッド化）", "K（両立性）"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fisher_binary(x, w, b):
    p = sigmoid(w @ x + b)
    return p * (1 - p) * np.outer(w, w)


def fisher_multinomial(x, W):
    z = W @ x
    pi = np.exp(z - z.max()); pi /= pi.sum()
    return W.T @ (np.diag(pi) - np.outer(pi, pi)) @ W


def fisher_gaussian(w, sigma):
    # 口コミの点数 ~ N(w·x, sigma^2) のフィッシャー計量
    return np.outer(w, w) / sigma ** 2


def fisher_exponential(x, w):
    # 再来店までの間隔 ~ 指数分布、率 = exp(w·x)。フィッシャー計量は w w^T（x によらない）
    return np.outer(w, w)


def fisher_mc(x, w, b, n=200000):
    """二値モデルのフィッシャー計量を、スコアの２乗の平均で検算する。"""
    p = sigmoid(w @ x + b)
    y = rng.random(n) < p
    score = (y - p)[:, None] * w[None, :]
    return score.T @ score / n


def rank(G, tol=1e-9):
    ev = np.linalg.eigvalsh(G)
    return int((ev > tol * max(ev.max(), 1)).sum()), np.sort(ev)[::-1]


def main():
    jpfont.setup()
    x = np.array([0.8, 0.7, 0.8, 0.4])
    w_close = np.array([-0.5, 1.2, -0.3, -1.5]); b = -0.5
    G1 = fisher_binary(x, w_close, b)
    r1, ev1 = rank(G1)
    Gmc = fisher_mc(x, w_close, b)
    print("二値だけ：階数", r1, " 固有値", np.round(ev1, 4))
    print("  モンテカルロでの検算：最大の固有値", np.round(np.linalg.eigvalsh(Gmc).max(), 4))
    assert r1 == 1

    W3 = rng.normal(size=(3, 4))
    r3, ev3 = rank(fisher_multinomial(x, W3))
    print("結果が 3 通り：階数", r3, "（= 3 - 1）")
    assert r3 == 2

    G_all = G1 + fisher_gaussian(np.array([0.9, -0.2, 0.6, 0.8]), 0.5) \
        + fisher_exponential(x, np.array([0.3, 0.1, -0.9, 0.4])) \
        + fisher_gaussian(np.array([0.1, 0.9, 0.2, -0.3]), 0.4)
    r_all, ev_all = rank(G_all)
    print("向きの違う結果を４つ集めると：階数", r_all, " 固有値", np.round(ev_all, 3))
    assert r_all == 4

    # 指数の項 e^(1-K) は、K が 0 から 1 のとき、1 から e までしか動かない
    K = np.linspace(0.01, 1, 100)
    share_num = 1 / (1 + K)  # ln S の K への反応のうち、分子の K から来る割合
    print(f"e^(1-K) の範囲：{np.exp(0):.2f} 〜 {np.e:.2f}")
    print(f"分子の K から来る割合：K=1 で {share_num[-1]:.2f}、K=0.01 で {share_num[0]:.2f}（K が 1 より小さければ、いつも半分より多い）")
    assert (share_num >= 0.5).all()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = np.arange(4)
    ax.bar(idx - 0.2, ev1, 0.4, color="#d9822b", label="結果が「閉店したかどうか」だけ")
    ax.bar(idx + 0.2, ev_all, 0.4, color="#2b7bb9", label="向きの違う結果を４つ集めた")
    ax.set_xticks(idx, ["1 番目", "2 番目", "3 番目", "4 番目"])
    ax.set_ylabel("フィッシャー計量の固有値（目盛りの長さ）")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("二値の結果だけなら、定規は１つの向きにしか目盛りを持たない")
    ax.legend()
    jpfont.save(fig, "ramen_py06_fisher_rank.jpg")


if __name__ == "__main__":
    main()
