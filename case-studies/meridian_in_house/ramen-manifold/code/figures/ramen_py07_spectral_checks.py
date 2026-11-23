"""第９章・第10章：ラプラシアンの響き方（固有値と固有関数）による、地図の品質の点検。

1. 島の数：網の目のラプラシアンの、固有値が 0 になる数 = つながった塊の数。
   近似ではなく、網の目なら、いつでも成り立つ。
2. データの側と、地図の側の響き方を比べる。
   - データの点は、円筒の面（周の長さ 2π、高さ 3）の上にある。
   - 正しい地図：円筒を、輪のまま潜在空間に写す（環の形の潜在空間）。
   - 引き裂いた地図：円筒を１か所で切って、長方形に開いてしまう。
   - どちらも、点は１つずつ対応している。網の目は、潜在空間で近い点どうしを結び、
     線の重みは、復号した点どうしの距離（引き戻し計量の近似）で測る。
   - 両方の側で、同じ近さの目盛り（ε）と、コイフマンとラフォンの正規化（α = 1）を使う。
3. チーガーの不等式の考え方：くびれが細いほど、0 の次に小さい固有値が小さい。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
import jpfont

rng = np.random.default_rng(10)


def knn_graph(Z, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    rows = np.repeat(np.arange(len(Z)), k)
    cols = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(Z), len(Z)))
    return ((A + A.T) > 0).astype(float)


def diffusion_spectrum(pairs_mask, X, eps, n_eig=10):
    """コイフマンとラフォンの正規化（α=1）をした、拡散のラプラシアンの小さいほうの固有値。
    pairs_mask：どの点どうしを結ぶか（潜在空間での近さで決める）
    X：線の重みを測るための点（復号した点）"""
    M = pairs_mask.tocoo()
    d2 = np.sum((X[M.row] - X[M.col]) ** 2, 1)
    K = np.zeros((len(X), len(X)))
    K[M.row, M.col] = np.exp(-d2 / eps)
    K += np.eye(len(X))
    q = K.sum(1)
    K1 = K / np.outer(q, q)               # 点の濃さを取り除く（α = 1）
    d = K1.sum(1)
    S = K1 / np.sqrt(np.outer(d, d))      # 対称にした形（固有値は P と同じ）
    ev = np.sort(1 - eigh(S, eigvals_only=True))[: n_eig]
    return 4 * ev / eps                   # ガウス核の場合の目盛りの合わせ方（Δ の近似）


def islands():
    A = np.vstack([rng.normal([0, 0], 0.3, (60, 2)), rng.normal([4, 0], 0.3, (60, 2))])
    G = knn_graph(A, 6)
    L = laplacian(G).toarray()
    ev = np.linalg.eigvalsh(L)
    n_zero = int((ev < 1e-9).sum())
    n_cc = connected_components(G)[0]
    print(f"島の点検：固有値が 0 になる数 {n_zero}、つながった塊の数 {n_cc}")
    assert n_zero == n_cc == 2
    return ev


def cylinder_maps(n=2000, k=300, eps=0.05):
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, 3, n)
    X = np.column_stack([np.cos(th), np.sin(th), z])        # データの点（円筒）
    good = np.column_stack([(1 + z / 3) * np.cos(th), (1 + z / 3) * np.sin(th)])  # 環の形の潜在空間
    torn = np.column_stack([th, z])                         # 長方形に開いた潜在空間
    ev_data = diffusion_spectrum(knn_graph(X, k), X, eps)
    ev_good = diffusion_spectrum(knn_graph(good, k), X, eps)
    ev_torn = diffusion_spectrum(knn_graph(torn, k), X, eps)
    # 理論値：円筒 k^2 + (pi m / 3)^2、開いた長方形 (n/2)^2 + (pi m / 3)^2（ノイマン境界）
    th_cyl = sorted([a * a + (np.pi * m / 3) ** 2 for a in range(4) for m in range(4)
                     for _ in ((0,) if a == 0 else (0, 1))])[:10]
    th_strip = sorted([(a / 2) ** 2 + (np.pi * m / 3) ** 2 for a in range(8) for m in range(4)])[:10]
    print("円筒の理論値      ", np.round(th_cyl, 2))
    # 注：有限の点の数と、有限の ε のため、網の目の値は理論値より数％〜10％ほど小さく出る。
    #     この点検で比べるのは、理論値とのずれではなく、データの側と地図の側のずれである。
    print("データの側        ", np.round(ev_data, 2))
    print("正しい地図の側    ", np.round(ev_good, 2))
    print("引き裂いた地図の側", np.round(ev_torn, 2))
    print("開いた長方形の理論値", np.round(th_strip, 2))
    return ev_data, ev_good, ev_torn, th_cyl, th_strip


def dumbbell(neck, n=900):
    pts = []
    while len(pts) < n:
        p = rng.uniform([-2.2, -1.1], [2.2, 1.1])
        in_left = np.hypot(p[0] + 1.2, p[1]) < 1
        in_right = np.hypot(p[0] - 1.2, p[1]) < 1
        in_neck = abs(p[0]) < 1.2 and abs(p[1]) < neck / 2
        if in_left or in_right or in_neck:
            pts.append(p)
    P = np.array(pts)
    return diffusion_spectrum(knn_graph(P, 60), P, 0.01, n_eig=3)[1]


def main():
    jpfont.setup()
    ev_is = islands()
    ev_data, ev_good, ev_torn, th_cyl, th_strip = cylinder_maps()
    gaps = np.abs(ev_good[:6] - ev_data[:6]).max(), np.abs(ev_torn[:6] - ev_data[:6]).max()
    print(f"小さいほうの 6 個の固有値の、データの側との最大のずれ：正しい地図 {gaps[0]:.3f}、引き裂いた地図 {gaps[1]:.3f}")
    assert gaps[1] > 3 * gaps[0]

    necks = [0.1, 0.2, 0.35, 0.5, 0.8, 1.2]
    lam2 = [dumbbell(w) for w in necks]
    print("くびれの幅と、0 の次に小さい固有値：", [(w, round(float(l), 3)) for w, l in zip(necks, lam2)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].bar(np.arange(1, 9), ev_is[:8], color="#2b7bb9")
    axes[0].set_title("島が２つ：固有値 0 が２つ")
    axes[0].set_xlabel("固有値の番号（小さい順）")
    i = np.arange(1, 9)
    axes[1].plot(i, ev_data[:8], "o-", color="black", label="データの側（円筒）")
    axes[1].plot(i, ev_good[:8], "s--", color="#2b7bb9", label="正しい地図の側")
    axes[1].plot(i, ev_torn[:8], "^--", color="#d9822b", label="引き裂いた地図の側")
    axes[1].set_title("響き方を、小さいほうから比べる")
    axes[1].set_xlabel("固有値の番号（小さい順）")
    axes[1].legend(fontsize=9)
    axes[2].plot(necks, lam2, "o-", color="#2b7bb9")
    axes[2].set_title("くびれが細いほど、0 の次の固有値は小さい")
    axes[2].set_xlabel("くびれの幅")
    axes[2].set_ylabel("0 の次に小さい固有値")
    jpfont.save(fig, "ramen_py07_spectral_checks.jpg")


if __name__ == "__main__":
    main()
