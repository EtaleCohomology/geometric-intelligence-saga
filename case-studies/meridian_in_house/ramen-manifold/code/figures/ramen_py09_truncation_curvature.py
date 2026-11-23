"""第10章：有限の解像度、オリヴィエの曲率、割り算の空間と非可換トーラス。

1. スペクトルの切り捨て（コンヌとファン・スーレイコム）の円の例
   円の上の関数 f を、低いほうの N 個のフーリエの波だけの世界に閉じ込めると、
   N 行 N 列の行列 T_f になる。成分は f のフーリエ係数 f^(j-k) で、
   左上から右下への斜め（「＼」の向き）に、同じ数が並ぶ（テプリッツ行列）。
   もとの関数どうしは fg = gf なのに、T_f T_g と T_g T_f は一致しない。
2. オリヴィエの曲率 κ(x,y) = 1 - W1(m_x, m_y) / d(x,y)
   m_x は、x から１歩、でたらめに歩いた先の分布。W1 は線形計画法で解く。
3. 割り算の空間：円を無理数の角度で回し続けると、軌道は円のどこにでも近づく。
   非可換トーラスの有限の模型：時計の行列 U と、ずらしの行列 V で VU = e^{2πiθ} UV（θ = p/q）。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path
import jpfont


def toeplitz_truncation(coeffs, N):
    """coeffs：フーリエ係数 {k: f^(k)}。T_{jk} = f^(j - k)、j, k = 0..N-1。"""
    T = np.zeros((N, N), complex)
    for j in range(N):
        for k in range(N):
            T[j, k] = coeffs.get(j - k, 0)
    return T


def ollivier(A, x, y):
    n = len(A)
    Dm = shortest_path(A, unweighted=True)
    def walk(v):
        m = A[v] / A[v].sum()
        return m
    mx, my = walk(x), walk(y)
    # 輸送計画 P_ij >= 0、行の和 = mx、列の和 = my、費用 = Σ P_ij d(i,j)
    c = Dm.ravel()
    Aeq = []; beq = []
    for i in range(n):
        r = np.zeros((n, n)); r[i, :] = 1; Aeq.append(r.ravel()); beq.append(mx[i])
    for j in range(n):
        r = np.zeros((n, n)); r[:, j] = 1; Aeq.append(r.ravel()); beq.append(my[j])
    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=(0, None), method="highs")
    return 1 - res.fun / Dm[x, y]


def cycle(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def main():
    jpfont.setup()
    # 1. テプリッツ
    cos = {1: 0.5, -1: 0.5}
    sin = {1: -0.5j, -1: 0.5j}
    for N in (4, 8, 16):
        Tc, Ts = toeplitz_truncation(cos, N), toeplitz_truncation(sin, N)
        c = np.linalg.norm(Tc @ Ts - Ts @ Tc, 2)
        print(f"N = {N:2d}：||T_cos T_sin − T_sin T_cos|| = {c:.3f}（もとの関数どうしなら 0）")
        assert c > 0.1
    T = toeplitz_truncation({0: 1.0, 1: 0.3, -1: 0.3, 2: 0.1, -2: 0.1}, 6)
    assert all(np.allclose(np.diag(T, k), np.diag(T, k)[0]) for k in range(-5, 6))
    print("テプリッツ行列：左上から右下への斜めに、同じ数が並ぶ ― 確認済み")

    # 2. オリヴィエの曲率
    K5 = np.ones((5, 5)) - np.eye(5)
    star = np.zeros((5, 5)); star[0, 1:] = star[1:, 0] = 1
    tree = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]:
        tree[i, j] = tree[j, i] = 1
    vals = {
        "完全グラフ K5（丸い）": ollivier(K5, 0, 1),
        "輪 C8（平ら）": ollivier(cycle(8), 0, 1),
        "枝分かれする木（鞍の形）": ollivier(tree, 0, 1),
    }
    for k, v in vals.items():
        print(f"オリヴィエの曲率 {k}：{v:+.3f}")
    assert vals["完全グラフ K5（丸い）"] > 0 and abs(vals["輪 C8（平ら）"]) < 1e-9 and vals["枝分かれする木（鞍の形）"] < 0

    # 3. 無理数の回転と、非可換トーラスの有限の模型
    theta = (np.sqrt(5) - 1) / 2
    for p, q in ((3, 5), (5, 8), (8, 13)):
        w = np.exp(2j * np.pi * p / q)
        U = np.diag(w ** np.arange(q))
        V = np.roll(np.eye(q), -1, axis=0)  # V e_k = e_{k-1}
        assert np.allclose(V @ U, w * U @ V)
        print(f"θ ≈ {p}/{q}：VU = e^(2πiθ) UV を確認（θ の近似の誤差 {abs(theta - p/q):.4f}）")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im = axes[0].imshow(np.real(toeplitz_truncation({0: 1.0, 1: 0.6, -1: 0.6, 2: 0.25, -2: 0.25, 3: 0.1, -3: 0.1}, 10)),
                        cmap="viridis")
    axes[0].set_title("円の例：テプリッツ行列（＼の向きに同じ数）")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].bar(range(3), list(vals.values()), color=["#2b7bb9", "gray", "#d9822b"])
    axes[1].set_xticks(range(3), ["完全グラフ", "輪", "木"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("オリヴィエの曲率：正、0、負")
    n = np.arange(400)
    ang = 2 * np.pi * ((n * theta) % 1)
    r = 1 + n / 400  # 回した回数ごとに、半径を少しずつ外へずらして描く
    axes[2].scatter(r * np.cos(ang), r * np.sin(ang), s=6, c=n, cmap="plasma")
    axes[2].text(0, 0, "内側：最初の数回\n外側：400 回目まで", ha="center", va="center", fontsize=9)
    axes[2].set_aspect("equal")
    axes[2].set_title("無理数の角度で回し続ける：二度と戻らず、円を埋めていく")
    axes[2].axis("off")
    jpfont.save(fig, "ramen_py09_truncation_curvature.jpg")


if __name__ == "__main__":
    main()
