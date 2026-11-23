"""第10章：コンヌの距離と、網の目の上のディラック型の作用素 D の作り方。

コンヌの距離：d(p, q) = sup { |a(p) - a(q)| : ||[D, a]|| <= 1 }

長谷部の追伸の３つの作り方を、小さな網の目で計算する。
- 例１：点の上の関数に働く行列 D（つながった２点にだけ、線の長さの逆数を置く）
        d <= 網の目の距離。三角形（線の長さは全部 1）では √(2/3)。
        ２点だけの網の目（線１本）では、網の目の距離と一致する。
        一直線の３点の端と端では √2 で、網の目の距離 2 より短い（木の形でも一致しない）。
        出典：F. Besnard, Estimating noncommutative distances on graphs, arXiv:2105.09056
- ２点の空間：D = [[0, m], [m, 0]] なら、距離は 1/|m|。
- 例２：線を１本ずつ切り離して、線ごとの２点の空間の D を並べる。網の目の距離と一致する。
- 例３：D = d + d*（隣の点との差を取る作用素と、その随伴）。D^2 は、点の上では網の目のラプラシアン。
- ホッジの定理を網の目の上で言い直したもの（エックマン、1944年）：トーラスの三角形分割で、
  組み合わせのラプラシアンの固有値 0 の数は、0次元 1、1次元 2、2次元 1、合計 4。
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import jpfont

rng = np.random.default_rng(0)


def comm_norm(D, a):
    C = D * (a[None, :] - a[:, None])  # [D, a]_{ij} = D_ij (a_j - a_i)
    return np.linalg.norm(C, 2)


def connes_distance(D, p, q, starts=30):
    n = len(D)
    best = 0.0
    cons = {"type": "ineq", "fun": lambda a: 1 - comm_norm(D, a)}
    for _ in range(starts):
        a0 = rng.normal(size=n) * 0.3
        res = minimize(lambda a: -(a[p] - a[q]), a0, constraints=[cons], method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success and comm_norm(D, res.x) <= 1 + 1e-6:
            best = max(best, res.x[p] - res.x[q])
    return best


def edges_to_D(n, edges, lengths=None):
    D = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        w = 1.0 if lengths is None else lengths[k]
        D[i, j] = D[j, i] = 1 / w
    return D


def separated_D(n, edges):
    """例２：線ごとに、両端の２点だけの空間を作り、D を並べる（直和）。
    線 (i, j) のブロックの交換子のノルムは |a_i - a_j| なので、全体のノルムはその最大値。"""
    return lambda a: max(abs(a[i] - a[j]) for i, j in edges)


def separated_distance(n, edges, p, q):
    norm = separated_D(n, edges)
    cons = {"type": "ineq", "fun": lambda a: 1 - norm(a)}
    best = 0
    for _ in range(20):
        res = minimize(lambda a: -(a[p] - a[q]), rng.normal(size=n), constraints=[cons], method="SLSQP")
        if norm(res.x) <= 1 + 1e-6:
            best = max(best, res.x[p] - res.x[q])
    return best


def boundary_matrices_torus(m=3):
    """m×m の格子の正方形を対角線で２つの三角形に分けた、トーラスの三角形分割。"""
    V = [(i, j) for i in range(m) for j in range(m)]
    vid = {v: k for k, v in enumerate(V)}
    tris = []
    for i in range(m):
        for j in range(m):
            a = vid[(i, j)]; b = vid[((i + 1) % m, j)]
            c = vid[(i, (j + 1) % m)]; d = vid[((i + 1) % m, (j + 1) % m)]
            tris += [tuple(sorted((a, b, d))), tuple(sorted((a, c, d)))]
    edges = sorted({tuple(sorted(e)) for t in tris for e in itertools.combinations(t, 2)})
    eid = {e: k for k, e in enumerate(edges)}
    B1 = np.zeros((len(V), len(edges)))
    for k, (i, j) in enumerate(edges):
        B1[i, k], B1[j, k] = -1, 1
    B2 = np.zeros((len(edges), len(tris)))
    for k, (a, b, c) in enumerate(tris):
        B2[eid[(b, c)], k] += 1; B2[eid[(a, c)], k] -= 1; B2[eid[(a, b)], k] += 1
    assert np.allclose(B1 @ B2, 0)
    return B1, B2, len(V), len(edges), len(tris)


def main():
    jpfont.setup()
    tri = edges_to_D(3, [(0, 1), (1, 2), (0, 2)])
    d_tri = connes_distance(tri, 0, 1)
    print(f"例１ 三角形：コンヌの距離 {d_tri:.4f}（√(2/3) = {np.sqrt(2/3):.4f}）、網の目の距離 1")
    assert abs(d_tri - np.sqrt(2 / 3)) < 1e-3

    two = edges_to_D(2, [(0, 1)])
    d_two = connes_distance(two, 0, 1)
    print(f"例１ ２点（線１本）：{d_two:.4f}（網の目の距離 1 と一致）")
    assert abs(d_two - 1) < 1e-3

    path = edges_to_D(3, [(0, 1), (1, 2)])
    d_path = connes_distance(path, 0, 2)
    print(f"例１ 一直線の３点の端と端：{d_path:.4f}（√2 = {np.sqrt(2):.4f}）、網の目の距離 2")
    assert abs(d_path - np.sqrt(2)) < 1e-3

    for m in (0.5, 2.0, 10.0):
        d = connes_distance(np.array([[0, m], [m, 0]]), 0, 1)
        print(f"２点の空間 m = {m}：距離 {d:.4f}（1/|m| = {1/m:.4f}）")
        assert abs(d - 1 / m) < 1e-3

    d_sep = separated_distance(3, [(0, 1), (1, 2), (0, 2)], 0, 1)
    d_sep_path = separated_distance(3, [(0, 1), (1, 2)], 0, 2)
    print(f"例２ 線ごとに切り離した D：三角形 {d_sep:.4f}、一直線の３点の端と端 {d_sep_path:.4f}（網の目の距離と一致）")
    assert abs(d_sep - 1) < 1e-4 and abs(d_sep_path - 2) < 1e-4

    # 例３：D = d + d* の２乗
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    n = 4
    B = np.zeros((len(edges), n))  # 隣の点との差を取る作用素 d：点の関数 → 線の関数
    for k, (i, j) in enumerate(edges):
        B[k, i], B[k, j] = -1, 1
    Dfull = np.block([[np.zeros((n, n)), B.T], [B, np.zeros((len(edges), len(edges)))]])
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    L_graph = np.diag(A.sum(1)) - A
    assert np.allclose((Dfull @ Dfull)[:n, :n], L_graph)
    print("例３ D = d + d* の２乗の、点の上の部分 = 網の目のラプラシアン（次数 − 隣接）：確認済み")

    B1, B2, nv, ne, nt = boundary_matrices_torus(3)
    L0 = B1 @ B1.T; L1 = B1.T @ B1 + B2 @ B2.T; L2 = B2.T @ B2
    z = [int((np.linalg.eigvalsh(L) < 1e-9).sum()) for L in (L0, L1, L2)]
    print(f"トーラスの三角形分割（点 {nv}、線 {ne}、三角形 {nt}、オイラー標数 {nv-ne+nt}）："
          f"固有値 0 の数 {z[0]} + {z[1]} + {z[2]} = {sum(z)}")
    assert z == [1, 2, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["三角形\n（隣どうし）", "一直線の３点\n（端と端）", "２点\n（線１本）"]
    geo = [1, 2, 1]
    ex1 = [d_tri, d_path, d_two]
    ex2 = [d_sep, d_sep_path, 1.0]
    x = np.arange(3)
    axes[0].bar(x - 0.27, geo, 0.27, color="gray", label="網の目の距離")
    axes[0].bar(x, ex1, 0.27, color="#d9822b", label="例１の D から出る距離")
    axes[0].bar(x + 0.27, ex2, 0.27, color="#2b7bb9", label="例２の D から出る距離")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("D の作り方で、出てくる距離が変わる")
    axes[0].set_ylim(0, 2.7)
    axes[0].legend(fontsize=9, loc="upper left")
    ms = np.linspace(0.3, 5, 50)
    axes[1].plot(ms, 1 / ms, color="#2b7bb9", lw=2)
    axes[1].set_xlabel("m（２枚を結ぶ D の成分）")
    axes[1].set_ylabel("２枚のあいだの距離 1/|m|")
    axes[1].set_title("２点の空間：m が大きいほど、２枚は近い")
    jpfont.save(fig, "ramen_py08_connes_distance.jpg")


if __name__ == "__main__":
    main()
