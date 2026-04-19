# optimal_transport.py
import numpy as np
import ot

def w2_distance(X, Y):
    """
    経験分布間の Wasserstein-2 距離 (厳密).
    X, Y: (n, d), (m, d) numpy arrays.
    """
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric="sqeuclidean")  # (n, m)
    W2_sq = ot.emd2(a, b, M)
    return float(np.sqrt(W2_sq))


def sinkhorn_w2(X, Y, reg=0.1):
    """
    エントロピー正則化付き Wasserstein-2. 大規模データ向け高速化.
    """
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric="sqeuclidean")
    return float(np.sqrt(ot.sinkhorn2(a, b, M, reg=reg)))


def wasserstein_geodesic(X, Y, t_values):
    """
    McCann interpolation: μ_t = ((1-t) T_id + t T_*) # μ
    離散分布での 2-OT 最適カップリング T を使って、時点 t での中間分布を返す.
    """
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric="sqeuclidean")
    G = ot.emd(a, b, M)   # (n, m) 最適輸送計画
    # 各 x_i の輸送先を重み付き平均で
    # (離散近似: barycentric projection)
    T_X = (G @ Y) / a[:, None]   # (n, d)
    interpolations = []
    for t in t_values:
        mu_t = (1 - t) * X + t * T_X
        interpolations.append(mu_t)
    return interpolations


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(200, 3)
    Y = np.random.randn(200, 3) + np.array([2.0, 0.0, 0.0])
    print(f"W_2(X, Y) exact   = {w2_distance(X, Y):.4f}")
    print(f"W_2(X, Y) Sinkhorn = {sinkhorn_w2(X, Y, reg=0.1):.4f}")

    t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    geo = wasserstein_geodesic(X, Y, t_vals)
    print("Geodesic samples:", [g.mean(0) for g in geo])