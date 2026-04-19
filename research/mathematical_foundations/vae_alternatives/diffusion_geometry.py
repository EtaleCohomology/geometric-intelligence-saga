# diffusion_geometry.py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh

def diffusion_map_scratch(X, n_components=2, epsilon=None, alpha=1.0):
    """
    Diffusion map をゼロから実装。alpha=1 で Laplace-Beltrami に収束 (Coifman-Lafon).
    
    Args:
        X: (N, D)
        n_components: 埋め込み次元
        epsilon: カーネル幅 (None なら median heuristic)
        alpha: 正規化指数 ∈ [0, 1]
    
    Returns:
        embedding: (N, n_components)
        eigenvalues: (n_components + 1,)
    """
    N = len(X)
    D2 = cdist(X, X, metric="sqeuclidean")
    if epsilon is None:
        epsilon = np.median(D2[D2 > 0])
    K = np.exp(-D2 / epsilon)

    # alpha-normalization
    q = K.sum(axis=1)
    K_alpha = K / np.outer(q**alpha, q**alpha)

    # Markov normalization
    d = K_alpha.sum(axis=1)
    P = K_alpha / d[:, None]

    # 対称化版 (数値安定): A = D^{1/2} P D^{-1/2}
    # A の固有値 = P の固有値、対称なので eigsh 使用可
    A = (d[:, None] ** 0.5) * P / (d[None, :] ** 0.5)
    A = 0.5 * (A + A.T)  # 対称化 (数値誤差吸収)

    # 上位 k+1 固有対 (最大固有値 1 を除く)
    eigvals, eigvecs = eigsh(A, k=n_components + 1, which="LM")
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 対応する P の右固有ベクトルへ変換
    psi = eigvecs / (d[:, None] ** 0.5)

    # 最初の固有ベクトル (trivial, 定数) を除外
    embedding = psi[:, 1:n_components + 1] * eigvals[1:n_components + 1]
    return embedding, eigvals


def compare_pullback_vs_diffusion(X, vae, n_components=2):
    """
    VAE pullback 距離と diffusion 距離の相関を測定。
    高い相関 → 二手法が一致した幾何を抽出。
    """
    import torch
    # VAE encoder での埋め込み
    with torch.no_grad():
        mu, _ = vae.encode(torch.tensor(X, dtype=torch.float32))
    Z_vae = mu.numpy()

    # Diffusion map での埋め込み
    Z_diff, _ = diffusion_map_scratch(X, n_components=n_components)

    # ペア距離を比較 (subsample for tractability)
    idx = np.random.choice(len(X), 200, replace=False)
    D_vae  = cdist(Z_vae[idx],  Z_vae[idx])
    D_diff = cdist(Z_diff[idx], Z_diff[idx])
    corr = np.corrcoef(D_vae.ravel(), D_diff.ravel())[0, 1]
    return corr


if __name__ == "__main__":
    from toy_data import make_swiss_roll
    X, t = make_swiss_roll(n_samples=800, noise=0.05)
    emb, eigvals = diffusion_map_scratch(X, n_components=2)
    print("Diffusion map embedding shape:", emb.shape)
    print("Top 5 eigenvalues:", eigvals[:5])
    # emb[:, 0] vs true t で単調性を確認 (Swiss Roll の場合)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(emb[:, 0], t)
    print(f"Spearman rank correlation with true t: {rho:.3f}")