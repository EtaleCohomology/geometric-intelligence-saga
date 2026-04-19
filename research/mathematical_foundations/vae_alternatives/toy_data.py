# toy_data.py
import numpy as np
import torch

def make_swiss_roll(n_samples=1000, noise=0.05, seed=0):
    """
    3D Swiss Roll: 2D 多様体が 3D に埋め込まれた古典的 toy データ。
    GI 理論の検証として最適: 真の内在次元 = 2、曲率は場所依存。
    
    Returns:
        X: (n_samples, 3) データ点
        t: (n_samples,) 真の内在パラメータ (後で色付けに使う)
    """
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n_samples))
    h = 21 * rng.random(n_samples)
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    X = np.stack([x, y, z], axis=1)
    X += noise * rng.standard_normal(X.shape)
    X = X.astype(np.float32)
    return X, t.astype(np.float32)


def make_gaussian_mixture(n_samples=1000, n_components=3, dim=5, seed=0):
    """
    高次元ガウス混合: 連結性が明確でない設定。
    Regime change 検出やクラスタ分離の検証用。
    """
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_components, dim)) * 3
    covs = [np.eye(dim) * 0.5 for _ in range(n_components)]
    labels = rng.integers(0, n_components, n_samples)
    X = np.zeros((n_samples, dim))
    for i, label in enumerate(labels):
        X[i] = rng.multivariate_normal(means[label], covs[label])
    return X.astype(np.float32), labels


def to_torch(X, device="cpu", dtype=torch.float32):
    return torch.tensor(X, dtype=dtype, device=device)


if __name__ == "__main__":
    X, t = make_swiss_roll(n_samples=500)
    print(f"Swiss Roll: X.shape = {X.shape}, X.dtype = {X.dtype}")
    print(f"Range per dim: {X.min(0)} to {X.max(0)}")