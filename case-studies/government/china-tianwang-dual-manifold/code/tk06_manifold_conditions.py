"""
敵の多様体 補助文書03 コード6：命題1.1の条件2と条件3 ── 検閲が「近傍保存率」と「最小特異値」に与える影響
架空のデータ：35個の指標のうち、本当に独立に動く方向（内在次元）は、正直なデータで35、検閲されたデータで26。
・条件3：データを主成分（線形の近似）で35次元に写したとき、最小の特異値が 0 に近ければ、その方向は「空回り」している。
・条件2：12近傍の保存率。都合の悪い指標を「平穏な値」にならすと、近傍の構造が単純になり、保存率が上がる。
数値は架空のもの。本物の VAE（非線形）の代わりに、考え方を見るための線形の近似を使う。
"""
import numpy as np
rng = np.random.default_rng(35)
n, D = 1200, 35
def make(intrinsic):
    Z = rng.standard_normal((n, intrinsic))
    A = rng.standard_normal((intrinsic, D))
    return Z @ A + 0.01 * rng.standard_normal((n, D))
for name, k in [("正直なデータ（内在次元35）", 35), ("検閲されたデータ（内在次元26）", 26)]:
    X = make(k); X = X - X.mean(0)
    s = np.linalg.svd(X / np.sqrt(n), compute_uv=False)
    print(f"{name}: 35番目（最小）の特異値 {s[-1]:.4f}   26番目 {s[25]:.4f}   27番目 {s[26]:.4f}")
print("→ 内在次元が26しかないと、27〜35番目の特異値がほぼ0になり、35次元の地図は9次元ぶん空回りする。")

def knn_preservation(X, Y, k=12):
    def nn(M):
        d = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]
    a, b = nn(X), nn(Y)
    return np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))])
# 現実のデータ：10個の指標が、それぞれ独立に揺れる（地域ごとの不揃いな実態）
m = 400
honest = np.vstack([rng.standard_normal((m, 10)) * 0.8, rng.standard_normal((m, 10)) * 0.8 + 3.0])
# 検閲されたデータ：都合の悪い8つの指標（社会不安、隠れ債務など）が、上からの報告で「平穏な値」にならされる
censored = honest.copy(); censored[:, 2:] = censored[:, 2:].mean(0) + 0.05 * rng.standard_normal((2 * m, 8))
def pca2(X):
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); return Xc @ Vt[:2].T
print(f"近傍保存率（正直なデータを2次元に圧縮）: {knn_preservation(honest, pca2(honest)):.3f}")
print(f"近傍保存率（ならされた検閲データを2次元に圧縮）: {knn_preservation(censored, pca2(censored)):.3f}")
print("→ 都合の悪い指標がならされると、近傍の構造が単純になり、保存率は『良く』見える。")
