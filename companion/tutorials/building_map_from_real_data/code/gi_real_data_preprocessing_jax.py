"""
連載第4回：本物の社会経済データに近い「乱れたデータ」で、前処理の違いが3つの条件をどう変えるかを確かめる（JAX）
答えが分かっている形（馬の鞍の形）を10個の指標に写し、実データでよく起きる乱れを加える：
  単位と桁の違い、右に長く裾を引く金額の指標、外れ値、欠測、四半期にしか観測されない指標
使い方: python gi_real_data_preprocessing_jax.py naive 0.05
        python gi_real_data_preprocessing_jax.py zscore 0.05
        python gi_real_data_preprocessing_jax.py careful 0.05
        python gi_real_data_preprocessing_jax.py careful_mask 0.05
        python gi_real_data_preprocessing_jax.py careful_quarterly 0.05
        python gi_real_data_preprocessing_jax.py careful_quarterly 0.05 3   # 3つ目の引数：学習の乱数の種（省略すると1）
"""
import sys, numpy as np, jax, jax.numpy as jnp, optax
from sklearn.neighbors import NearestNeighbors
jax.config.update("jax_enable_x64", True)

MODE, BETA = sys.argv[1], float(sys.argv[2])       # 前処理の方法と、KL情報量の重み（β）
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(0); D, N = 10, 3000

# ---- 1. 答えが分かっている形（馬の鞍の形）を、10個の指標に写す ----
x = rng.uniform(-1, 1, N); y = rng.uniform(-1, 1, N)
P = np.stack([x, y, 0.5*(x**2 - y**2)], 1)             # 本当の位置（答え）
Q, _ = np.linalg.qr(rng.normal(size=(D, 3)))
clean = P @ Q.T                                        # 乱れのない10個の指標

# ---- 2. 実データでよく起きる乱れを加える ----
raw = clean.copy()
skewed = [0, 1, 2]                                     # 金額のように、右に長く裾を引く指標
raw[:, skewed] = np.exp(1.5*raw[:, skewed])
scale = 10.0**np.array([6, 6, 3, 0, 0, 2, -2, 1, 4, 0])  # 単位と桁がばらばら（円、千円、比率など）
raw = raw * scale
out = rng.random(raw.shape) < 0.01                      # 1%の値が、入力の誤りなどで大きく外れる
raw[out] = raw[out] * rng.choice([-20, 20], out.sum())
miss = rng.random(raw.shape) < 0.10                     # 10%の値が、ばらばらに欠けている
quarterly = [7, 8, 9]                                   # 3つの指標は、四半期にしか観測されない（3回に2回は欠ける）
miss[:, quarterly] |= (np.arange(N) % 3 != 0)[:, None]
obs = ~miss

# ---- 3. 前処理 ----
def preprocess(raw, obs, mode):
    Xp = raw.copy()
    if mode == "clean":                                 # 参考：乱れを加える前のデータ（実務では手に入らない）
        return clean.copy(), np.ones_like(obs, dtype=float)
    if mode == "naive":                                 # 欠けた値を0で埋めるだけ
        Xp[~obs] = 0.0; return Xp, np.ones_like(obs)
    if mode == "zscore":                                # 列の平均で埋めて、平均0・標準偏差1にそろえる
        for j in range(D):
            m, s = Xp[obs[:, j], j].mean(), Xp[obs[:, j], j].std()
            Xp[:, j] = (Xp[:, j] - m) / s
        Xp[~obs] = 0.0; return Xp, np.ones_like(obs)
    if mode in ("careful", "careful_mask"):             # 対数変換、外れ値に強い標準化、外れ値を欠測に回す、欠測を学習で使わない
        for j in range(D):
            v = Xp[obs[:, j], j]
            if j in skewed:
                Xp[:, j] = np.log(np.clip(Xp[:, j], 1e-12, None)); v = Xp[obs[:, j], j]
            med = np.median(v); iqr = np.subtract(*np.percentile(v, [75, 25]))
            Xp[:, j] = (Xp[:, j] - med) / (iqr / 1.349)      # 中央値と四分位範囲で、外れ値に引きずられずに標準化する
        obs2 = obs & (np.abs(Xp) < 5.0)                   # 標準化した値が±5を超えるものは、外れ値として「欠けた値」に回す
        Xp[~obs2] = 0.0; return Xp, obs2.astype(float)
if MODE == "careful_quarterly":                         # 粒度をそろえる：四半期の指標が観測された時点だけを使う
    keep = np.arange(N) % 3 == 0
    P, clean, raw, obs = P[keep], clean[keep], raw[keep], obs[keep]; N = int(keep.sum())
X_np, M_np = preprocess(raw, obs, "careful_mask" if MODE == "careful_quarterly" else MODE)
X, M = jnp.array(X_np), jnp.array(M_np)
USE_MASK = MODE in ("careful_mask", "careful_quarterly")                       # 「どの値が欠けているか」の印を、エンコーダにも渡す
XIN = jnp.concatenate([X, M], 1) if USE_MASK else X

# ---- 4. VAE（デコーダは tanh でなめらかにする。欠けた値は、損失の計算から外せる）----
def init(key, sizes):
    ps = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (m, n))*np.sqrt(1/m), jnp.zeros(n)))
    return ps
def mlp(ps, v):
    for W, b in ps[:-1]: v = jnp.tanh(v @ W + b)
    W, b = ps[-1]; return v @ W + b
k1, k2 = jax.random.split(jax.random.PRNGKey(SEED))
params = {"enc": init(k1, [XIN.shape[1], 64, 64, 4]), "dec": init(k2, [2, 64, 64, D])}
def loss(p, k):
    h = mlp(p["enc"], XIN); mu, lv = h[:, :2], h[:, 2:]
    z = mu + jnp.exp(0.5*lv) * jax.random.normal(k, mu.shape)
    rec = jnp.mean(jnp.sum(M * (mlp(p["dec"], z) - X)**2, 1))      # 観測された値だけで誤差を測る
    kl = jnp.mean(-0.5*jnp.sum(1 + lv - mu**2 - jnp.exp(lv), 1))
    return rec + BETA*kl
opt = optax.adam(2e-3); st = opt.init(params)
@jax.jit
def step(p, s, k):
    l, g = jax.value_and_grad(loss)(p, k); u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
key = jax.random.PRNGKey(SEED + 1)
for i in range(6000):
    key, kk = jax.random.split(key); params, st, l = step(params, st, kk)
dec = lambda z: mlp(params["dec"], z)
mu = mlp(params["enc"], XIN)[:, :2]

# ---- 5. 3つの条件を確かめる ----
mse = float(jnp.sum(M * (dec(mu) - X)**2) / jnp.sum(M))
from sklearn.metrics.pairwise import nan_euclidean_distances
def neighbors(A, k=10):                       # 欠けた値を除いて距離を測り、近い順に k 点を選ぶ
    Dm = nan_euclidean_distances(A); np.fill_diagonal(Dm, np.inf); return np.argsort(Dm, 1)[:, :k]
def knn_rate(na, nb, k=10):
    return np.mean([len(set(na[i]) & set(nb[i]))/k for i in range(len(na))])
Xnan = np.where(M_np > 0, X_np, np.nan)       # 欠けた値（外れ値として外した値を含む）は、距離の計算から除く
nz = neighbors(np.array(mu))
knn_data = knn_rate(neighbors(Xnan), nz)      # 実務で測れる値：前処理したデータと、潜在空間の近さの一致
knn_true = knn_rate(neighbors(P), nz)         # 答えが分かっているから測れる値：本当の位置と、潜在空間の近さの一致
min_sv = float(jnp.linalg.svd(jax.vmap(jax.jacfwd(dec))(mu), compute_uv=False).min())
print(f"[{MODE}, beta={BETA}, seed={SEED}] 条件1 MSE/観測値 = {mse:.4f}")
print(f"[{MODE}, beta={BETA}, seed={SEED}] 条件2 KNN保存率（前処理したデータとの比較） = {knn_data:.3f}  ({'合格' if knn_data >= 0.90 else '不合格'}：目安0.90以上)")
print(f"[{MODE}, beta={BETA}, seed={SEED}] 参考 KNN保存率（本当の位置との比較。答えが分かっているときだけ測れる） = {knn_true:.3f}")
print(f"[{MODE}, beta={BETA}, seed={SEED}] 条件3 最小特異値 = {min_sv:.4f}")
