"""連載第3回 実験1（改）：丘のある地形で、測地線が「丘を避けて回り込む」ことを確かめる（JAX）
地形 z = 0.8·exp(-(x²+y²)/0.18)（中央に急な丘）。この面の点を、距離を保ったまま10次元に置いてVAEを学習する。
丘をはさんだ2点について、(i) 正しい地形の上の測地線（答え）と、(ii) 学習した地図の上の測地線を比べる。"""
import sys, numpy as np, jax, jax.numpy as jnp, optax
from sklearn.neighbors import NearestNeighbors
jax.config.update("jax_enable_x64", True)
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # 学習の乱数の種（データは同じまま、学習の初期値と途中の乱数だけを変える）
rng = np.random.default_rng(0); D = 10
Q, _ = np.linalg.qr(rng.normal(size=(D, 3)))
surf = lambda x, y: 0.8*jnp.exp(-(x**2 + y**2)/0.18)
xy = rng.uniform(-1, 1, (3000, 2))
P = np.column_stack([xy, np.array(surf(xy[:, 0], xy[:, 1]))])
X = jnp.array(P @ Q.T + 0.002*rng.normal(size=(len(P), D)))
def init(key, sizes):
    ps = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (m, n))*np.sqrt(1/m), jnp.zeros(n)))
    return ps
def mlp(ps, x):
    for W, b in ps[:-1]: x = jnp.tanh(x @ W + b)
    W, b = ps[-1]; return x @ W + b
k1, k2 = jax.random.split(jax.random.PRNGKey(SEED))
params = {"enc": init(k1, [D, 64, 64, 4]), "dec": init(k2, [2, 64, 64, D])}
def loss(p, x, k):
    h = mlp(p["enc"], x); mu, lv = h[:, :2], h[:, 2:]
    z = mu + jnp.exp(0.5*lv) * jax.random.normal(k, mu.shape)
    return jnp.mean(jnp.sum((mlp(p["dec"], z) - x)**2, 1)) + BETA*jnp.mean(-0.5*jnp.sum(1 + lv - mu**2 - jnp.exp(lv), 1))
opt = optax.adam(2e-3); st = opt.init(params)
@jax.jit
def step(p, s, k):
    l, g = jax.value_and_grad(loss)(p, X, k); u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
key = jax.random.PRNGKey(SEED + 1)
for i in range(6000):
    key, kk = jax.random.split(key); params, st, l = step(params, st, kk)
dec = lambda z: mlp(params["dec"], z)
mu = mlp(params["enc"], X)[:, :2]
# 3つの条件
mse = float(jnp.mean(jnp.sum((dec(mu) - X)**2, 1)) / D); k = 10
nx = NearestNeighbors(n_neighbors=k+1).fit(np.array(X)).kneighbors(np.array(X), return_distance=False)[:, 1:]
nz = NearestNeighbors(n_neighbors=k+1).fit(np.array(mu)).kneighbors(np.array(mu), return_distance=False)[:, 1:]
knn = np.mean([len(set(nx[i]) & set(nz[i]))/k for i in range(len(nx))])
min_sv = float(jnp.linalg.svd(jax.vmap(jax.jacfwd(dec))(mu), compute_uv=False).min())
print(f"[beta={BETA}] 条件1 MSE/次元={mse:.5f}  条件2 KNN保存率={knn:.3f}  条件3 最小特異値={min_sv:.4f}")

N = 64
def geodesic(embed, za, zb, iters=5000, lr=1e-3):
    t = jnp.linspace(0, 1, N)[:, None]; straight = za*(1 - t) + zb*t
    def energy(inner):
        Y = jax.vmap(embed)(jnp.vstack([za, inner, zb]))
        return jnp.sum(jnp.sum((Y[1:] - Y[:-1])**2, axis=1)) * (N - 1)
    inner = straight[1:-1]; o = optax.adam(lr); s = o.init(inner)
    @jax.jit
    def gs(inner, s):
        g = jax.grad(energy)(inner); u, s = o.update(g, s); return optax.apply_updates(inner, u), s
    for _ in range(iters): inner, s = gs(inner, s)
    return straight, jnp.vstack([za, inner, zb])
length = lambda embed, Z: float(jnp.sum(jnp.linalg.norm(jax.vmap(embed)(Z)[1:] - jax.vmap(embed)(Z)[:-1], axis=1)))
true_embed = lambda p: jnp.array([p[0], p[1], surf(p[0], p[1])])        # 正しい地形（答え）
A, B = np.array([-0.8, 0.05]), np.array([0.8, -0.05])
s_true, g_true = geodesic(true_embed, jnp.array(A), jnp.array(B))
# 学習した地図の上で：A と B に最も近いデータ点の潜在座標から出発
ia = int(np.argmin(np.linalg.norm(xy - A, axis=1))); ib = int(np.argmin(np.linalg.norm(xy - B, axis=1)))
za, zb = mu[ia], mu[ib]
s_lat, g_lat = geodesic(dec, za, zb)
to_xy = lambda Z: np.array(jax.vmap(lambda z: (jnp.array(Q).T @ dec(z))[:2])(Z))
gl_xy = to_xy(g_lat); sl_xy = to_xy(s_lat)
print("[正しい地形（答え）]")
print(f"  まっすぐ丘を越える道の長さ = {length(true_embed, s_true):.4f}")
print(f"  測地線の長さ               = {length(true_embed, g_true):.4f}   測地線が中央からいちばん離れる距離 = {float(np.max(np.abs(np.array(g_true)[:,1]))):.3f}")
print("[学習した地図]")
print(f"  潜在空間のまっすぐな線の長さ（定規で測る） = {length(dec, s_lat):.4f}   その道の中央からのずれ = {float(np.max(np.abs(sl_xy[:,1]))):.3f}")
print(f"  学習した地図の上の測地線の長さ             = {length(dec, g_lat):.4f}   測地線が中央からいちばん離れる距離 = {float(np.max(np.abs(gl_xy[:,1]))):.3f}")
# 学習した測地線の、答えの測地線からのずれ（元の地形の (x, y) で比べる）
gt = np.array(g_true); dev = np.max([np.min(np.linalg.norm(gt - p, axis=1)) for p in gl_xy])
print(f"  学習した測地線と、答えの測地線のずれ（最大） = {dev:.3f}")
