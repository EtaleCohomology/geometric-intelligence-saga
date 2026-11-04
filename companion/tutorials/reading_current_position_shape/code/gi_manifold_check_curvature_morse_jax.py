"""
連載第2回：答えが分かっている形（球の表面・馬の鞍の形）で、
  (1) 3つの条件を確かめる  (2) スカラー曲率を計算する  (3) モース理論で地形の要所を見つける
JAX版（筆者の環境で実行・出力確認済み）
使い方: python gi_manifold_check_curvature_morse_jax.py sphere 0.001
        python gi_manifold_check_curvature_morse_jax.py sphere 0.05
        python gi_manifold_check_curvature_morse_jax.py saddle 0.05
        python gi_manifold_check_curvature_morse_jax.py saddle 0.05 3   # 3つ目の引数：学習の乱数の種（省略すると1）
"""
import sys, numpy as np, jax, jax.numpy as jnp, optax
from sklearn.neighbors import NearestNeighbors
jax.config.update("jax_enable_x64", True)

SHAPE, BETA = sys.argv[1], float(sys.argv[2])   # 形と、KL情報量の重み（β）
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # 学習の乱数の種（データは同じまま、学習の初期値と途中の乱数だけを変える）
rng = np.random.default_rng(0)
D = 10                                           # データ空間の次元
Q, _ = np.linalg.qr(rng.normal(size=(D, 3)))     # 3次元の形を、距離を保ったまま10次元に埋め込む

# ---- 1. 答えが分かっている形のデータ ----
def make_sphere(n):        # 半径1の球の表面の一部（北極のまわり）。正解のスカラー曲率 R = 2
    th = np.arccos(rng.uniform(np.cos(1.0), 1.0, n)); ph = rng.uniform(0, 2*np.pi, n)
    return np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1)
def make_saddle(n, a=0.5): # 馬の鞍の形 z = a(x^2 - y^2)。正解の R = -2/(1+x^2+y^2)^2（負）
    x = rng.uniform(-1, 1, n); y = rng.uniform(-1, 1, n)
    return np.stack([x, y, a*(x**2 - y**2)], 1)
P = make_sphere(3000) if SHAPE == "sphere" else make_saddle(3000)
X = jnp.array(P @ Q.T + 0.002*rng.normal(size=(len(P), D)))
true_R = (lambda P: np.full(len(P), 2.0)) if SHAPE == "sphere" else (lambda P: -2.0/(1+P[:,0]**2+P[:,1]**2)**2)

# ---- 2. VAE（デコーダは tanh でなめらかにする）----
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
    z = mu + jnp.exp(0.5*lv) * jax.random.normal(k, mu.shape)          # 再パラメータ化トリック
    rec = jnp.mean(jnp.sum((mlp(p["dec"], z) - x)**2, 1))                # 再構成の誤差
    kl = jnp.mean(-0.5*jnp.sum(1 + lv - mu**2 - jnp.exp(lv), 1))         # KL情報量
    return rec + BETA*kl
opt = optax.adam(2e-3); st = opt.init(params)
@jax.jit
def step(p, s, k):
    l, g = jax.value_and_grad(loss)(p, X, k); u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
key = jax.random.PRNGKey(SEED + 1)
for i in range(6000):
    key, kk = jax.random.split(key); params, st, l = step(params, st, kk)
dec = lambda z: mlp(params["dec"], z)
mu = mlp(params["enc"], X)[:, :2]

# ---- 3. 3つの条件を確かめる ----
mse = float(jnp.mean(jnp.sum((dec(mu) - X)**2, 1)) / D)
k = 10
nx = NearestNeighbors(n_neighbors=k+1).fit(np.array(X)).kneighbors(np.array(X), return_distance=False)[:, 1:]
nz = NearestNeighbors(n_neighbors=k+1).fit(np.array(mu)).kneighbors(np.array(mu), return_distance=False)[:, 1:]
knn = np.mean([len(set(nx[i]) & set(nz[i]))/k for i in range(len(nx))])
min_sv = float(jnp.linalg.svd(jax.vmap(jax.jacfwd(dec))(mu), compute_uv=False).min())
print(f"[{SHAPE}, beta={BETA}] 条件1 MSE/次元 = {mse:.5f}")
print(f"[{SHAPE}, beta={BETA}] 条件2 KNN保存率 = {knn:.3f}  ({'合格' if knn >= 0.90 else '不合格'}：目安0.90以上)")
print(f"[{SHAPE}, beta={BETA}] 条件3 最小特異値 = {min_sv:.4f}")

# ---- 4. スカラー曲率（引き戻し計量 g = J^T J から、2回微分まで計算）----
def g(z): J = jax.jacfwd(dec)(z); return J.T @ J
def christoffel(z):                     # Γ^k_ij = 1/2 g^kl (∂_i g_jl + ∂_j g_il − ∂_l g_ij)
    gi = jnp.linalg.inv(g(z)); dg = jax.jacfwd(g)(z)       # dg[a,b,c] = ∂_c g_ab
    t = jnp.einsum("jli->ijl", dg) + jnp.einsum("ilj->ijl", dg) - dg
    return 0.5 * jnp.einsum("kl,ijl->kij", gi, t)
def scalar_curvature(z):
    G = christoffel(z); dG = jax.jacfwd(christoffel)(z)    # dG[k,i,j,m] = ∂_m Γ^k_ij
    R = (jnp.einsum("rnsm->rsmn", dG) - jnp.einsum("rmsn->rsmn", dG)
         + jnp.einsum("rml,lns->rsmn", G, G) - jnp.einsum("rnl,lms->rsmn", G, G))
    Ric = jnp.einsum("rsrn->sn", R)
    return jnp.einsum("sn,sn->", jnp.linalg.inv(g(z)), Ric)
idx = rng.choice(len(mu), 200, replace=False)
Rs = np.array(jax.vmap(scalar_curvature)(mu[idx])); T = true_R(P[idx])
print(f"[{SHAPE}, beta={BETA}] スカラー曲率 R の中央値 = {np.median(Rs):+.3f}  （正解の中央値 {np.median(T):+.3f}）")

# ---- 5. モース理論：分析者が選んだ「高さ」の谷底・峠・山頂を見つける ----
if SHAPE == "saddle":
    def height(z):          # 高さは分析者が決める。ここでは h(x, y) = (x^2 - 0.5)^2 + y^2 を選んだ
        p = Q.T @ dec(z); return (p[0]**2 - 0.5)**2 + p[1]**2
    gh, Hh = jax.jit(jax.grad(height)), jax.jit(jax.hessian(height))
    lo, hi = np.array(mu).min(0), np.array(mu).max(0); found = []
    for z0 in np.array(mu)[rng.choice(len(mu), 400, replace=False)]:
        z = jnp.array(z0)
        for _ in range(60):  # ニュートン法で「勾配 = 0」の点を探す
            s = jnp.linalg.solve(Hh(z), gh(z)); nrm = float(jnp.linalg.norm(s))
            z = z - (s if nrm < 0.2 else s/nrm*0.2)
            if float(jnp.linalg.norm(gh(z))) < 1e-10: break
        if float(jnp.linalg.norm(gh(z))) < 1e-7 and np.all(np.array(z) >= lo) and np.all(np.array(z) <= hi):
            if all(np.linalg.norm(np.array(z) - f[0]) > 1e-2 for f in found):
                n_neg = int((np.linalg.eigvalsh(np.array(Hh(z))) < 0).sum())   # 指数 = 負の固有値の数
                found.append((np.array(z), n_neg, np.array(Q.T @ dec(z))[:2], float(height(z))))
    names = {0: "谷底（指数0）", 1: "峠（指数1）", 2: "山頂（指数2）"}
    for z, n_neg, p, v in sorted(found, key=lambda t: t[1]):
        print(f"[{SHAPE}] 臨界点：{names[n_neg]}  元の形の上の位置 (x, y) = ({p[0]:+.3f}, {p[1]:+.3f})  高さ = {v:.4f}")
