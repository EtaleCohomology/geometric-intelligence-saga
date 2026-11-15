"""
ラベルで VAE の潜在空間に意味を与える ── 少数のラベルで、軸に意味が出てくる様子を確かめる（JAX）

使い方：
  python gi_few_labels_jax.py            # すべての実験
  python gi_few_labels_jax.py labels     # 実験1：ラベルの数を変える（情報量と、軸への並び）
  python gi_few_labels_jax.py compare    # 実験2：同時に学習する ／ 後から条件づける（乱数の種を5通り）
  python gi_few_labels_jax.py manipulate # 実験2b：軸や方向を動かすと、ねらった要因だけが動くか
  python gi_few_labels_jax.py correlated # 実験3：要因どうしに相関があるデータ

架空のデータ：意味のある4つの要因 y（0〜1）と、意味のない2つの要因 h から、30個の指標 x を作る。
潜在空間は8次元。最初の4つの軸に、y の意味を持たせたい（残りの4つの軸は自由）。数値はすべて架空のもの。
"""
import sys
import jax, jax.numpy as jnp
import optax

N, D, E, F = 3000, 30, 4, 4
TE = slice(200, N)                     # ラベルを使っていない2,800件で評価する（ラベルは先頭の事例にだけ付ける）

# ---------- 架空のデータ ----------
def make_mixing(key):
    k3, k4 = jax.random.split(key)
    return jax.random.normal(k3, (6, 40)) * 0.8, jax.random.normal(k4, (40, D)) * 0.3

def make_x(key, y, A, B):
    k2, k5 = jax.random.split(key)
    h = jax.random.normal(k2, (y.shape[0], 2))
    return jnp.tanh(jnp.concatenate([y * 2 - 1, h * 0.5], 1) @ A) @ B + 0.02 * jax.random.normal(k5, (y.shape[0], D))

def make_data(key):                    # 第1版と同じデータ（要因どうしは独立）
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    y = jax.random.uniform(k1, (N, 4)); h = jax.random.normal(k2, (N, 2))
    A = jax.random.normal(k3, (6, 40)) * 0.8; B = jax.random.normal(k4, (40, D)) * 0.3
    x = jnp.tanh(jnp.concatenate([y * 2 - 1, h * 0.5], 1) @ A) @ B + 0.02 * jax.random.normal(k5, (N, D))
    return x, y

def correlated_y(key, n, rho):         # 要因1と要因2の相関が rho（周辺分布は0〜1の一様分布）
    C = jnp.eye(4).at[0, 1].set(rho).at[1, 0].set(rho)
    g = jax.random.normal(key, (n, 4)) @ jnp.linalg.cholesky(C).T
    return jax.scipy.stats.norm.cdf(g)

# ---------- VAE ----------
def mlp_init(key, sizes):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (a, b)) / jnp.sqrt(a), jnp.zeros(b)))
    return ps

def mlp(ps, x):
    for W, c in ps[:-1]: x = jnp.tanh(x @ W + c)
    W, c = ps[-1]; return x @ W + c

def train(x, y, n_lab, init_seed=2, noise_seed=9, steps=4000):
    """n_lab > 0：同時に学習する（ラベルの付いた事例では、最初の4つの軸とラベルの差の2乗を加える）。
       n_lab = 0：ラベルを使わない、ふつうの VAE。"""
    k1, k2 = jax.random.split(jax.random.PRNGKey(init_seed))
    p = {"enc": mlp_init(k1, [D, 64, 2 * (E + F)]), "dec": mlp_init(k2, [E + F, 64, D])}
    yl = y[:max(n_lab, 1)]
    def loss(pp, key=jax.random.PRNGKey(noise_seed)):
        out = mlp(pp["enc"], x); mu, lv = out[:, :E + F], out[:, E + F:]
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        ze, zf = jax.nn.sigmoid(z[:, :E]), z[:, E:]                       # 意味を持たせる軸（0〜1）と、自由な軸
        rec = jnp.mean(jnp.sum((mlp(pp["dec"], jnp.concatenate([ze, zf], 1)) - x) ** 2, 1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), 1))
        sup = jnp.mean(jnp.sum((jax.nn.sigmoid(mu[:max(n_lab, 1), :E]) - yl) ** 2, 1)) if n_lab > 0 else 0.0
        return rec + 0.01 * kl + 20.0 * sup
    opt = optax.adam(3e-3); st = opt.init(p)
    @jax.jit
    def step(p, st):
        g = jax.grad(loss)(p); u, st = opt.update(g, st); return optax.apply_updates(p, u), st
    for _ in range(steps): p, st = step(p, st)
    return p

def latent(p, x):
    return mlp(p["enc"], x)[:, :E + F]                                    # 潜在空間の点（平均）

# ---------- 測り方 ----------
def r2(pred, t):
    return float(1 - jnp.sum((pred - t) ** 2) / jnp.sum((t - t.mean()) ** 2))

def lin_fit_predict(Xtr, ttr, Xte):
    Xtr1 = jnp.concatenate([Xtr, jnp.ones((Xtr.shape[0], 1))], 1)
    Xte1 = jnp.concatenate([Xte, jnp.ones((Xte.shape[0], 1))], 1)
    c = jnp.linalg.lstsq(Xtr1, ttr)[0]
    return Xte1 @ c

def quad(X):                                                              # 2次までの項
    i, j = jnp.triu_indices(X.shape[1])
    return jnp.concatenate([X, X[:, i] * X[:, j]], 1)

def axis_r2(mu, y):
    """並び：軸 k だけで、要因 k を直線で説明できる割合"""
    ze = jax.nn.sigmoid(mu[:, :E])
    return [r2(lin_fit_predict(ze[:, k:k + 1], y[:, k], ze[:, k:k + 1]), y[:, k]) for k in range(E)]

def info_r2(mu, y):
    """情報量：潜在空間の8つの軸の全体（2次の項まで）で、要因 k を説明できる割合"""
    Q = quad(mu)
    return [r2(lin_fit_predict(Q, y[:, k], Q), y[:, k]) for k in range(E)]

def posthoc_r2(mu, y, n_lab):
    """後から条件づける：ラベルの付いた n_lab 件だけで、潜在空間 → 要因 k の直線（意味の方向）を求め、残りで確かめる"""
    return [r2(lin_fit_predict(mu[:n_lab], y[:n_lab, k], mu[TE]), y[TE, k]) for k in range(E)]

fmt = lambda v: "[" + ", ".join(f"{a:.2f}" for a in v) + "]"
mean = lambda v: sum(v) / len(v)
def mean_sd(v):
    m = mean(v); return m, (sum((a - m) ** 2 for a in v) / (len(v) - 1)) ** 0.5

# ---------- 実験 ----------
def exp_labels():
    print("=== 実験1：ラベルの数を変える ── 情報量（潜在空間の全体）と、並び（軸ごと） ===")
    x, y = make_data(jax.random.PRNGKey(0))
    for n_lab in [0, 20, 200]:
        mu = latent(train(x, y, n_lab), x)
        a, i = axis_r2(mu[TE], y[TE]), info_r2(mu[TE], y[TE])
        print(f"ラベル {n_lab:>3} 件：情報量 {fmt(i)} 平均 {mean(i):.3f} ／ 並び {fmt(a)} 平均 {mean(a):.3f}")
    print("→ ラベルがなくても、要因の情報は潜在空間の中にある。ラベルがないと、それが意味のある軸に並ばない")

def exp_compare(seeds=(2, 3, 4, 5, 6)):
    print("=== 実験2：同時に学習する ／ 後から条件づける（ラベル20件。乱数の種を5通り） ===")
    x, y = make_data(jax.random.PRNGKey(0))
    rows = {"教師なし（ラベル0件）の軸ごとの並び": [], "後から条件づける（教師なしの VAE ＋ 20件で意味の方向）": [],
            "同時に学習する（20件）の軸ごとの並び": [], "後から条件づける（教師なしの VAE ＋ 200件で意味の方向）": []}
    keys = list(rows)
    for s in seeds:
        mu0 = latent(train(x, y, 0, init_seed=s, noise_seed=s + 7), x)
        mu20 = latent(train(x, y, 20, init_seed=s, noise_seed=s + 7), x)
        vals = [mean(axis_r2(mu0[TE], y[TE])), mean(posthoc_r2(mu0, y, 20)),
                mean(axis_r2(mu20[TE], y[TE])), mean(posthoc_r2(mu0, y, 200))]
        for k, v in zip(keys, vals): rows[k].append(v)
        print(f"乱数の種 {s}：" + "、".join(f"{v:.3f}" for v in vals))
    for k in keys:
        m, sd = mean_sd(rows[k]); print(f"{k}：平均 {m:.3f}、標準偏差 {sd:.3f}")
    print("→ 教師なしの並びは低く、乱数の種で揺れる。要因を言い当てるだけなら、後から直線で条件づける方法が、20件でも強い")

def exp_correlated(seeds=(2, 3, 4)):
    print("=== 実験3：要因1と要因2に相関があるデータで学習し、相関のないデータで確かめる（乱数の種を3通り） ===")
    A, B = make_mixing(jax.random.PRNGKey(11))
    yi = jax.random.uniform(jax.random.PRNGKey(14), (N, 4)); xi = make_x(jax.random.PRNGKey(15), yi, A, B)
    for rho in (0.0, 0.8):
        y = correlated_y(jax.random.PRNGKey(12), N, rho); x = make_x(jax.random.PRNGKey(13), y, A, B)
        for n_lab in (20, 200):
            a2, mix = [], []
            for s in seeds:
                mui = latent(train(x, y, n_lab, init_seed=s, noise_seed=s + 7), xi)
                ze = jax.nn.sigmoid(mui[:, :E])
                a2.append(axis_r2(mui, yi)[1])                                    # 軸2が要因2を表す割合
                mix.append(r2(lin_fit_predict(ze[:, 0:1], yi[:, 1], ze[:, 0:1]), yi[:, 1]))   # 軸1が要因2を表してしまう割合
            m, sd = mean_sd(a2); mm, msd = mean_sd(mix)
            print(f"相関 {rho:.1f}、ラベル {n_lab:>3} 件：相関のないデータで、軸2 → 要因2 {m:.3f}（標準偏差 {sd:.3f}）"
                  f"、軸1 → 要因2（混ざり） {mm:.3f}（標準偏差 {msd:.3f}）")
    print("→ 要因どうしに相関があると、ラベルが少ないとき、相関した要因の軸がぶれやすい。確かめは、相関のないデータで行う")

def exp_manipulate(seeds=(2, 3, 4, 5, 6), step=0.2, n_lab=20):
    print(f"=== 実験2b：軸や方向を動かすと、ねらった要因だけが動くか（ラベル{n_lab}件。乱数の種を5通り） ===")
    x, y = make_data(jax.random.PRNGKey(0))
    # 答えを知っている読み取り役：真の要因を、データ x から読み取る（3,000件すべての答えで学習する）
    ro = mlp_init(jax.random.PRNGKey(21), [D, 64, 64, 4]); opt = optax.adam(3e-3); st = opt.init(ro)
    @jax.jit
    def rstep(p, st):
        g = jax.grad(lambda q: jnp.mean((mlp(q, x) - y) ** 2))(p); u, st = opt.update(g, st); return optax.apply_updates(p, u), st
    for _ in range(4000): ro, st = rstep(ro, st)
    print(f"読み取り役の決定係数（平均）：{mean([r2(mlp(ro, x)[:, k], y[:, k]) for k in range(E)]):.3f}")
    idx = jnp.arange(200, 700)
    def effect(dec, zin_fn):
        tgt, oth = [], []
        for k in range(E):
            d = mlp(ro, mlp(dec, zin_fn(k))) - mlp(ro, mlp(dec, zin_fn(None)))
            tgt.append(float(jnp.mean(d[:, k]))); oth.append(float(jnp.mean(jnp.abs(jnp.delete(d, k, axis=1)))))
        return mean(tgt), mean(oth)
    res = {"同時に学習する（軸を動かす）": [], "後から条件づける（意味の方向を動かす）": []}
    for s in seeds:
        pj = train(x, y, n_lab, init_seed=s, noise_seed=s + 7); mu = latent(pj, x)[idx]
        ze, zf = jnp.clip(jax.nn.sigmoid(mu[:, :E]), 0.05, 0.75), mu[:, E:]
        def zj(k):
            return jnp.concatenate([ze if k is None else ze.at[:, k].add(step), zf], 1)
        res["同時に学習する（軸を動かす）"].append(effect(pj["dec"], zj))
        p0 = train(x, y, 0, init_seed=s, noise_seed=s + 7); mu0 = latent(p0, x)
        X1 = jnp.concatenate([mu0[:n_lab], jnp.ones((n_lab, 1))], 1)
        W = jnp.linalg.lstsq(X1, y[:n_lab])[0][:-1]                           # 潜在空間 → 要因 の直線（8×4）
        m0 = mu0[idx]
        def zp(k):
            z = m0 if k is None else m0 + step * W[:, k] / jnp.sum(W[:, k] ** 2)   # 予測した要因 k が step だけ増える、最短の動き
            return jnp.concatenate([jax.nn.sigmoid(z[:, :E]), z[:, E:]], 1)
        res["後から条件づける（意味の方向を動かす）"].append(effect(p0["dec"], zp))
    for k, v in res.items():
        t = mean_sd([a for a, _ in v]); o = mean_sd([b for _, b in v])
        print(f"{k}：ねらった要因の変化 {t[0]:.3f}（標準偏差 {t[1]:.3f}）、ほかの要因の変化の大きさ {o[0]:.3f}（標準偏差 {o[1]:.3f}）")
    print(f"→ ねらった変化は {step}。ほかの要因の変化が小さいほど、狙った要因だけを動かせている")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("labels", "all"): exp_labels()
    if which in ("compare", "all"): exp_compare()
    if which in ("manipulate", "all"): exp_manipulate()
    if which in ("correlated", "all"): exp_correlated()
