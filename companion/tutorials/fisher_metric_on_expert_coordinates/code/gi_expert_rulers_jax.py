"""
【GI理論 番外編】専門家の座標に、定規を入れる ── 4つの実験（JAX）

使い方：
  python gi_expert_rulers_jax.py toy         # 手で計算できる小さな例（座標2個 → 指標3個）
  python gi_expert_rulers_jax.py rank        # 二値の結果だけのフィッシャー計量は、階数が1
  python gi_expert_rulers_jax.py two_rulers  # 引き戻し計量とフィッシャー計量を、同じ点で比べる
  python gi_expert_rulers_jax.py sigma_test  # 安定性の式 Σ の形を、入れ子のモデルの比較で確かめる
  python gi_expert_rulers_jax.py             # すべて
数値はすべて架空のもの。考え方を確かめるための教材である。
"""
import sys
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def fmt(v, d=4):
    return "[" + ", ".join(f"{float(a):.{d}f}" for a in v) + "]"

# ---------------------------------------------------------------- toy
def toy():
    print("=== 手で計算できる小さな例：専門家の座標 (y1, y2) → 指標 (x1, x2, x3) ===")
    f = lambda y: jnp.array([y[0] + y[1], y[0] ** 2, y[0] * y[1]])
    y = jnp.array([1.0, 2.0])
    J = jax.jacfwd(f)(y)
    G = J.T @ J
    print("JAX のヤコビ行列 J（3行×2列）：", [fmt(r, 1) for r in J])
    print("JAX の定規 G = JᵀJ（2行×2列）：", [fmt(r, 1) for r in G])
    ev = jnp.linalg.eigvalsh(G)
    print(f"G の固有値：{fmt(ev, 3)}（手計算：(11 ± √85) / 2 ＝ {(11 - 85 ** 0.5) / 2:.3f}、{(11 + 85 ** 0.5) / 2:.3f}）")
    for name, d in [("y1 を 0.01 動かす", jnp.array([0.01, 0.0])), ("y2 を 0.01 動かす", jnp.array([0.0, 0.01]))]:
        by_G = float(jnp.sqrt(d @ G @ d))
        direct = float(jnp.linalg.norm(f(y + d) - f(y)))
        print(f"{name}：定規 G で測った長さ {by_G:.5f}、指標の実際の動きの長さ {direct:.5f}")

# ---------------------------------------------------------------- rank
W = jnp.array([[ 2.0, -3.0,  1.5,  2.5],
               [ 1.0,  2.0, -1.0,  0.5],
               [-0.5,  1.5,  2.0, -1.0],
               [ 0.5,  0.5,  0.5, -2.0]])

def fisher_binary_lin(th, w):
    p = lambda t: jax.nn.sigmoid(w @ (t - 0.5))
    gp = jax.grad(p)(th); pv = p(th)
    return jnp.outer(gp, gp) / (pv * (1 - pv))

def fisher_gauss_lin(th, w):
    gm = jax.grad(lambda t: w @ t)(th)
    return jnp.outer(gm, gm)

def show(name, G):
    ev = jnp.linalg.eigvalsh(G)
    rank = int(jnp.sum(ev > 1e-6 * ev.max()))
    txt = ", ".join(f"{abs(float(v)):.4f}" if abs(float(v)) > 1e-6 * float(ev.max()) else "0" for v in ev)
    print(f"{name}：固有値 [{txt}]、階数 {rank}")

def rank():
    print("=== 二値の結果だけのフィッシャー計量は、階数が1 ===")
    th = jnp.full(4, 0.5)
    G1 = fisher_binary_lin(th, W[0]); show("結果1（崩壊の有無）だけ", G1)
    G2 = G1 + fisher_gauss_lin(th, W[1]); show("結果1 ＋ 結果2（期間）", G2)
    G3 = G2 + fisher_gauss_lin(th, W[2]); show("結果1 ＋ 結果2 ＋ 結果3（規模）", G3)
    G4 = G3 + fisher_gauss_lin(th, W[3]); show("結果1〜4", G4)
    show("結果1だけを100件（同じ座標の事例）", 100 * G1)

# ---------------------------------------------------------------- two rulers
def make_world(key, k_scale=0.1):
    """専門家の座標 → 30個の指標。両立性（4つ目の座標）は、指標をほとんど動かさない（k_scale 倍）"""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (4, 24)) * 0.9
    A = A.at[3].multiply(k_scale)
    B = jax.random.normal(k2, (24, 30)) * 0.3
    return lambda th: jnp.tanh((th * 2 - 1) @ A) @ B

def fisher_outcomes(th, indicators, R, direct):
    """結果の4種類。1つ目は二値（解消の確率。ロジットが s0）、2〜4つ目は正規分布（分散1）の平均。
    s ＝ R・（指標） ＋ direct・（座標 − 0.5）。direct は、指標を通らずに、座標が結果に直接かかる重み"""
    s = lambda t: R @ indicators(t) + direct @ (t - 0.5)
    Jm = jax.jacfwd(s)(th)
    p = jax.nn.sigmoid(s(th)[0])
    G = (p * (1 - p)) * jnp.outer(Jm[0], Jm[0])
    for k in range(1, 4): G = G + jnp.outer(Jm[k], Jm[k])
    return G

def two_rulers():
    print("=== 引き戻し計量とフィッシャー計量を、同じ点で比べる ===")
    print("（架空のデータ。両立性は、30個の指標をほとんど動かさないように作った）")
    indicators = make_world(jax.random.PRNGKey(0))
    R = jax.random.normal(jax.random.PRNGKey(1), (4, 30)) * 0.5
    names = ["相補性", "相互作用の強さ", "ハイブリッド化", "両立性"]
    th = jnp.full(4, 0.5)
    Jx = jax.jacfwd(indicators)(th); Gpb = Jx.T @ Jx
    print(f"引き戻し計量（指標の動き）の、4つの軸の向きの長さの2乗：{fmt(jnp.diag(Gpb), 2)}")
    cases = {
        "場面1：結果は、指標を通じてだけ決まる": jnp.zeros((4, 4)),
        "場面2：両立性が、指標を通らずに、結果に直接かかる": jnp.zeros((4, 4)).at[:, 3].set(jnp.array([6.0, 4.0, -5.0, 3.0])),
    }
    for name, D in cases.items():
        Gf = fisher_outcomes(th, indicators, R, D)
        a = Gf / jnp.trace(Gf); b = Gpb / jnp.trace(Gpb)          # 2つの定規の大きさをそろえる
        L = jnp.linalg.cholesky(b); Li = jnp.linalg.inv(L)
        w, U = jnp.linalg.eigh(Li @ a @ Li.T)                         # 方向ごとの「フィッシャー ÷ 引き戻し」の長さの比
        v = Li.T @ U[:, -1]; v = v / jnp.linalg.norm(v)
        top = int(jnp.argmax(jnp.abs(v)))
        print(f"{name}")
        print(f"  フィッシャー計量の、4つの軸の向きの長さの2乗：{fmt(jnp.diag(Gf), 2)}")
        print(f"  結果は大きく動くのに、指標はあまり動かない向き：長さの比 {float(w[-1]):.1f}倍、いちばん大きな成分は {names[top]}（{float(v[top]):+.2f}）")
    print("→ 座標が、指標を通らずに結果にかかる向きがあると、2つの定規は、その向きで大きく食い違う。データの地図が見落としている向きの手がかりになる")

# ---------------------------------------------------------------- sigma test
def log_feats(T):
    o, I, h, K = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    return jnp.stack([jnp.log(o), jnp.log(h), jnp.log(K), jnp.log(I), K], 1)

def fit_logistic(X, y, steps=60):
    X1 = jnp.concatenate([jnp.ones((X.shape[0], 1)), X], 1)
    def nll(b):
        z = X1 @ b; return jnp.sum(jnp.logaddexp(0, z) - y * z)
    b = jnp.zeros(X1.shape[1])
    g, H = jax.grad(nll), jax.hessian(nll)
    for _ in range(steps): b = b - jnp.linalg.solve(H(b) + 1e-9 * jnp.eye(len(b)), g(b))
    return b, -float(nll(b))

def sigma_test(n=800):
    print(f"=== 安定性の式 Σ の形を、入れ子のモデルの比較で確かめる（事例 {n} 件） ===")
    T = jax.random.uniform(jax.random.PRNGKey(5), (n, 4), minval=0.05, maxval=0.95)
    F = log_feats(T)
    logS = F @ jnp.array([1.0, 1.0, 1.0, -1.0, 1.0]) - 1.0              # log Σ = log o + log h + log K − log I − (1 − K)
    scen = {
        "場面1：結果が、Σ の式どおりに生まれる": 0.5 + 1.2 * logS,
        "場面2：結果が、別の形で生まれる（ハイブリッド化が高いほど崩れやすい）": 0.5 + 1.2 * (F @ jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]) - 1.0),
    }
    for name, z in scen.items():
        y = (jax.random.uniform(jax.random.PRNGKey(6), (n,)) < jax.nn.sigmoid(z)).astype(jnp.float64)
        _, ll_small = fit_logistic(logS[:, None], y)
        b_big, ll_big = fit_logistic(F, y)
        lr = 2 * (ll_big - ll_small); p = float(jax.scipy.stats.chi2.sf(lr, 4))
        print(f"{name}")
        print(f"  Σ だけのモデルの対数尤度 {ll_small:.1f}、4つの軸を自由に使うモデルの対数尤度 {ll_big:.1f}")
        print(f"  尤度比の統計量 {lr:.2f}（自由度4）、p 値 {p:.3g}")
        print(f"  自由なモデルの重み（log o, log h, log K, log I, K）：{fmt(b_big[1:], 2)}")
    print("→ Σ の式どおりなら、自由なモデルは Σ のモデルをほとんど上回らない。式の形が違えば、はっきり上回る")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in [("toy", toy), ("rank", rank), ("two_rulers", two_rulers), ("sigma_test", sigma_test)]:
        if which in (k, "all"): f()
