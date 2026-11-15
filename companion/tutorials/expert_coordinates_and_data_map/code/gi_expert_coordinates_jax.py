"""
【GI理論 番外編】専門家の座標と、データの地図をつなぐ ── ２つの方法を、架空のデータで確かめる（JAX）

使い方：
  python gi_expert_coordinates_jax.py method2   # 方法２：先に地図（VAE）を作り、後から専門家の座標をつなぐ
  python gi_expert_coordinates_jax.py method1   # 方法１：最初から、専門家の座標を潜在空間の一部として学習する
  python gi_expert_coordinates_jax.py geodesic  # 方法２の地図で、専門家の座標の上に測地線を引き、直線と比べる
  python gi_expert_coordinates_jax.py noise     # 専門家の評価のばらつきが、計量と測地線をどれだけ揺らすか
  python gi_expert_coordinates_jax.py adversary # 方法１の漏れへの手当て（z_自由 から座標を当てようとする役と競わせる）
  python gi_expert_coordinates_jax.py diagnose  # つぶれかけた方向は、モデルの失敗か、データの性質か
数値はすべて架空のもの。考え方を確かめるための、小さな教材である。

架空のデータ：専門家の座標 y（4つ。0〜1）と、専門家の座標では表せない隠れた要因 h（2つ）から、
30個の指標 x を、決まった非線形の写像で作る。専門家の座標が付いているのは、3,000件のうち200件だけ。
"""
import sys
import jax, jax.numpy as jnp
import optax

KEY = jax.random.PRNGKey(0)
N, D, N_LAB = 3000, 30, 200

def make_data(key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    y = jax.random.uniform(k1, (N, 4))                         # 専門家の座標（相補性、相互作用の強さ、ハイブリッド化、両立性）
    h = jax.random.normal(k2, (N, 2))                          # 専門家の座標では表せない要因
    A = jax.random.normal(k3, (6, 40)) * 0.8; B = jax.random.normal(k4, (40, D)) * 0.3
    x = jnp.tanh(jnp.concatenate([y * 2 - 1, h * 0.5], 1) @ A) @ B
    x = x + 0.02 * jax.random.normal(k5, x.shape)
    return x, y

def mlp_init(key, sizes):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * (1.0 / jnp.sqrt(a)), jnp.zeros(b)))
    return ps

def mlp(ps, x):
    for W, c in ps[:-1]: x = jnp.tanh(x @ W + c)               # tanh：何回でも微分できる
    W, c = ps[-1]; return x @ W + c

def fit(loss, params, steps, lr=3e-3):
    opt = optax.adam(lr); st = opt.init(params)
    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss)(p); u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
    for _ in range(steps): params, st, l = step(params, st)
    return params, float(l)

def min_sv(f, y):
    return float(jnp.linalg.svd(jax.jacfwd(f)(y), compute_uv=False).min())

def r2(pred, true):
    return float(1 - jnp.sum((pred - true) ** 2) / jnp.sum((true - true.mean(0)) ** 2))

def method2():
    x, y = make_data(KEY); xl, yl = x[:N_LAB], y[:N_LAB]
    LAT = 8
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    vae = {"enc": mlp_init(k1, [D, 64, 2 * LAT]), "dec": mlp_init(k2, [LAT, 64, D])}
    def vae_loss(p, key=jax.random.PRNGKey(7)):
        out = mlp(p["enc"], x); mu, lv = out[:, :LAT], out[:, LAT:]
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        rec = jnp.mean(jnp.sum((mlp(p["dec"], z) - x) ** 2, 1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), 1))
        return rec + 0.01 * kl
    vae, l = fit(vae_loss, vae, 3000)                           # 手順１：データだけで地図（VAE）を作る
    enc_mu = lambda xx: mlp(vae["enc"], xx)[:, :LAT]
    zl = enc_mu(xl)                                             # 手順２：座標の付いた事例を、潜在空間に写す
    g = mlp_init(k3, [4, 64, LAT])                              # 手順３：専門家の座標 → 潜在空間 の写像
    g, lg = fit(lambda p: jnp.mean(jnp.sum((mlp(p, yl) - zl) ** 2, 1)), g, 3000)
    path = lambda yy: mlp(vae["dec"], mlp(g, yy))               # 手順４：専門家の座標 → 潜在空間 → データ
    print(f"手順１ 地図（VAE）の学習の損失：{l:.3f}")
    print(f"手順３ 専門家の座標 → 潜在空間 の写像の誤差（座標の付いた200件）：{lg:.4f}")
    yt, xt = y[N_LAB:N_LAB + 500], x[N_LAB:N_LAB + 500]
    full = jnp.mean(jnp.sum((mlp(vae['dec'], enc_mu(xt)) - xt) ** 2, 1))
    only = jnp.mean(jnp.sum((jax.vmap(path)(yt) - xt) ** 2, 1))
    print(f"確認３ 再構成の誤差：地図そのもの {float(full):.3f}、専門家の座標だけから {float(only):.3f}（差が、専門家の座標で表せない変動）")
    yc = jnp.array([0.5, 0.5, 0.5, 0.5])
    G = jax.jacfwd(path)(yc); metric = G.T @ G                  # 手順５：専門家の座標の上の、引き戻し計量
    print(f"手順５ 中央の点 (0.5, 0.5, 0.5, 0.5) での引き戻し計量の固有値：{jnp.round(jnp.linalg.eigvalsh(metric), 4)}")
    print(f"確認２ 中央の点でのヤコビ行列の最小特異値：{min_sv(path, yc):.4f}")
    # 確認：写像の行き先が、実際のデータのある領域から外れていないか（座標の範囲の外では外挿になる）
    zc = jnp.mean(enc_mu(x), 0); sd = jnp.std(enc_mu(x), 0)
    for yy in [yc, jnp.array([1.4, 1.4, 1.4, 1.4])]:
        dist = float(jnp.sqrt(jnp.mean(((mlp(g, yy) - zc) / sd) ** 2)))
        print(f"確認 座標 {jnp.round(yy, 1)} の行き先の、データの中心からの距離（標準偏差の何倍か）：{dist:.2f}")

def method1():
    for gam in [5.0, 50.0]:
        print(f"--- 漏れを防ぐ罰則の重み = {gam}")
        _method1(gam)

def _method1(GAM):
    x, y = make_data(KEY); xl, yl = x[:N_LAB], y[:N_LAB]
    E, F = 4, 4                                                 # z_専門家（4）と z_自由（4）
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    p = {"enc": mlp_init(k1, [D, 64, 2 * (E + F)]), "dec": mlp_init(k2, [E + F, 64, D])}
    def split(out):
        mu, lv = out[:, :E + F], out[:, E + F:]; return mu, lv
    def dec(pp, ze, zf): return mlp(pp["dec"], jnp.concatenate([ze, zf], -1))
    def loss(pp, key=jax.random.PRNGKey(9), lam=20.0, gam=GAM):
        mu, lv = split(mlp(pp["enc"], x))
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        ze, zf = jax.nn.sigmoid(z[:, :E]), z[:, E:]            # z_専門家 は 0〜1（単位立方体）に収める
        rec = jnp.mean(jnp.sum((dec(pp, ze, zf) - x) ** 2, 1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), 1))
        sup = jnp.mean(jnp.sum((jax.nn.sigmoid(mu[:N_LAB, :E]) - yl) ** 2, 1))    # 座標の付いた事例：差の２乗
        zfc = zf - zf.mean(0); zec = ze - ze.mean(0)
        leak = jnp.sum((zfc.T @ zec / N) ** 2)                  # z_自由 と z_専門家 の相関を小さくする（漏れを防ぐ簡単な代用）
        return rec + 0.01 * kl + lam * sup + gam * leak
    p, l = fit(loss, p, 4000)
    mu, _ = split(mlp(p["enc"], x))
    ze, zf = jax.nn.sigmoid(mu[:, :E]), mu[:, E:]
    te = slice(N_LAB, N)
    print(f"学習の損失：{l:.3f}")
    print(f"z_専門家 が、座標の付いていない2,800件の専門家の座標をどれだけ言い当てるか（決定係数）：{r2(ze[te], y[te]):.3f}")
    Xf = jnp.concatenate([zf[te], jnp.ones((N - N_LAB, 1))], 1)
    coef = jnp.linalg.lstsq(Xf, y[te])[0]
    print(f"漏れの確認：z_自由 だけから専門家の座標を予測したときの決定係数：{r2(Xf @ coef, y[te]):.3f}（0 に近いほど、漏れが少ない）")
    # 専門家の座標の上の定規は、z_自由 の値ごとに変わる
    def path(ze1, zf1): return dec(p, ze1[None], zf1[None])[0]
    yc = jnp.array([0.5, 0.5, 0.5, 0.5])
    for name, zf1 in [("z_自由 = 事例Aの値", zf[N_LAB]), ("z_自由 = 事例Bの値", zf[N_LAB + 1])]:
        G = jax.jacfwd(lambda t: path(t, zf1))(yc)
        print(f"{name}：中央の点での引き戻し計量の固有値 {jnp.round(jnp.linalg.eigvalsh(G.T @ G), 3)}")
    # 立方体の角：シグモイドの傾きが小さくなり、専門家の座標を動かしてもデータが動きにくい
    # z_専門家 を、シグモイドを通す前の値 u で動かすときの、データの動き（u の 1 単位あたり）
    f_u = lambda u: path(jax.nn.sigmoid(u), zf[N_LAB])
    for label, yy in [("中央 (0.5)", 0.5), ("角の近く (0.02)", 0.02)]:
        u = jnp.full(4, jnp.log(yy / (1 - yy)))
        print(f"立方体の{label}：ヤコビ行列の最小特異値 {min_sv(f_u, u):.5f}")
    print("→ 角の近くでは、最小特異値が小さくなり、引き戻し計量がつぶれかける")

if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] in ("method2", "method1")):
    {"method2": method2, "method1": method1}[sys.argv[1] if len(sys.argv) > 1 else "method2"]()

# ============================================================
# 追加の実験（測地線、評価のばらつき、漏れへの手当て、つぶれかけた方向の診断）
# ============================================================
def _train_method2(y_labels=None, seed_g=3):
    x, y = make_data(KEY); xl, yl = x[:N_LAB], (y[:N_LAB] if y_labels is None else y_labels)
    LAT = 8
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    vae = {"enc": mlp_init(k1, [D, 64, 2 * LAT]), "dec": mlp_init(k2, [LAT, 64, D])}
    def vae_loss(p, key=jax.random.PRNGKey(7)):
        out = mlp(p["enc"], x); mu, lv = out[:, :LAT], out[:, LAT:]
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        return jnp.mean(jnp.sum((mlp(p["dec"], z) - x) ** 2, 1)) + 0.01 * jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), 1))
    vae, _ = fit(vae_loss, vae, 3000)
    zl = mlp(vae["enc"], xl)[:, :LAT]
    g = mlp_init(jax.random.PRNGKey(seed_g), [4, 64, LAT])
    g, _ = fit(lambda p: jnp.mean(jnp.sum((mlp(p, yl) - zl) ** 2, 1)), g, 3000)
    return lambda yy: mlp(vae["dec"], mlp(g, yy))

def _geodesic(path, a, b, n=16, steps=1500):
    t = jnp.linspace(0, 1, n + 1)[1:-1, None]
    inner = a + t * (b - a)                                    # 直線から出発する
    def energy(pts):
        P = jnp.vstack([a, pts, b]); X = jax.vmap(path)(P)
        return jnp.sum(jnp.sum((X[1:] - X[:-1]) ** 2, 1)) * (n)  # データの空間での区分ごとの長さの２乗の和
    inner, _ = fit(energy, inner, steps, lr=1e-2)
    def length(pts):
        P = jnp.vstack([a, pts, b]); X = jax.vmap(path)(P)
        return float(jnp.sum(jnp.linalg.norm(X[1:] - X[:-1], axis=1)))
    straight = a + t * (b - a)
    return length(straight), length(inner), jnp.vstack([a, inner, b])

def geodesic():
    path = _train_method2()
    a = jnp.array([0.2, 0.3, 0.7, 0.5]); b = jnp.array([0.8, 0.6, 0.3, 0.5])
    Ls, Lg, P = _geodesic(path, a, b)
    print(f"出発点 {a} → 到着点 {b}（専門家の座標）")
    print(f"専門家の座標の上の直線の道の長さ（データの空間で測る）：{Ls:.3f}")
    print(f"測地線（最も無理のない道筋）の長さ                  ：{Lg:.3f}（直線より {100*(1-Lg/Ls):.1f}% 短い）")
    mid = P[len(P) // 2]
    print(f"道の真ん中の点：直線なら {jnp.round((a + b) / 2, 3)}、測地線では {jnp.round(mid, 3)}")
    print("→ 専門家の座標の上でも、無理のない道筋は直線とは限らない。どの座標を、どの順に動かすかが変わる")

def noise():
    x, y = make_data(KEY)
    a = jnp.array([0.2, 0.3, 0.7, 0.5]); b = jnp.array([0.8, 0.6, 0.3, 0.5]); yc = jnp.array([0.5, 0.5, 0.5, 0.5])
    res = []
    for s in range(5):                                         # 専門家ごとに、評価が標準偏差 0.1 ずつばらつくと仮定
        yl = jnp.clip(y[:N_LAB] + 0.1 * jax.random.normal(jax.random.PRNGKey(100 + s), (N_LAB, 4)), 0, 1)
        path = _train_method2(yl, seed_g=3 + s)
        G = jax.jacfwd(path)(yc); ev = jnp.linalg.eigvalsh(G.T @ G)
        _, Lg, _ = _geodesic(path, a, b, steps=800)
        res.append((float(ev[0]), float(ev[-1]), Lg))
        print(f"評価のばらつきの組 {s+1}：計量の最小の固有値 {float(ev[0]):7.1f}、最大の固有値 {float(ev[-1]):7.1f}、測地線の長さ {Lg:.3f}")
    import statistics as st
    L = [r[2] for r in res]
    print(f"測地線の長さの平均 {st.mean(L):.3f}、標準偏差 {st.pstdev(L):.3f}（平均の {100*st.pstdev(L)/st.mean(L):.1f}%）")
    print("→ 専門家の評価のばらつきが、地図の上の結果をどれだけ揺らすかを、幅として報告できる")

def _method1_adv(use_adv):
    x, y = make_data(KEY); yl = y[:N_LAB]; E, F = 4, 4
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    p = {"enc": mlp_init(k1, [D, 64, 2 * (E + F)]), "dec": mlp_init(k2, [E + F, 64, D])}
    adv = mlp_init(k3, [F, 32, 4])                              # z_自由 から専門家の座標を当てようとする役
    def parts(pp, key=jax.random.PRNGKey(9)):
        out = mlp(pp["enc"], x); mu, lv = out[:, :E + F], out[:, E + F:]
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        ze, zf = jax.nn.sigmoid(z[:, :E]), z[:, E:]
        rec = jnp.mean(jnp.sum((mlp(pp["dec"], jnp.concatenate([ze, zf], 1)) - x) ** 2, 1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), 1))
        sup = jnp.mean(jnp.sum((jax.nn.sigmoid(mu[:N_LAB, :E]) - yl) ** 2, 1))
        return rec, kl, sup, mu[:, E:]
    opt_p = optax.adam(3e-3); opt_a = optax.adam(3e-3); sp = opt_p.init(p); sa = opt_a.init(adv)
    @jax.jit
    def step(p, adv, sp, sa):
        def lp(pp):
            rec, kl, sup, zf = parts(pp)
            l = rec + 0.01 * kl + 20.0 * sup
            if use_adv:
                a_loss = jnp.mean(jnp.sum((mlp(adv, zf[:N_LAB]) - yl) ** 2, 1))
                l = l - 2.0 * a_loss                            # 当てられないように、z_自由 を作る
            return l
        g = jax.grad(lp)(p); u, sp = opt_p.update(g, sp); p = optax.apply_updates(p, u)
        zf = jax.lax.stop_gradient(parts(p)[3])
        ga = jax.grad(lambda aa: jnp.mean(jnp.sum((mlp(aa, zf[:N_LAB]) - yl) ** 2, 1)))(adv)
        u, sa = opt_a.update(ga, sa); adv = optax.apply_updates(adv, u)
        return p, adv, sp, sa
    for _ in range(4000): p, adv, sp, sa = step(p, adv, sp, sa)
    out = mlp(p["enc"], x); ze = jax.nn.sigmoid(out[:, :E]); zf = out[:, E:E + F]
    te = slice(N_LAB, N)
    Gm = jax.jacfwd(lambda t: mlp(p["dec"], jnp.concatenate([t, zf[N_LAB]])))(jnp.full(E, 0.5))
    ev = jnp.linalg.eigvalsh(Gm.T @ Gm)
    print(f"  中央の点での、専門家の座標の上の計量の固有値：{jnp.round(ev, 3)}")
    # 漏れの確認：学習に使っていない、新しい予測役（小さなニューラルネット）で z_自由 から専門家の座標を当てる
    probe = mlp_init(jax.random.PRNGKey(77), [F, 32, 4])
    probe, _ = fit(lambda q: jnp.mean(jnp.sum((mlp(q, zf[N_LAB:2000]) - y[N_LAB:2000]) ** 2, 1)), probe, 2000)
    return r2(ze[te], y[te]), r2(mlp(probe, zf[2000:]), y[2000:])

def adversary():
    for use_adv in [False, True]:
        acc, leak = _method1_adv(use_adv)
        name = "当てようとする役と競わせる" if use_adv else "競わせない（罰則なし）"
        print(f"{name}：z_専門家 の言い当て 決定係数 {acc:.3f}、z_自由 からの漏れ（新しい予測役で測る） 決定係数 {leak:.3f}")
    print("→ 当てようとする役と競わせると、漏れは減る。ただし、言い当ての精度とのつり合いを見て、重みを決める")

def diagnose():
    # 真のデータの作り方（教材なので分かっている）で、専門家の座標を動かしたときのデータの動きを調べる
    k1, k2, k3, k4, k5 = jax.random.split(KEY, 5)
    A = jax.random.normal(k3, (6, 40)) * 0.8; B = jax.random.normal(k4, (40, D)) * 0.3
    def true_gen(yy, hh=jnp.zeros(2)): return jnp.tanh(jnp.concatenate([yy * 2 - 1, hh * 0.5]) @ A) @ B
    yc = jnp.array([0.5, 0.5, 0.5, 0.5])
    J = jax.jacfwd(true_gen)(yc); ev, V = jnp.linalg.eigh(J.T @ J)
    print(f"真のデータの作り方での、中央の点の計量の固有値：{jnp.round(ev, 2)}")
    print(f"最も小さい固有値の方向（専門家の座標の４つの成分）：{jnp.round(V[:, 0], 2)}")
    if float(ev[0]) < 0.01 * float(ev[-1]):
        print("→ 真の作り方でも、この方向のデータの動きは小さい。つぶれかけはデータそのものの性質であり、その方向の結果は報告しない（または幅を広く取る）")
    else:
        print("→ 真の作り方では、どの方向にもデータはよく動く。方法１で見えたつぶれかけは、データの性質ではなく、学習が生んだもの")
        print("   （専門家の座標の情報が z_自由 に漏れると、デコーダは z_専門家 のその方向を使わなくなり、計量がつぶれかける）")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("geodesic", "noise", "adversary", "diagnose"):
    {"geodesic": geodesic, "noise": noise, "adversary": adversary, "diagnose": diagnose}[sys.argv[1]]()
