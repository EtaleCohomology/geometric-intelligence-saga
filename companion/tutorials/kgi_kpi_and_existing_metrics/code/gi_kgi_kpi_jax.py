"""
Geometric Intelligence論への招待 基礎編 第5回
既存の指標（VaR、期待ショートフォール、KRI）と、地図から得た経営インテリジェンス（崖までの近さ）を、架空のデータで並べて読む（JAX）

使い方：
  python gi_kgi_kpi_jax.py compare   # 実験1：VaR は落ち着いているのに、崖に近づいている／VaR は跳ねるのに、谷の底で安定している
  python gi_kgi_kpi_jax.py signals   # 実験2：「早く気づける」の裏づけ（当たり・見逃し・空振り・先行期間・ノイズ対シグナル比）
  python gi_kgi_kpi_jax.py onepage   # 実験3：架空の会社「エタール重工株式会社」の、経営会議の1枚
数値はすべて架空のもの。考え方を確かめるための、小さな教材である。

架空の設定：会社の状態は、2次元の地図の上の点 z で表される。地形の高さ h(z) は、谷（安定）と、
曲率が大きく負になる「崖」の領域を持つ。日々の損益は、その日の外部のショックと、地形の傾きで決まる。
"""
import sys
import jax, jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---- 架空の地形（地図）──────────────────────────────
def height(z):
    x, y = z[0], z[1]
    valley = 0.5 * (x ** 2 + y ** 2)                       # 原点のまわりの谷
    cliff = -1.6 * jnp.exp(-((x - 2.2) ** 2 + y ** 2) / 0.5)   # x = 2.2 付近に、地形が急に落ち込む場所
    return valley + cliff

def gauss_curvature(z):
    g = jax.grad(height)(z); H = jax.hessian(height)(z)
    return (H[0, 0] * H[1, 1] - H[0, 1] ** 2) / (1 + g @ g) ** 2

CLIFF_K = -0.5                                             # 曲率がこれより小さい場所を「崖」とみなす

def cliff_points(h_fn=height, n=161):
    xs = jnp.linspace(-1.0, 3.5, n); ys = jnp.linspace(-1.5, 1.5, n)
    X, Y = jnp.meshgrid(xs, ys); Z = jnp.stack([X.ravel(), Y.ravel()], 1)
    def K(z):
        g = jax.grad(h_fn)(z); H = jax.hessian(h_fn)(z)
        return (H[0, 0] * H[1, 1] - H[0, 1] ** 2) / (1 + g @ g) ** 2
    k = jax.vmap(K)(Z)
    return Z[k < CLIFF_K]

def distance_to_cliff(z, pts):
    return float(jnp.min(jnp.linalg.norm(pts - z, axis=1)))

# ---- 架空の損益 ──────────────────────────────────
def simulate(path, vol, seed):
    """path: 各日の状態 z_t（地図の上の位置）、vol: 各日の外部ショックの大きさ。損益（億円）を返す"""
    rng = np.random.default_rng(seed)
    T = len(path)
    slope = np.array([float(jnp.linalg.norm(jax.grad(height)(jnp.array(p)))) for p in path])
    return 0.3 + rng.standard_normal(T) * vol * (1.0 + 0.2 * slope)

def hist_var_es(pnl, q=0.99):
    loss = -np.asarray(pnl)
    var = np.quantile(loss, q); es = loss[loss >= var].mean()
    return var, es

# ---- 地形を、ノイズのある観測から推定する（幅を出すため）──────────────
def fitted_height(seed, noise=0.25):
    """地形の高さを、ノイズのある観測点から小さなモデルで推定し直す（乱数の種ごとに少しずつ違う地図になる）"""
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-1.0, -1.5], [3.5, 1.5], size=(80, 2))
    obs = np.array([float(height(jnp.array(p))) for p in pts]) + noise * rng.standard_normal(80)
    # 推定するのは、谷の強さ a、落ち込みの深さ b、落ち込みの中心 c（形はわかっているとする簡単な例）
    def model(theta, z):
        a, b, c = theta
        return 0.5 * a * (z[0] ** 2 + z[1] ** 2) - b * jnp.exp(-((z[0] - c) ** 2 + z[1] ** 2) / 0.5)
    def loss(theta):
        pred = jax.vmap(lambda z: model(theta, z))(jnp.array(pts))
        return jnp.mean((pred - obs) ** 2)
    theta = jnp.array([1.0, 1.0, 2.0]); g = jax.jit(jax.grad(loss))
    for _ in range(400):
        theta = theta - 0.1 * g(theta)
    return lambda z: model(theta, z)

# =====================================================
def compare():
    T = 500
    # 局面A：外部のショックは小さいまま、会社の状態が少しずつ崖のほうへ動いていく
    pathA = np.stack([np.linspace(0.0, 0.85, T), np.zeros(T)], 1)
    pnlA = simulate(pathA, vol=1.0, seed=1)
    # 局面B：一時的な外部ショックで損益は大きく揺れるが、会社の状態は谷の底にとどまる
    pathB = np.zeros((T, 2)); volB = np.where((np.arange(T) > 380) & (np.arange(T) < 430), 3.0, 1.0)
    pnlB = simulate(pathB, vol=volB, seed=2)
    pts = cliff_points()
    print("            | 前半250日の VaR（99%） | 後半250日の VaR（99%） | 期初の崖までの近さ | 期末の崖までの近さ | 期末の曲率")
    for name, path, pnl in [("局面A", pathA, pnlA), ("局面B", pathB, pnlB)]:
        v1, _ = hist_var_es(pnl[:250]); v2, _ = hist_var_es(pnl[250:])
        d0 = distance_to_cliff(jnp.array(path[0]), pts); d1 = distance_to_cliff(jnp.array(path[-1]), pts)
        k1 = float(gauss_curvature(jnp.array(path[-1])))
        print(f"{name}      | {v1:8.2f} 億円 | {v2:8.2f} 億円 | {d0:6.2f} | {d1:6.2f} | {k1:+.3f}")
    # 地図そのものを推定し直したときの幅（乱数の種を変えて20回）
    ds = []
    for s in range(20):
        h_hat = fitted_height(100 + s); p_hat = cliff_points(h_hat, n=121)
        ds.append(distance_to_cliff(jnp.array(pathA[-1]), p_hat))
    ds = np.array(ds)
    print(f"局面Aの期末の崖までの近さ（地図を20回推定し直した幅）：中央値 {np.median(ds):.2f}、10〜90% の範囲 {np.quantile(ds,0.1):.2f}〜{np.quantile(ds,0.9):.2f}")

def signals():
    """多数の架空の期間を作り、出来事（状態が崖に入る）の前に、各指標がシグナルを出したかを数える"""
    rng = np.random.default_rng(7)
    pts = cliff_points(n=121)
    N, T, W = 120, 60, 12      # 期間の数、各期間の長さ（月）、事前の窓（12か月）
    rows = []
    for i in range(N):
        crisis = rng.random() < 0.35
        start = rng.uniform(0.0, 0.4)
        end = rng.uniform(1.75, 2.0) if crisis else rng.uniform(0.2, 0.8)
        t_cross = int(rng.integers(36, 58)) if crisis else None
        xs = np.linspace(start, end, T if not crisis else t_cross)
        if crisis: xs = np.concatenate([xs, np.full(T - t_cross, end)])
        path = np.stack([xs + 0.03 * rng.standard_normal(T), 0.05 * rng.standard_normal(T)], 1)
        vol = 1.0 + (rng.random(T) < 0.08) * rng.uniform(1.0, 2.0, T)   # 出来事と関係のない、一時的なショック
        dist = np.array([distance_to_cliff(jnp.array(p), pts) for p in path])
        slope = np.array([float(jnp.linalg.norm(jax.grad(height)(jnp.array(p)))) for p in path])
        kri = vol * (1.0 + 0.2 * slope) + 0.15 * rng.standard_normal(T)   # 既存の KRI（損益の揺れの大きさ。崖の直前で大きくなる）
        rows.append((crisis, t_cross, dist, kri))
    train, test = rows[:60], rows[60:]
    def evaluate(rows, sig_fn):
        hit = miss = fa = quiet = 0; leads = []
        for crisis, tc, dist, kri in rows:
            s = sig_fn(dist, kri)
            if crisis:
                win = s[max(0, tc - W):tc]
                if win.any(): hit += 1; leads.append(W - int(np.argmax(win)))
                else: miss += 1
                pre = s[:max(0, tc - W)]
                fa += int(pre.sum()); quiet += int((~pre).sum())
            else:
                fa += int(s.sum()); quiet += int((~s).sum())
        n_crisis = hit + miss
        hit_rate = hit / max(n_crisis, 1); fa_rate = fa / max(fa + quiet, 1)
        nsr = fa_rate / hit_rate if hit_rate > 0 else float("inf")
        return hit, miss, fa_rate, nsr, (np.median(leads) if leads else float("nan"))
    best = {}
    for name, grid, mk in [("崖までの近さ", np.linspace(0.2, 1.2, 21), lambda th: (lambda d, k: d < th)),
                           ("既存の KRI", np.linspace(1.2, 3.0, 19), lambda th: (lambda d, k: k > th))]:
        scores = [(evaluate(train, mk(th))[3], th) for th in grid]
        nsr, th = min(scores); best[name] = th
        h, m, far, nsr_t, lead = evaluate(test, mk(th))
        print(f"{name}（閾値 {th:.2f}、学習に使っていない60期間で評価）：当たり {h}、見逃し {m}、空振りの率 {far:.3f}、ノイズ対シグナル比 {nsr_t:.3f}、先行期間の中央値 {lead}か月")

def onepage():
    """架空の会社エタール重工の、ある四半期の『経営会議の1枚』の数値を作る"""
    rng = np.random.default_rng(11)
    # 既存の指標（架空）
    pnl = 0.3 + 1.0 * rng.standard_normal(250)
    var, es = hist_var_es(pnl, 0.99); _, es975 = hist_var_es(pnl, 0.975)
    nopat, el, ic, ec = 180.0, 9.0, 800.0, 36.0               # 防衛・産業機械事業（架空、億円）
    roic = nopat / ic; saroic = (nopat - el) / (ic + ec)
    plan, actual = 410.0, 398.0                                # 四半期の営業利益（計画と実績、架空、億円）
    # 地図から得た経営インテリジェンス：状態が少しずつ崖へ動く。今後4四半期で崖に入る確率をシミュレーションで求める
    pts = cliff_points(n=121)
    z_now = jnp.array([0.55, 0.0])
    d_now = distance_to_cliff(z_now, pts)
    def p_cross(seed):
        r = np.random.default_rng(seed); hits = 0
        for _ in range(400):
            z = np.array(z_now)
            for _q in range(4):
                z = z + np.array([0.08, 0.0]) + 0.15 * r.standard_normal(2)
                if distance_to_cliff(jnp.array(z), pts) < 0.05: hits += 1; break
        return hits / 400
    ps = np.array([p_cross(200 + s) for s in range(10)])
    p_mid, p_lo, p_hi = np.median(ps), np.quantile(ps, 0.1), np.quantile(ps, 0.9)
    def words(p):
        for lo, hi, w in [(0.01, 0.05, "ほぼ可能性なし"), (0.05, 0.20, "非常に可能性が低い"), (0.20, 0.45, "可能性が低い"),
                          (0.45, 0.55, "ほぼ五分五分"), (0.55, 0.80, "可能性が高い"), (0.80, 0.95, "非常に可能性が高い"), (0.95, 1.0, "ほぼ確実")]:
            if lo <= p < hi: return w
        return "ほぼ可能性なし" if p < 0.01 else "ほぼ確実"
    print("【既存の指標】")
    print(f"  VaR（99%、1日）：{var:.2f} 億円の損失　期待ショートフォール（97.5%、1日）：{es975:.2f} 億円の損失")
    print(f"  営業利益の予実差異：{actual - plan:+.1f} 億円（計画 {plan:.0f}、実績 {actual:.0f}）")
    print(f"  ROIC：{100*roic:.1f}%　SAROIC（本稿執筆者の提案する指標）：{100*saroic:.1f}%")
    print("【地図から得た経営インテリジェンス】")
    print(f"  崖までの近さ：{d_now:.2f}（地図の上の距離）")
    print(f"  今後4四半期で崖に入る確率：中央値 {p_mid:.2f}（10〜90% の範囲 {p_lo:.2f}〜{p_hi:.2f}）→ 言葉では「{words(p_mid)}」")

if __name__ == "__main__":
    {"compare": compare, "signals": signals, "onepage": onepage}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
