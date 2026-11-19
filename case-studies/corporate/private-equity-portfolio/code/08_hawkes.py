"""08 ホークス過程で、投稿の連鎖の強さ（分岐の比）を推定する（第7章・第8章）。

投稿の起きやすさ λ(t) = μ(t) + Σ_{t_i < t} n β exp(−β (t − t_i))
  μ(t)：外からやってくる部分。決算の発表の日には、相手がいなくても投稿が増える
  n   ：分岐の比（1つの投稿が、平均でいくつの投稿を呼ぶか）
  β   ：呼ぶ力が弱まる速さ。半分になるまでの時間は ln2 / β（ここでは約5時間）
架空のデータを作り、2つの模型で推定する。
  模型1：μ を一定とみなす（決算の日の山を入れない）
  模型2：決算の日の山を、μ に先に入れておく
模型1は、決算の日の山まで「前の投稿が呼んだ投稿」と数えて、分岐の比を大きく見積もる。
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(4)
T = 24 * 120                      # 120日（時間の単位）
earn = np.array([20, 50, 80, 110]) * 24.0 + 17.0   # 4回の決算の発表（取引が終わった夕方）
true_mu0, true_amp, width = 0.05, 1.6, 6.0
true_n, true_beta = 0.55, np.log(2) / 5.0


def bump(t):
    return sum(np.exp(-0.5 * ((t - e) / width) ** 2) for e in earn)


def mu(t, mu0, amp):
    return mu0 + amp * bump(t)


def simulate():
    """オガタの間引きの方法で、架空の投稿の時刻を作る。"""
    t, ev = 0.0, []
    lam_max_base = true_mu0 + true_amp * 1.0 + 5.0
    while t < T:
        exc = sum(true_n * true_beta * np.exp(-true_beta * (t - s)) for s in ev[-200:])
        lam_bar = lam_max_base + exc
        t += rng.exponential(1 / lam_bar)
        if t >= T:
            break
        exc = sum(true_n * true_beta * np.exp(-true_beta * (t - s)) for s in ev[-200:])
        if rng.random() * lam_bar <= mu(t, true_mu0, true_amp) + exc:
            ev.append(t)
    return np.array(ev)


ev = simulate()
grid = np.linspace(0, T, 200_001)
bump_grid_int = np.trapezoid(bump(grid), grid)


def negloglik(params, use_bump):
    mu0, n, beta = np.exp(params[0]), 1 / (1 + np.exp(-params[1])), np.exp(params[2])
    amp = np.exp(params[3]) if use_bump else 0.0
    # 再帰の式で、各投稿の時刻の自己励起の部分を計算する
    A = np.zeros(len(ev))
    for i in range(1, len(ev)):
        A[i] = np.exp(-beta * (ev[i] - ev[i - 1])) * (1 + A[i - 1])
    lam = mu(ev, mu0, amp) + n * beta * A
    comp = mu0 * T + amp * bump_grid_int + n * np.sum(1 - np.exp(-beta * (T - ev)))
    return -(np.sum(np.log(lam)) - comp)


def fit(use_bump):
    x0 = np.array([np.log(0.1), 0.0, np.log(0.2), np.log(1.0)])
    r = minimize(negloglik, x0, args=(use_bump,), method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    p = r.x
    return 1 / (1 + np.exp(-p[1])), np.exp(p[2])


print(f"投稿の数: {len(ev)}   本当の分岐の比: {true_n}   本当の半減の時間: 5.0 時間")
for name, ub in [("模型1：決算の日の山を入れない", False), ("模型2：決算の日の山を先に入れる", True)]:
    n_hat, b_hat = fit(ub)
    print(f"{name:20s} 分岐の比 {n_hat:.2f}   半減の時間 {np.log(2)/b_hat:5.1f} 時間")
print("注意：分岐の比が 1 より小さければ、連鎖はいずれ消える。1 に近いほど長く続く。"
      "幅を示すには、データを作り直して推定し直す（ブートストラップ）。")
