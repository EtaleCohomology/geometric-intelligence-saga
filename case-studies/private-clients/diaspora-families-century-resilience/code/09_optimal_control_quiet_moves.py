"""09 早めに、小さく、ゆっくり舵を切る：舵の大きさに代償を書き込んだ最適制御（本編第2章・第5章）。

すべて架空の模型。一族が、資産のうち G（ここでは 0.5）を、T か月かけて分けて移すとする。
  u_t ：t 月に移す量（u_t ≥ 0、合計は G）
  x_t ：t 月までに移し終えた量（x_t = u_1 + … + u_t）
  h_t ：t 月の危うさ（月ごとに少しずつ高まる、と仮定）
損失 J = Σ_t h_t (G − x_t)  ＋  α Σ_t u_t^2
  第１項：まだ移していない資産が、危うさにさらされる量
  第２項：１か月に大きく動くほど目立つ、という代償（舵を急に大きく切る代償）。α がその重さ
1. ポントリャーギンの最大原理（離散の時刻の版）が教える、最もよい舵が必ず満たす条件
     u_s = max(0, (Λ_s − μ) / (2α))、 Λ_s = h_s + h_{s+1} + … + h_T（s 月に移すと、その後ずっと避けられる危うさ）
   を満たす舵を、μ（合計を G にそろえる数）を探して作り、数値の最適化の答えと一致することを確かめる
2. 自己実現的な予言：多くの一族が同じ合図で同じ月に大きく動くと、その動き自体が危うさを高める
     n_t = γ n_{t-1} + κ ·（t 月に全体で動いた量）^2、 h_t = h0_t + n_t
   （n_t は、大きな動きが生む話の強さ。月ごとに γ 倍に薄れる）として、「合図を待って一度に動く」舵と、「早めに少しずつ動く」舵を比べる
"""
import numpy as np
from scipy.optimize import minimize, brentq

T, Gtot, alpha = 36, 0.5, 0.4
h0 = 0.004 + 0.0003 * np.arange(T)            # 月ごとに少しずつ高まる危うさ（架空）

def J(u, h):
    x = np.cumsum(u)
    return np.sum(h * (Gtot - x)) + alpha * np.sum(u ** 2)

# ---- 1. 必ず満たす条件から作る舵と、数値の最適化の答え -----------------------
Lam = np.cumsum(h0[::-1])[::-1]
def u_of(mu):
    return np.maximum(0, (Lam - mu) / (2 * alpha))
mu = brentq(lambda m: u_of(m).sum() - Gtot, -10, 10)
u_pmp = u_of(mu)

cons = [{"type": "eq", "fun": lambda u: u.sum() - Gtot}]
res = minimize(J, np.full(T, Gtot / T), args=(h0,), bounds=[(0, None)] * T, constraints=cons,
               method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
print("1. 必ず満たす条件から作った舵と、数値の最適化の舵の差の最大：", f"{np.max(np.abs(res.x - u_pmp)):.2e}")
print("   最初の6か月に移す量：", np.round(u_pmp[:6], 4), " 舵が0になる最初の月：", (int(np.argmax(u_pmp == 0)) + 1) if np.any(u_pmp == 0) else "なし")
print("   条件は『必ず満たす』もの。この模型は損失が凸なので、条件を満たす舵が最もよい舵になるが、凸でない模型では候補を絞るだけである。")

# ---- 2. 自己実現的な予言 ---------------------------------------------------
# 大きな動きほど目立ち、話の材料になる。話は月がたつと薄れるが、すぐには消えない（架空）。
#   n_t = γ n_{t-1} + κ (全体で動いた量_t)^2、  h_t = h0_t + n_t
kappa, gamma, share = 0.6, 0.85, 1.0   # share：同じ舵で動く一族が、全体のどれだけを占めるか（架空）
signal = 24                            # 合図が出る月（架空）

def total_cost(u):
    flow = share * u
    n = np.zeros(T)
    for t in range(T):
        n[t] = (gamma * n[t - 1] if t else 0.0) + kappa * flow[t] ** 2
    h = h0 + n
    return J(u, h), h

u_sudden = np.zeros(T); u_sudden[signal] = Gtot           # 合図を待って、１か月で全部
u_ramp = np.zeros(T); u_ramp[signal:signal + 3] = Gtot / 3 # 合図のあと３か月で

def fb_obj(u):
    return total_cost(u)[0]
res2 = minimize(fb_obj, u_pmp, bounds=[(0, None)] * T, constraints=cons, method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 2000})

print(f"\n2. 多くの一族が同じ舵で動くとき（κ = {kappa}、γ = {gamma}）")
print("   舵                         損失    １か月に動く量の最大   いちばん危うい月の危うさ   舵が0でない月の数")
for name, u in [("合図を待って１か月で全部", u_sudden), ("合図のあと３か月で", u_ramp),
                ("代償を書き込んだ最もよい舵", res2.x)]:
    c, h = total_cost(u)
    print(f"   {name:24s}  {c:.4f}  {u.max():.4f}               {h.max():.4f}                 {int(np.sum(u > 1e-4))}")
print("   一度に大きく動くと、その月の危うさを、自分たちの動きが押し上げる（模型の中の自己実現的な予言）。")
print("   代償 α を書き込むと、計算は、早めに、小さく、ゆっくり動く舵を選ぶ。")
print("   注意：この模型は、資産を移す舵だけを扱う。人が、いつ、どこへ移るかは、計算で決めない（本編第2章）。")
