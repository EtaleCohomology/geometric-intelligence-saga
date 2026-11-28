# s07 戻る速さを固有値で読む：クープマン作用素の近似（動的モード分解）（第9章「戻る速さ」）
# 区画Ａ、Ｂ、Ｃの「足りない分」を並べた組を毎日の状態とし、x_{k+1} ≒ K x_k となる行列 K を求める。
# データは架空。区画ＡとＢは毎日 0.80 倍、区画Ｃは 0.985 倍で縮むように作り、揺れを加える。
import math
import numpy as np
from common import plt, savefig, NAVY, CYAN, AMBER

rng = np.random.default_rng(1027)
days = 26                                    # 10月１日から10月26日
true = np.diag([0.80, 0.80, 0.985])
mix = np.array([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]])  # 区画ＡとＢは少し影響し合う
K_true = mix @ true @ np.linalg.inv(mix)
x = np.zeros((3, days)); x[:, 0] = [30000, 25000, 40000]  # 足りない個数
for k in range(days - 1):
    x[:, k + 1] = K_true @ x[:, k] * (1 + rng.normal(0, [0.01, 0.01, 0.0], 3))
# 区画Ｃは、ラインの一部だけが日によって動いたり止まったりするので、記録の揺れが大きい
xC = x[2] * np.exp(rng.normal(0, 0.03, days))
x[2] = xC

X, Y = x[:, :-1], x[:, 1:]
K = Y @ np.linalg.pinv(X)                   # 最小二乗で K を求める（動的モード分解の最も簡単な形）
ev = np.sort(np.abs(np.linalg.eigvals(K)))[::-1]
print("推定した固有値:", np.round(ev, 3))

print("0.8 を３回掛けると", round(0.8 ** 3, 3), "→ およそ３日で半分")
print("0.985 で半分になるまでの日数:", round(math.log(0.5) / math.log(0.985), 1))
for lam in [0.9804, 0.9885]:
    print(f"  固有値 {lam} なら半分まで {math.log(0.5)/math.log(lam):.0f} 日")

# 揺れの幅：区画Ｃの足りない分の対数を日にちに当てはめ、傾きの 90% の幅から半分までの日数の幅を出す
yC = np.log(xC)
t = np.arange(days)
A = np.vstack([t, np.ones(days)]).T
coef, res, *_ = np.linalg.lstsq(A, yC, rcond=None)
s2 = res[0] / (days - 2)
se = math.sqrt(s2 / ((t - t.mean()) ** 2).sum())
lo, hi = math.exp(coef[0] - 1.71 * se), math.exp(coef[0] + 1.71 * se)
print(f"区画Ｃの固有値の幅（90%）: {lo:.4f}〜{hi:.4f} → 半分まで "
      f"{math.log(0.5)/math.log(lo):.0f}〜{math.log(0.5)/math.log(hi):.0f} 日")

fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (lab, c) in enumerate([("区画Ａ", CYAN), ("区画Ｂ", NAVY), ("区画Ｃ", AMBER)]):
    ax.plot(range(1, days + 1), x[i] / x[i, 0], "o-", ms=3, color=c, label=lab)
ax.axhline(0.5, color="#999", ls="--", lw=1); ax.text(22, 0.53, "半分", fontsize=9)
ax.set_xlabel("10月の日"); ax.set_ylabel("足りない分（初日を１とする）")
ax.set_title("区画ＡとＢは約３日で半分、区画Ｃは約46日で半分（架空のデータ）", fontsize=10)
ax.legend(fontsize=9)
savefig(fig, "s07_recovery.png")
