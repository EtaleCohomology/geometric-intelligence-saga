"""第４章：差分の差分（difference-in-differences）。

巻雲と、ロンドンのほかのラーメン店 12 軒の、週ごとの「チャーシューの面積の比」（架空）。
2038 年 2 月の半ばから、ロンドンじゅうで豚肉の値段が上がり、どの店も少しずつ減った（風の分）。
巻雲だけは、それに加えて、もっと減った。

- 巻雲の前後の差 − ほかの店の前後の差 = 巻雲だけの変化
- 前提：変化の前は、巻雲とほかの店が、平行に動いていたこと（平行トレンド）
  この前提は、データで「証明」はできない。
  「平行でないという証拠は見つからなかった」と言えるだけである。
"""
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import jpfont

rng = np.random.default_rng(7)
N_OTHER = 12
start = dt.date(2037, 10, 5)
weeks = [start + dt.timedelta(weeks=k) for k in range(39)]
change = dt.date(2038, 2, 15)
post = np.array([w >= change for w in weeks], float)


def simulate():
    T = len(weeks)
    common = -0.010 * post  # ロンドンじゅうの風の分（-1%）
    rows = []
    for s in range(N_OTHER + 1):
        treated = s == 0
        level = 1.0 + rng.normal(0, 0.03)
        own = (-0.040 * post) if treated else 0.0  # 巻雲だけの分（-4%）
        y = level * (1 + common + own) + rng.normal(0, 0.012, T)
        for t in range(T):
            rows.append((s, t, treated, post[t], y[t]))
    return np.array(rows, float)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), dof


def main():
    jpfont.setup()
    D = simulate()
    shop, t, treat, pst, y = D.T
    ly = np.log(y)
    # 店ごとの定数と、週ごとの定数（二方向の固定効果）＋ 巻雲×変化のあと
    S = (shop[:, None] == np.arange(N_OTHER + 1)[None, :]).astype(float)
    W = (t[:, None] == np.arange(1, len(weeks))[None, :]).astype(float)
    X = np.column_stack([S, W, treat * pst])
    b, se, dof = ols(X, ly)
    did, did_se = b[-1], se[-1]
    print(f"差分の差分（対数）: {did:+.4f}  標準誤差 {did_se:.4f}  → およそ {100*(np.exp(did)-1):+.2f}%")

    # 平行トレンドの点検：変化の前だけで、巻雲×週の傾きを入れる
    m = pst == 0
    Xp = np.column_stack([S[m], W[m][:, : int(m.sum() / (N_OTHER + 1)) - 1], treat[m] * t[m]])
    bp, sep, dofp = ols(Xp, ly[m])
    tstat = bp[-1] / sep[-1]
    p = 2 * (1 - stats.t.cdf(abs(tstat), dofp))
    verdict = "平行でないという証拠は見つからなかった" if p > 0.05 else "平行でない疑いがある"
    print(f"変化の前の、巻雲だけの傾き: {bp[-1]:+.5f}/週  p 値 {p:.2f} → {verdict}")

    k = len(weeks)
    kumo = y[shop == 0]
    others = y[shop != 0].reshape(N_OTHER, k).mean(0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(weeks, kumo / kumo[post == 0].mean(), color="#2b7bb9", lw=2, label="巻雲")
    ax.plot(weeks, others / others[post == 0].mean(), color="#d9822b", lw=2, label="ほかの店 12 軒の平均")
    ax.axvline(change, color="gray", ls="--")
    ax.set_ylabel("変化の前の平均を 1 とした指数")
    ax.set_title(f"差分の差分：巻雲だけの変化はおよそ {100*(np.exp(did)-1):+.1f}%（架空のデータ）")
    ax.legend()
    fig.autofmt_xdate()
    jpfont.save(fig, "ramen_py03_did.jpg")


if __name__ == "__main__":
    main()
