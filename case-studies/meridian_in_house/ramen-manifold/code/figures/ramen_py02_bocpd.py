"""第４章：いつから変わったか。ベイズ流のオンライン変化点検出（Adams & MacKay, 2007）。

週ごとの「チャーシューの面積 ÷ 丼の縁の面積」の平均（はじめの週を 1 とした指数）に、
2038 年 2 月 15 日の週から、5% の減少を入れた、架空のデータを使う。

モデル：
- 各区間の中では、観測 = 区間の平均 + 正規分布の雑音（分散は既知）
- 区間の平均には、正規分布の事前分布（共役）
- 変化点の起こりやすさ（ハザード）は一定 1/H

最後の週での「直前の変化点から何週たったか（ラン長）」の事後分布から、
変化点が、どの週にあったかの確率を読む。
"""
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import jpfont

rng = np.random.default_rng(20380215)


def make_data():
    start = dt.date(2037, 10, 5)  # 月曜日
    weeks = [start + dt.timedelta(weeks=k) for k in range(39)]
    change = dt.date(2038, 2, 15)
    mu = np.array([1.0 if w < change else 0.95 for w in weeks])
    sd = 0.018
    y = mu + rng.normal(0, sd, len(weeks))
    return weeks, y, sd, change


def bocpd(y, sd, hazard=1 / 52, mu0=1.0, sd0=0.1):
    T = len(y)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    # ラン長ごとの、平均の事後分布（正規）のパラメータ
    m = np.array([mu0]); v = np.array([sd0 ** 2])
    for t, x in enumerate(y):
        pred = norm.pdf(x, m, np.sqrt(v + sd ** 2))
        growth = R[t, : t + 1] * pred * (1 - hazard)
        cp = np.sum(R[t, : t + 1] * pred * hazard)
        R[t + 1, 1 : t + 2] = growth
        R[t + 1, 0] = cp
        R[t + 1] /= R[t + 1].sum()
        # 共役な更新
        v_new = 1 / (1 / v + 1 / sd ** 2)
        m_new = v_new * (m / v + x / sd ** 2)
        m = np.concatenate([[mu0], m_new]); v = np.concatenate([[sd0 ** 2], v_new])
    return R


def main():
    jpfont.setup()
    weeks, y, sd, change = make_data()
    R = bocpd(y, sd)
    T = len(y)
    post = R[T]  # 最後の週のラン長の分布
    # ラン長 r → 新しい区間の最初の週 = T - r 番目
    loc = np.zeros(T)
    for r in range(1, T + 1):
        loc[T - r] += post[r]
    k_true = weeks.index(change)
    k_map = int(np.argmax(loc[1:]) + 1)
    p_pm1 = loc[k_true - 1 : k_true + 2].sum()
    print("いちばん確率の高い週:", weeks[k_map], f"（確率 {loc[k_map]:.2f}）")
    print(f"本当の変化点 {change} の週と、その前後１週の確率の合計: {p_pm1:.3f}")
    early = sum(loc[k] for k in range(1, T) if weeks[k] < dt.date(2037, 11, 1))
    print(f"2037 年 10 月の週に変化点がある確率の合計: {early:.3f}")
    assert weeks[k_map] == change or abs(k_map - k_true) <= 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    axes[0].plot(weeks, y, "o-", color="#2b7bb9", ms=4)
    axes[0].axvline(change, color="#d9822b", ls="--", label="データを作るときに入れた変化（2038年2月15日の週）")
    axes[0].set_ylabel("面積の比（指数）")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].set_title("週ごとの、チャーシューの面積の比（架空のデータ）")
    axes[1].bar(weeks[1:], loc[1:], width=5, color="#d9822b")
    axes[1].set_ylabel("変化点がこの週にある確率")
    axes[1].set_title("最後の週から振り返った、変化点の位置の事後確率")
    fig.autofmt_xdate()
    jpfont.save(fig, "ramen_py02_bocpd.jpg")


if __name__ == "__main__":
    main()
