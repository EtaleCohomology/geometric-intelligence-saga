# s12 グロモフ・ワッサースタイン距離で、対応のない２つの記録を間隔の模様で対応づける（第4章）
# 自動搬送車は同じ間隔の模様を繰り返す。模様が繰り返すと、１周ずらした対応も同じくらいよく合う。
# また、時刻の差の大きさだけで距離を測ると、時間の向きを逆にした対応も同じだけよく合う。
# 始め方を50通り変えて、答えがどう割れるかを数える。必要なもの：pip install pot
import numpy as np
import ot

rng = np.random.default_rng(1004)
period, reps = 25.0, 4
pattern = np.array([0, 7, 9, 11, 19], dtype=float)          # 充電の場所に戻る時刻の模様（分）
agv = np.concatenate([pattern + period * k for k in range(reps)])
net = agv + 413 + rng.normal(0, 0.15, len(agv))             # ずれた時計で、同じ出来事を記録
T = period * reps


def circ(t):            # 一日のうちで繰り返す動きなので、時刻の差を「輪の上の差」で測る
    d = np.abs(t[:, None] - t[None, :]) % T
    return np.minimum(d, T - d)


C1, C2 = circ(agv), circ(net - 413)
p = np.ones(len(agv)) / len(agv)
n = len(agv)
count = {}
for trial in range(50):
    G0 = ot.sinkhorn(p, p, rng.random((n, n)), 0.05)        # 始めの置き方をでたらめに変える
    Tm = ot.gromov.gromov_wasserstein(C1, C2, p, p, "square_loss", G0=G0)
    m = Tm.argmax(axis=1)
    step = int(round(((m[1] - m[0]) % n)))
    reverse = step != 1                                      # 順番が逆に回っている答え
    shift = int(m[0] // len(pattern)) if not reverse else None
    key = "時間の向きが逆" if reverse else f"{shift}周ずれ"
    count[key] = count.get(key, 0) + 1
for k, v in sorted(count.items(), key=lambda kv: -kv[1]):
    print(f"{k}: {v} 通り")
print("→ 時間の向きが逆の答えを除いても、何周ずれかが割れる。『決めない』と書く")
