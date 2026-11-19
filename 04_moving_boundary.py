"""04 毎期動く財務制限条項の線に、判定の日に届く確率（第3章・第6章）。

架空の条項（第6章の玄冬ポンプ工業の条項と同じ形）：
  各年度の決算期末の単体の純資産 ≥ max(0.75 × 直前の期末の純資産, 0.75 × 2032年3月期末の純資産)
直前の期末の純資産を状態に加えると、毎期動く線は、(直前の期末, いま) の平面の上の、
動かない折れ線になる。判定は決算期末の日にだけ行う。
比べるために、2つの誤ったやり方も計算する。
  誤り1：線を動かない線として扱い、次の期も今期の線のままだと考える
  誤り2：判定の日だけでなく、毎月の動きで、一度でも線の下に沈んだら届いたと数える
数字はすべて架空。単位は億円。
"""
import numpy as np

rng = np.random.default_rng(1)
n_sim = 20_000
base_2032 = 600.0
prev_end = 640.0      # 直前の期末（2036年3月期末）
now = 495.0           # 2037年3月期末（いまの判定で、線は 480）
months = 24           # 先の2年度ぶん（2038年3月期末と、2039年3月期末の2回の判定）
drift, vol = -0.6, 7.0   # 月ごとの純資産の動き（架空）


def line(prev):
    return np.maximum(0.75 * prev, 0.75 * base_2032)


paths = now + np.cumsum(drift + vol * rng.standard_normal((n_sim, months)), axis=1)
test_idx = [11, 23]   # 12か月後と24か月後が、決算期末の判定の日

# 正しいやり方：直前の期末の数字を状態に持ち、判定の日にだけ見る
hit_correct = np.zeros(n_sim, dtype=bool)
prev = np.full(n_sim, now)          # 2038年3月期末の判定では、直前の期末は 2037年3月期末の 495
for k in test_idx:
    na = paths[:, k]
    hit_correct |= na < line(prev)
    prev = na                       # 次の判定の線は、この期末の数字で決まる

# 誤り1：線を、いまの判定の線（480）に固定する
hit_fixed = np.zeros(n_sim, dtype=bool)
for k in test_idx:
    hit_fixed |= paths[:, k] < line(prev_end)

# 誤り2：毎月の動きで、一度でも線の下に沈んだら、届いたと数える
hit_monthly = np.zeros(n_sim, dtype=bool)
prev = np.full(n_sim, now)
for k in range(months):
    hit_monthly |= paths[:, k] < line(prev)
    if k in test_idx:
        prev = paths[:, k]


def wilson(p, n, z=1.645):
    """確率の推定の 90% の幅（ウィルソンの方法）。"""
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


for name, hit in [("正しいやり方（動く線、判定の日だけ）", hit_correct),
                  ("誤り1（線を動かない線にする）", hit_fixed),
                  ("誤り2（毎月の動きで数える）", hit_monthly)]:
    p = hit.mean(); lo, hi = wilson(p, n_sim)
    print(f"{name:24s} {p:6.1%}  （90％の幅 {lo:5.1%}〜{hi:5.1%}）")

print("\n2038年3月期末の線:", line(now), "億円（直前の期末 495 億円の 75％ と 450 億円の高い方）")
print("注意：幅は、シミュレーションの回数による幅だけである。模型の前提の不確かさは、別に示す。")
