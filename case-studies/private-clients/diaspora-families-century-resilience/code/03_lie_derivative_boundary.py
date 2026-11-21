"""03 境界までの近さと、その近さが縮む速さ（リー微分）（本編第2章・第3章）。

境界（越えたら戻れない線）を、仮説として x = 1 の直線に置く（架空）。
境界までの近さ h(x, y) = 1 − x。地図の上の流れ f(x, y; 月) に沿ったリー微分は
  L_f h = ∇h · f = −f_x
で、「いまの流れに乗って進むと、近さが１か月あたりどれだけ縮むか」を表す。
1. 流れが、ある月から少しずつ強まる場合に、近さと、近さが縮む速さを月ごとに並べる
2. 「近さが 0.5 を切ったら」と「縮む速さが 0.01 を超えたら」の、どちらが先に気づくかを比べる
3. データに雑音があるとき、速さを直近の何か月かの近さの傾きで推定すると、窓の長さで遅れと揺れが変わることを確かめる
"""
import numpy as np

rng = np.random.default_rng(3)
months = 120
grad_h = np.array([-1.0, 0.0])          # h = 1 − x の勾配

def field(z, m):
    """m 月目の流れ（架空）。48か月目から、境界へ向かう成分が少しずつ強まる。"""
    push = 0.002 + 0.0004 * max(0, m - 48)
    return np.array([push + 0.002 * np.sin(z[1]), 0.01 * np.cos(z[0])])

z = np.array([0.05, 0.0])
rec = []
for m in range(months):
    f = field(z, m)
    rec.append((m, 1 - z[0], -(grad_h @ f)))   # 近さ h と、縮む速さ −L_f h
    z = z + f
rec = np.array(rec)
h, shrink = rec[:, 1], rec[:, 2]

first_h = int(np.argmax(h < 0.5))
first_v = int(np.argmax(shrink > 0.01))
print("月   近さ h   近さが縮む速さ（−L_f h、１か月あたり）")
for m in [0, 24, 48, 54, 60, 66, 72, 78, 84, 90, 96]:
    print(f"{m:3d}  {h[m]:.3f}   {shrink[m]:.4f}")
print(f"\n近さが 0.5 を切った月：{first_h}")
print(f"縮む速さが 0.01 を超えた月：{first_v}（{first_h - first_v}か月早い）")
print("どちらの基準も仮の値である。基準の置き方は、06 の『見逃しと空振り』で扱う。")

# ---- 3. 雑音のある観測から、速さを推定する -----------------------------------
obs = h + rng.normal(0, 0.01, months)
print("\n雑音のある近さから、直近 w か月の傾き（最小二乗）で縮む速さを推定する")
for w in [3, 6, 12]:
    est = np.full(months, np.nan)
    for m in range(w - 1, months):
        t = np.arange(w)
        est[m] = -np.polyfit(t, obs[m - w + 1:m + 1], 1)[0]
    after = np.where(est > 0.01)[0]
    first = int(after[0]) if len(after) else -1
    calm = est[w - 1:48]
    false_months = int(np.sum(calm > 0.01))
    print(f"  w = {w:2d}  初めて 0.01 を超えた月 {first:3d}  落ち着いた時期（48か月目まで）の推定の標準偏差 {np.nanstd(calm):.4f}"
          f"  落ち着いた時期に 0.01 を超えた月の数 {false_months}")
print("窓が短いと早く気づくが、落ち着いた時期にも揺れが大きく、空振りが増える。窓が長いと揺れは小さいが、気づくのが遅れる。")
