"""
曲率の裁定（MUCAM）図 PY-02：平穏な市場と危機の市場の、分布どうしの距離
平均 μ を横軸、標準偏差 σ を縦軸とする平面（上半平面）に、
平穏な市場 N(0.1%, 1%) と危機の市場 N(−2%, 5%) を置き、
まっすぐな線（ユークリッド距離）と、フィッシャー情報計量の測地線（フィッシャー・ラオ距離）を描き分ける。
正規分布の族のフィッシャー情報計量は ds² = (dμ² + 2dσ²)/σ²。x = μ/√2 と置くと、双曲平面（上半平面）の計量の2倍になる。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
NAVY, TEAL, GOLD, RED, GRAY = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e", "#7a8590"

m1, s1 = 0.001, 0.01      # 平穏な市場
m2, s2 = -0.02, 0.05      # 危機の市場
d_euc = np.hypot(m1 - m2, s1 - s2)
d_fr = np.sqrt(2) * np.arccosh(1 + ((m1 - m2) ** 2 / 2 + (s1 - s2) ** 2) / (2 * s1 * s2))

# 測地線：x = μ/√2 の座標では、x軸に中心を持つ半円
x1, x2 = m1 / np.sqrt(2), m2 / np.sqrt(2)
c = (x1**2 + s1**2 - x2**2 - s2**2) / (2 * (x1 - x2))
r = np.hypot(x1 - c, s1)
t1, t2 = np.arctan2(s1, x1 - c), np.arctan2(s2, x2 - c)
t = np.linspace(t1, t2, 400)
geo_mu, geo_s = np.sqrt(2) * (c + r * np.cos(t)), r * np.sin(t)
# 検算：測地線に沿って、計量で長さを足し合わせる
dmu, ds = np.diff(geo_mu), np.diff(geo_s); sm = (geo_s[1:] + geo_s[:-1]) / 2
length = np.sum(np.sqrt(dmu**2 + 2 * ds**2) / sm)

fig = plt.figure(figsize=(16, 9), dpi=120)
ax = fig.add_axes([0.07, 0.12, 0.55, 0.74])
P = 100  # パーセントで表示
full_t = np.linspace(0.02, np.pi - 0.02, 400)
ax.plot(np.sqrt(2) * (c + r * np.cos(full_t)) * P, r * np.sin(full_t) * P, color=GOLD, lw=1.5, ls=":", alpha=0.7)
ax.plot([m1 * P, m2 * P], [s1 * P, s2 * P], color=GRAY, lw=3.5, ls="--", label="まっすぐな線（ユークリッド距離）")
ax.plot(geo_mu * P, geo_s * P, color=GOLD, lw=5, label="フィッシャー情報計量の測地線")
ax.scatter([m1 * P], [s1 * P], s=260, color=TEAL, zorder=5); ax.scatter([m2 * P], [s2 * P], s=260, color=RED, zorder=5)
ax.annotate("平穏な市場\nN(0.1%, 1%)", (m1 * P, s1 * P), xytext=(0.6, 1.2), fontsize=19, color=TEAL, weight="bold")
ax.annotate("危機の市場\nN(−2%, 5%)", (m2 * P, s2 * P), xytext=(-3.9, 5.4), fontsize=19, color=RED, weight="bold")
ax.set_xlim(-4.2, 2.2); ax.set_ylim(0, 6.6)
ax.set_xlabel("平均 μ（%）", fontsize=20); ax.set_ylabel("標準偏差 σ（%）", fontsize=20)
ax.tick_params(labelsize=16); ax.grid(alpha=0.25)
ax.legend(fontsize=16, loc="upper right", frameon=True)
ax.text(-4.05, 0.25, "σ = 0（横軸）は、双曲空間の\"果て\"", fontsize=14, color="#666")

fig.text(0.5, 0.955, "平穏な市場と危機の市場は、どれくらい遠いか", ha="center", va="top", fontsize=28, weight="bold", color=NAVY)
tx = 0.66
fig.text(tx, 0.80, "ユークリッド距離", fontsize=21, weight="bold", color=GRAY)
fig.text(tx, 0.74, f"{d_euc:.3f}", fontsize=34, weight="bold", color=GRAY)
fig.text(tx, 0.68, "平均と標準偏差の差から計算", fontsize=16, color="#444")
fig.text(tx, 0.58, "フィッシャー・ラオ距離", fontsize=21, weight="bold", color=GOLD)
fig.text(tx, 0.52, f"{d_fr:.1f}", fontsize=34, weight="bold", color=GOLD)
fig.text(tx, 0.455, "データからの\"見分けやすさ\"で測る", fontsize=16, color="#444")
fig.text(tx, 0.395, "測地線は、標準偏差の大きい側へ\nふくらんで進む。σ が大きいところでは、\n分布どうしを見分けにくく、\n距離が\"短く\"なるから。", fontsize=16, color="#333", linespacing=1.6, va="top")
fig.text(tx, 0.08, "２つの距離は、単位も意味も違うので、\n数字の大きさをそのまま比べない。", fontsize=15, color=RED, linespacing=1.5)
fig.savefig("mucam_py02_fisher_vs_euclid.jpg", dpi=120, pil_kwargs={"quality": 92})
print(f"ユークリッド距離 {d_euc:.4f}、フィッシャー・ラオ距離（式）{d_fr:.4f}、測地線に沿って足した長さ {length:.4f}")
