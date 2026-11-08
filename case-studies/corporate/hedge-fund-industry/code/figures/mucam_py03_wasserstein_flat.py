"""
曲率の裁定（MUCAM）図 PY-03：何の距離で見るかで、空間の形が変わる
左：フィッシャー情報計量で見た (μ, σ) 平面に、同じ半径（フィッシャー・ラオ距離）の円を描く。σ が小さいほど小さく見える（双曲空間）。
右：ワッサースタイン距離（W2）で見た同じ平面に、同じ半径の円を描く。どこでも同じ大きさ（平ら）。
1次元の正規分布どうしの W2 距離は sqrt((μ1−μ2)² + (σ1−σ2)²) で、(μ, σ) 平面のユークリッド距離と一致する。
フィッシャー情報計量 ds² = (dμ² + 2dσ²)/σ² のもとで、中心 (μ0, σ0)、半径 ρ の円は、
x = μ/√2 の座標で、中心 (x0, σ0 cosh R)、半径 σ0 sinh R のふつうの円（R = ρ/√2）になる。
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
NAVY, TEAL, GOLD, RED = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e"

centers = [(-2.0, 1.0), (0.0, 2.0), (2.0, 4.0)]   # (μ, σ) [%]
th = np.linspace(0, 2 * np.pi, 300)

def fisher_circle(mu0, s0, rho):
    R = rho / np.sqrt(2)
    cx, cy, rr = mu0 / np.sqrt(2), s0 * np.cosh(R), s0 * np.sinh(R)
    return np.sqrt(2) * (cx + rr * np.cos(th)), cy + rr * np.sin(th)

def fr_dist(m1, s1, m2, s2):
    return np.sqrt(2) * np.arccosh(1 + ((m1 - m2) ** 2 / 2 + (s1 - s2) ** 2) / (2 * s1 * s2))

fig = plt.figure(figsize=(16, 9), dpi=120)
fig.text(0.5, 0.965, "何の距離で見るかで、空間の形が変わる", ha="center", va="top", fontsize=30, weight="bold", color=NAVY)
for k, (title, sub, col) in enumerate([("フィッシャー情報計量で見る", "同じ半径（0.6）の円が、σ が小さいほど小さく見える ＝ 双曲空間", GOLD),
                                       ("ワッサースタイン距離で見る", "同じ半径（0.6%）の円が、どこでも同じ大きさ ＝ 平ら", TEAL)]):
    ax = fig.add_axes([0.06 + k * 0.49, 0.13, 0.42, 0.62]); ax.set_aspect("equal")
    for (m0, s0) in centers:
        if k == 0:
            x, y = fisher_circle(m0, s0, 0.6)
            assert np.allclose(fr_dist(m0, s0, x, y), 0.6, atol=1e-6)   # 検算：円の上の点は、中心からちょうど 0.6
        else:
            x, y = m0 + 0.6 * np.cos(th), s0 + 0.6 * np.sin(th)
        ax.fill(x, y, color=col, alpha=0.25); ax.plot(x, y, color=col, lw=3)
        ax.scatter([m0], [s0], s=70, color=NAVY, zorder=5)
    ax.set_xlim(-4, 4.5); ax.set_ylim(0, 6.5)
    ax.set_xlabel("平均 μ（%）", fontsize=19); ax.set_ylabel("標準偏差 σ（%）", fontsize=19)
    ax.tick_params(labelsize=15); ax.grid(alpha=0.25)
    fig.text(0.27 + k * 0.49, 0.85, title, ha="center", fontsize=24, weight="bold", color=col)
    fig.text(0.27 + k * 0.49, 0.80, sub, ha="center", fontsize=15, color="#333")
fig.text(0.5, 0.035, "同じ正規分布の集まりでも、"
         "見分けやすさで測ると曲がって（双曲空間）、確率の重みを運ぶ手間で測ると平らに見える。", ha="center", fontsize=16, color="#333")
fig.savefig("mucam_py03_wasserstein_flat.jpg", dpi=120, pil_kwargs={"quality": 92})
print("ok")
