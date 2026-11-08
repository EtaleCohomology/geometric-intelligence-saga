"""
曲率の裁定（MUCAM）図 PY-06：相関が1に近づくと、散らばりの「体積」（共分散行列の行列式）がつぶれる
２つの銘柄の日次リターン（標準偏差はどちらも1%）の散布図を、相関 0、0.8、0.99 で描き、
95%の範囲を表す楕円を重ねる。楕円の面積は、共分散行列の行列式の平方根に比例する。数値は架空のもの。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
NAVY, TEAL, GOLD, RED = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e"

rng = np.random.default_rng(7)
s = 1.0                       # 標準偏差 1%（単位：%）
k = np.sqrt(5.991)            # 2次元正規分布の95%の範囲（カイ二乗、自由度2）
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.text(0.5, 0.965, "相関が 1 に近づくと、散らばりの\"体積\"がつぶれる", ha="center", va="top", fontsize=29, weight="bold", color=NAVY)
for i, rho in enumerate([0.0, 0.8, 0.99]):
    C = np.array([[s * s, rho * s * s], [rho * s * s, s * s]])
    X = rng.multivariate_normal([0, 0], C, size=600)
    det = np.linalg.det(C)
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    area = np.pi * k * k * np.sqrt(det)           # 楕円の面積 ＝ π k² √det
    ax = fig.add_axes([0.04 + i * 0.325, 0.27, 0.29, 0.58]); ax.set_aspect("equal")
    ax.scatter(X[:, 0], X[:, 1], s=9, color=TEAL, alpha=0.45)
    ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(max(w[0], 1e-12)), angle=ang,
                         fc=GOLD, alpha=0.25, ec=GOLD, lw=3))
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel("銘柄Aのリターン（%）", fontsize=16)
    if i == 0: ax.set_ylabel("銘柄Bのリターン（%）", fontsize=16)
    ax.tick_params(labelsize=13); ax.grid(alpha=0.25)
    cx = 0.185 + i * 0.325
    fig.text(cx, 0.165, f"相関 {rho:.2f}", ha="center", fontsize=26, weight="bold", color=NAVY)
    fig.text(cx, 0.10, f"行列式 {det:.4f}（%⁴）", ha="center", fontsize=22, weight="bold", color=RED if rho > 0.9 else NAVY)
    fig.text(cx, 0.05, f"楕円の面積 {area:.1f}（%²）", ha="center", fontsize=17, color="#444")
fig.text(0.98, 0.005, "数値は架空のもの。楕円は95%の範囲。面積 ＝ π × 5.991 × √(行列式)", ha="right", fontsize=12, color="#777")
fig.savefig("mucam_py06_volume_collapse.jpg", dpi=120, pil_kwargs={"quality": 92})
print("ok")
