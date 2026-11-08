"""
曲率の裁定（MUCAM）図 PY-05a・PY-05b：k形式が測るもの
PY-05a：1形式＝矢印１本に数を１つ返す（増え方の目盛り）、2形式＝向きのついた面積、3形式＝向きのついた体積
PY-05b：同じ矢印を２本並べると、平行四辺形はつぶれて面積は0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, FancyArrowPatch, Arc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
NAVY, TEAL, GOLD, RED, GRAY = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e", "#9aa3ad"

def arrow(ax, p, q, color, lw=4, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=30, color=color, lw=lw, zorder=z))

# ================= PY-05a =================
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.text(0.5, 0.975, "k形式が測るもの", ha="center", va="top", fontsize=30, weight="bold", color=NAVY)

# --- 左：1形式 ---
ax = fig.add_axes([0.02, 0.10, 0.31, 0.72]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.3, 4.3); ax.set_ylim(-0.6, 4.3)
for k in range(9):                      # 1形式を、等間隔の目盛りの線（等高線）で表す
    x = 0.45 * k
    ax.plot([x, x + 1.2], [0, 4.0], color=GRAY, lw=1.6, zorder=1)
arrow(ax, (0.2, 1.0), (3.3, 2.0), TEAL)
ax.text(1.8, 2.15, "v", fontsize=26, color=TEAL, weight="bold", style="italic")
ax.text(2.0, -0.45, "矢印が、目盛りの線を何本横切るか", ha="center", fontsize=17, color="#333")
fig.text(0.175, 0.845, "1形式", ha="center", fontsize=26, weight="bold", color=NAVY)
fig.text(0.175, 0.80, "矢印１本 → 数１つ（増え方）", ha="center", fontsize=18, color="#333")

# --- 中：2形式 ---
ax = fig.add_axes([0.35, 0.10, 0.30, 0.72]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.6, 4.3)
o, u, w = np.array([0.3, 0.6]), np.array([3.0, 0.6]), np.array([0.9, 2.6])
ax.add_patch(Polygon([o, o + u, o + u + w, o + w], closed=True, fc="#f3dfb4", ec=GOLD, lw=2.5, zorder=1))
arrow(ax, o, o + u, TEAL); arrow(ax, o, o + w, RED)
ax.text(*(o + u * 0.55 + np.array([0, -0.45])), "v", fontsize=26, color=TEAL, weight="bold", style="italic")
ax.text(*(o + w * 0.5 + np.array([-0.5, 0])), "w", fontsize=26, color=RED, weight="bold", style="italic")
c = o + (u + w) / 2
ax.add_patch(Arc(c, 1.3, 1.3, theta1=20, theta2=320, color=NAVY, lw=2.5))
ax.add_patch(FancyArrowPatch(c + 0.65 * np.array([np.cos(np.radians(300)), np.sin(np.radians(300))]),
                             c + 0.65 * np.array([np.cos(np.radians(320)), np.sin(np.radians(320))]),
                             arrowstyle="-|>", mutation_scale=22, color=NAVY, lw=2.5))
ax.text(2.0, -0.45, "回る向きのついた面積", ha="center", fontsize=17, color="#333")
fig.text(0.50, 0.845, "2形式", ha="center", fontsize=26, weight="bold", color=NAVY)
fig.text(0.50, 0.80, "矢印２本 → 向きのついた面積", ha="center", fontsize=18, color="#333")

# --- 右：3形式 ---
ax = fig.add_axes([0.67, 0.10, 0.31, 0.72], projection="3d"); ax.set_axis_off()
o = np.zeros(3); a, b, cc = np.array([1.0, 0, 0]), np.array([0.35, 1.0, 0]), np.array([0.2, 0.25, 1.0])
V = [o, a, a + b, b, cc, a + cc, a + b + cc, b + cc]
faces = [[V[0], V[1], V[2], V[3]], [V[4], V[5], V[6], V[7]], [V[0], V[1], V[5], V[4]],
         [V[2], V[3], V[7], V[6]], [V[1], V[2], V[6], V[5]], [V[0], V[3], V[7], V[4]]]
ax.add_collection3d(Poly3DCollection(faces, facecolor="#cfe3e3", edgecolor="#5a6b78", linewidth=1.4, alpha=0.30))
for vec, col, lab in [(a, TEAL, "u"), (b, RED, "v"), (cc, GOLD, "w")]:
    ax.quiver(0, 0, 0, *vec, color=col, linewidth=5, arrow_length_ratio=0.15, zorder=10)
    ax.text(*(vec * 1.18 + np.array([0.0, 0.0, 0.03])), lab, fontsize=26, color=col, weight="bold", style="italic")
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.4); ax.set_zlim(0, 1.2); ax.view_init(elev=22, azim=-60)
fig.text(0.825, 0.845, "3形式", ha="center", fontsize=26, weight="bold", color=NAVY)
fig.text(0.825, 0.80, "矢印３本 → 向きのついた体積", ha="center", fontsize=18, color="#333")
fig.text(0.825, 0.13, "向きのついた体積", ha="center", fontsize=17, color="#333")
fig.text(0.98, 0.02, "n次元の空間では、n形式まである", ha="right", fontsize=15, color="#555")
fig.savefig("mucam_py05a_k_forms.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)

# ================= PY-05b =================
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.text(0.5, 0.94, "同じ矢印を２本並べると、平行四辺形はつぶれて、面積は 0", ha="center", va="top", fontsize=28, weight="bold", color=NAVY)
for i, (ang, title) in enumerate([(55, "２本の向きが違う"), (25, "２本の向きが近い"), (0, "同じ向き（v と v）")]):
    ax = fig.add_axes([0.03 + i * 0.32, 0.12, 0.30, 0.66]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.3, 4.3); ax.set_ylim(-1.0, 2.6)
    o = np.array([0.1, 0.3]); u = np.array([2.5, 0.0]); w = 1.5 * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    if ang == 0: w = np.array([2.5, 0.0])
    o = o + np.array([0, 0.6])
    ax.add_patch(Polygon([o, o + u, o + u + w, o + w], closed=True, fc="#f3dfb4", ec=GOLD, lw=2.5))
    if ang == 0:
        arrow(ax, o + np.array([0, 0.12]), o + u + np.array([0, 0.12]), TEAL); arrow(ax, o - np.array([0, 0.12]), o + w - np.array([0, 0.12]), RED, lw=3)
        ax.text(2.75, 0.72, "v", fontsize=24, color=TEAL, weight="bold", style="italic"); ax.text(2.75, 0.22, "v", fontsize=24, color=RED, weight="bold", style="italic")
    else:
        arrow(ax, o, o + u, TEAL); arrow(ax, o, o + w, RED, lw=3)
    area = abs(u[0] * w[1] - u[1] * w[0])
    ax.text(1.9, -0.75, f"面積 = {area:.2f}", ha="center", fontsize=22, color=RED if area < 1e-9 else NAVY, weight="bold")
    fig.text(0.18 + i * 0.32, 0.80, title, ha="center", fontsize=21, color="#333")
fig.text(0.5, 0.05, "だから 2形式では、v と v を並べると 0。順番を入れ替えると、符号が反対になる（交代的）。", ha="center", fontsize=19, color="#333")
fig.savefig("mucam_py05b_collapse.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)
print("ok")
