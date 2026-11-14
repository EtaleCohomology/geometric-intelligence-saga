"""記事の図を、実験と同じ架空のデータから描く（図の数値は、gi_kgi_kpi_jax.py の計算そのもの）"""
import numpy as np, jax.numpy as jnp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import gi_kgi_kpi_jax as g
fp = [f for f in font_manager.findSystemFonts() if "NotoSansCJK" in f and "Regular" in f]
plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp[0]).get_name()
plt.rcParams["font.size"] = 13
OUT = "figs/"
import os; os.makedirs(OUT, exist_ok=True)

# 図1：地形と崖の領域、局面Aの道筋
xs = np.linspace(-1.0, 3.0, 161); ys = np.linspace(-1.5, 1.5, 121)
X, Y = np.meshgrid(xs, ys)
H = np.vectorize(lambda a, b: float(g.height(jnp.array([a, b]))))(X, Y)
K = np.vectorize(lambda a, b: float(g.gauss_curvature(jnp.array([a, b]))))(X[::2, ::2], Y[::2, ::2])
fig, ax = plt.subplots(figsize=(10, 6))
cs = ax.contourf(X, Y, H, levels=20, cmap="Blues_r")
ax.contourf(X[::2, ::2], Y[::2, ::2], (K < g.CLIFF_K).astype(float), levels=[0.5, 1.5], colors=["#d62728"], alpha=0.45)
ax.plot(np.linspace(0, 0.85, 50), np.zeros(50), color="#ff9f1c", lw=4, label="局面Aの道筋（期初 → 期末）")
ax.scatter([0], [0], s=120, color="white", edgecolor="black", zorder=5, label="局面Bの位置（谷の底）")
ax.text(1.25, 0.95, "崖の領域\n（曲率が大きく負）", color="#8b0000", fontsize=15, ha="center")
ax.set_xlabel("地図の座標1"); ax.set_ylabel("地図の座標2"); ax.set_title("架空の地形：谷と崖")
fig.colorbar(cs, ax=ax, label="地形の高さ"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "article_b5_09_terrain_figure.png", dpi=150)

# 図2：VaR と崖までの近さ（局面A、局面B）
T = 500
pathA = np.stack([np.linspace(0.0, 0.85, T), np.zeros(T)], 1); pnlA = g.simulate(pathA, 1.0, 1)
pathB = np.zeros((T, 2)); volB = np.where((np.arange(T) > 380) & (np.arange(T) < 430), 3.0, 1.0); pnlB = g.simulate(pathB, volB, 2)
pts = g.cliff_points()
def rolling_var(p, w=120):
    return np.array([np.quantile(-p[max(0, t - w):t], 0.99) if t >= w else np.nan for t in range(1, len(p) + 1)])
dA = np.array([g.distance_to_cliff(jnp.array(z), pts) for z in pathA[::10]])
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for col, (name, pnl, d) in enumerate([("局面A：VaR は落ち着いている", pnlA, dA), ("局面B：VaR は跳ねる", pnlB, np.full(50, 1.0))]):
    axes[0, col].plot(rolling_var(pnl), color="#1f77b4"); axes[0, col].set_title(name); axes[0, col].set_ylabel("VaR（99%、億円）"); axes[0, col].set_ylim(0, 5.8)
    axes[1, col].plot(np.arange(0, T, 10), d, color="#d62728", lw=3); axes[1, col].set_ylabel("崖までの近さ"); axes[1, col].set_ylim(0, 1.1)
    axes[1, col].set_xlabel("日")
axes[1, 0].annotate("崖に近づいている", xy=(450, dA[-5]), xytext=(150, 0.25), arrowprops=dict(arrowstyle="->"), fontsize=13)
axes[1, 1].annotate("谷の底のまま", xy=(400, 1.0), xytext=(150, 0.6), arrowprops=dict(arrowstyle="->"), fontsize=13)
fig.suptitle("速度計（VaR、120日の移動窓）と地形（崖までの近さ）は、別のことを言っている")
fig.tight_layout(); fig.savefig(OUT + "article_b5_10_var_vs_distance.png", dpi=150)

# 図3：当たり・見逃し・空振り（実験2の数値）
fig, ax = plt.subplots(figsize=(10, 5))
labels = ["崖までの近さ", "既存の KRI"]; hits = [18, 3]; miss = [0, 15]; fa = [0.100, 0.014]
x = np.arange(2)
ax.bar(x - 0.2, hits, 0.4, color="#2ca02c", label="当たり（18件中）"); ax.bar(x + 0.2, miss, 0.4, color="#d62728", label="見逃し（18件中）")
for i in range(2):
    ax.text(x[i], 19.5, f"空振りの率 {fa[i]:.3f}", ha="center", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 22); ax.set_ylabel("出来事の数")
ax.set_title("学習に使っていない60期間での評価（架空のデータ）"); ax.legend(loc="center right")
fig.tight_layout(); fig.savefig(OUT + "article_b5_11_signals_chart.png", dpi=150)
print("ok")
