"""
曲率の裁定（MUCAM）図 PY-04：損失の分布の裾 ── VaR の境目と、境目を超えた損失の平均（期待ショートフォール）
平穏な時期と、送金網が切断された後の時期の、日次の損失（%）の分布を並べ、
右の裾（上位10%）に一般化パレート分布を当てはめる（閾値を超えた損失の分析）。
VaR（97.5%）と期待ショートフォール（97.5%）は、当てはめた一般化パレート分布から計算する。数値は架空のもの。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
NAVY, TEAL, GOLD, RED, GRAY = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e", "#9aa3ad"

rng = np.random.default_rng(11)
periods = [("平穏な時期", stats.t(df=6, loc=0.0, scale=0.8).rvs(4000, random_state=rng), TEAL),
           ("送金網が切断された後", stats.t(df=3, loc=0.2, scale=1.1).rvs(4000, random_state=rng), RED)]
alpha, q_u = 0.975, 0.90

def evt(losses):
    u = np.quantile(losses, q_u)                       # 閾値（上位10%の境目）
    exc = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)     # 一般化パレート分布（形 xi、尺度 beta）
    p_u = np.mean(losses > u)
    var = u + beta / xi * (((1 - alpha) / p_u) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)              # xi < 1 のときの期待ショートフォール
    return u, xi, beta, p_u, var, es

fig = plt.figure(figsize=(16, 9), dpi=120)
fig.text(0.5, 0.985, "損失の裾の形が変わる ── VaR と期待ショートフォール（97.5%）", ha="center", va="top", fontsize=27, weight="bold", color=NAVY)
for i, (name, L, col) in enumerate(periods):
    u, xi, beta, p_u, var, es = evt(L)
    ax = fig.add_axes([0.06 + i * 0.48, 0.30, 0.42, 0.52])
    bins = np.linspace(-5, 10, 90)
    ax.hist(L, bins=bins, density=True, color=col, alpha=0.30, edgecolor="white")
    xs = np.linspace(u, 10, 300)
    ax.plot(xs, p_u * stats.genpareto.pdf(xs - u, xi, loc=0, scale=beta), color=GOLD, lw=4, label="裾に当てはめた一般化パレート分布")
    ax.axvspan(var, 10, color=RED, alpha=0.10)
    ax.axvline(var, color=NAVY, lw=3, label="VaR（97.5%）"); ax.axvline(es, color=RED, lw=3, ls="--", label="期待ショートフォール")
    ax.set_xlim(-5, 10); ax.set_xlabel("日次の損失（%）", fontsize=17); ax.tick_params(labelsize=13)
    if i == 0: ax.set_ylabel("頻度（密度）", fontsize=17)
    ax.legend(fontsize=13, loc="upper right")
    # 裾の拡大図
    ins = ax.inset_axes([0.55, 0.12, 0.42, 0.45])
    ib = np.linspace(u, 10, 40); tail = L[(L > u) & (L <= 10)]
    ins.hist(tail, bins=ib, weights=np.full(len(tail), 1.0 / (len(L) * (ib[1] - ib[0]))), color=col, alpha=0.30, edgecolor="white")   # 全体の標本数で割った密度（本体の図と同じ尺度）
    ins.plot(xs, p_u * stats.genpareto.pdf(xs - u, xi, loc=0, scale=beta), color=GOLD, lw=3)
    ins.axvspan(var, 10, color=RED, alpha=0.12); ins.axvline(var, color=NAVY, lw=2.5); ins.axvline(es, color=RED, lw=2.5, ls="--")
    ins.set_xlim(u, 10); ins.set_ylim(0, 0.08); ins.tick_params(labelsize=10)
    ins.set_title("裾の拡大", fontsize=13, color="#333")
    cx = 0.27 + i * 0.48
    fig.text(cx, 0.865, name, ha="center", fontsize=23, weight="bold", color=col)
    fig.text(cx, 0.19, f"VaR {var:.2f}%", ha="center", fontsize=24, weight="bold", color=NAVY)
    fig.text(cx, 0.125, f"期待ショートフォール {es:.2f}%", ha="center", fontsize=24, weight="bold", color=RED)
    fig.text(cx, 0.07, f"裾の形のパラメーター ξ = {xi:.2f}（大きいほど裾が厚い）", ha="center", fontsize=15, color="#444")
    print(f"{name}: 閾値 {u:.2f}、xi {xi:.3f}、beta {beta:.3f}、VaR {var:.2f}、ES {es:.2f}、標本の ES {L[L>=np.quantile(L,alpha)].mean():.2f}")
fig.text(0.98, 0.01, "数値は架空のもの。損失の上位10%に一般化パレート分布を当てはめて計算。赤く塗った部分の平均が期待ショートフォール。", ha="right", fontsize=11.5, color="#777")
fig.savefig("mucam_py04_tail_and_es.jpg", dpi=120, pil_kwargs={"quality": 92})
