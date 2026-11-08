"""
曲率の裁定（MUCAM）図 PY-01a・PY-01b：通貨のグラフを一周したときの、両替のずれ（離散的なホロノミー）
PY-01a：通貨を点、両替の経路を線とするグラフ（一周の経路と、各辺の対数レートと手数料）
PY-01b：ドル→人民元→タイバーツ→ドルの一周の計算の表
数値は、物語の中の架空の数値である。
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"

# ---- 物語の中の架空の数値 ----
loop = [
    ("USD", "CNY", 7.1000, 0.0010),
    ("CNY", "THB", 4.9500, 0.0005),
    ("THB", "USD", 1 / 35.30, 0.0015),
]
rows, total = [], 0.0
for a, b, rate, fee in loop:
    lr, lf = math.log(rate), math.log(1 - fee)
    rows.append((a, b, rate, lr, lf)); total += lr + lf
gain = math.exp(total) - 1

NAVY, GRAY, TEAL, GOLD, RED = "#1e2a3a", "#9aa3ad", "#1f8a8a", "#c8962e", "#a0471e"

# ================= PY-01a：通貨のグラフ =================
pos = {"USD": (0.0, 1.05), "CNY": (1.35, -0.45), "THB": (-1.35, -0.45), "AED": (0.0, -1.7)}
name = {"USD": "米ドル\nUSD", "CNY": "人民元\nCNY", "THB": "タイバーツ\nTHB", "AED": "UAE\nディルハム\nAED"}
fig = plt.figure(figsize=(16, 9), dpi=120)
ax = fig.add_axes([0.0, 0.10, 1.0, 0.80]); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.2, 1.6)

def edge(a, b, color, lw, ls="-", rad=0.0, arrow=True, z=1):
    ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>" if arrow else "-", mutation_scale=34,
                 color=color, lw=lw, ls=ls, shrinkA=58, shrinkB=58, zorder=z))

edge("USD", "AED", GRAY, 3, arrow=False)
edge("CNY", "AED", TEAL, 3, ls="--", arrow=False)
edge("THB", "AED", TEAL, 3, ls="--", arrow=False)
edge("USD", "CNY", GRAY, 3, rad=-0.28, arrow=False)
edge("USD", "CNY", GOLD, 7, rad=0.12, z=3)
edge("CNY", "THB", GOLD, 7, rad=0.12, z=3)
edge("THB", "USD", GOLD, 7, rad=0.12, z=3)
for k, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), 0.42, color=NAVY if k != "AED" else "#56606b", zorder=4))
    ax.text(x, y, name[k], ha="center", va="center", color="white", fontsize=19, weight="bold", zorder=5, linespacing=1.15)
lab = {("USD", "CNY"): (1.55, 0.55), ("CNY", "THB"): (0.0, -0.05), ("THB", "USD"): (-1.55, 0.55)}
for a, b, rate, lr, lf in rows:
    x, y = lab[(a, b)]
    ax.text(x, y, f"log レート {lr:+.4f}\n手数料 {lf:+.4f}", ha="center", va="center", fontsize=18,
            color="#5a4310", bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3e2", ec=GOLD, lw=1.6), zorder=6)
fig.text(0.5, 0.955, "通貨のグラフ：一周の経路（ドル → 人民元 → タイバーツ → ドル）", ha="center", va="top",
         fontsize=26, weight="bold", color=NAVY)
fig.text(0.5, 0.055, "金色の矢印：一周の経路　　灰色の実線：従来の送金網　　青緑の点線：中央銀行デジタル通貨の決済網",
         ha="center", va="center", fontsize=17, color="#444")
fig.text(0.98, 0.02, "数値は、物語の中の架空の数値", ha="right", fontsize=13, color="#777")
fig.savefig("mucam_py01a_currency_graph.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)

# ================= PY-01b：一周の計算の表 =================
fig = plt.figure(figsize=(16, 9), dpi=120)
tx = fig.add_axes([0.06, 0.04, 0.88, 0.86]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1); tx.set_autoscale_on(False)
fig.text(0.5, 0.955, "一周の和が 0 からずれる ── 離散的なホロノミー", ha="center", va="top", fontsize=28, weight="bold", color=NAVY)
tx.text(0.0, 0.97, "各辺の対数レートと手数料を足し合わせる。ずれがなければ、一周の和は 0 になる。", fontsize=19, va="top", color="#333")
cols = [0.00, 0.30, 0.52, 0.76]
y = 0.80
for c, h in zip(cols, ["経路", "レート", "log レート", "手数料（log）"]):
    tx.text(c, y, h, fontsize=21, weight="bold", color=NAVY)
for a, b, rate, lr, lf in rows:
    y -= 0.11
    for c, v in zip(cols, [f"{a} → {b}", f"{rate:.4f}", f"{lr:+.4f}", f"{lf:+.4f}"]):
        tx.text(c, y, v, fontsize=21)
y -= 0.07; tx.plot([0, 1], [y + 0.035, y + 0.035], color="#999", lw=1.5)
tx.text(0.00, y - 0.04, "一周の和", fontsize=24, weight="bold")
tx.text(0.52, y - 0.04, f"{total:+.4f}", fontsize=26, weight="bold", color=RED)
tx.text(0.00, y - 0.17, f"1ドルから出発して一周すると、{1+gain:.4f} ドルで戻る（{gain*100:+.2f}%）。", fontsize=21, color=RED)
tx.text(0.00, y - 0.30, "経路の網の目（送金網の切断や、新しい決済網）が変わると、ずれが現れる場所も変わる。", fontsize=19, color="#333")
fig.text(0.98, 0.02, "数値は、物語の中の架空の数値", ha="right", fontsize=13, color="#777")
fig.savefig("mucam_py01b_loop_sum.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)
print(f"一周の和 = {total:+.4f}、増減 = {gain*100:+.2f}%")
