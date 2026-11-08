"""
曲率の裁定（MUCAM）図 PY-07a・PY-07b：大きな輪のずれを、三角形ごとに分ける（離散版のストークスの定理）
両替の対数レートのずれ（理論値からの差、単位：ベーシスポイント、1bp = 0.01%）を、辺の上の値（1形式）として置く。
大きな輪 ドル→人民元→ディルハム→バーツ→ドル のずれは、
三角形がそろっている部分（三角形ごとの一周のずれ）と、三角形で埋められない部分（穴のまわりのずれ）の和に分かれる。
人民元とバーツの経路は切断されたまま（点線）で、ユーロとディルハムの経路もないため、ユーロ・人民元・ディルハム・バーツに囲まれた部分は三角形で埋められない。
数値は架空のもの。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, FancyArrowPatch

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Bold" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
NAVY, TEAL, GOLD, RED, GRAY = "#1e2a3a", "#1f8a8a", "#c8962e", "#a0471e", "#9aa3ad"

# ---- 辺の上の値（向きのある辺。逆向きは符号が反対）----
sig = {("USD", "CNY"): 12, ("CNY", "EUR"): -5, ("EUR", "USD"): -4,
       ("EUR", "THB"): 9, ("THB", "USD"): -6,
       ("CNY", "AED"): 8, ("AED", "THB"): -78}
def s(a, b): return sig[(a, b)] if (a, b) in sig else -sig[(b, a)]
def loop(*v): return sum(s(v[i], v[(i + 1) % len(v)]) for i in range(len(v)))
T1 = loop("USD", "CNY", "EUR")          # 三角形１
T2 = loop("USD", "EUR", "THB")          # 三角形２
H  = loop("EUR", "CNY", "AED", "THB")   # 穴のまわり
L  = loop("USD", "CNY", "AED", "THB")   # 大きな輪
assert L == T1 + T2 + H                 # 離散版のストークスの定理による分解の検算

pos = {"USD": (-1.6, 1.4), "CNY": (1.6, 1.4), "AED": (1.6, -1.4), "THB": (-1.6, -1.4), "EUR": (-0.75, 0.25)}
name = {"USD": "ドル", "CNY": "人民元", "AED": "ディルハム", "THB": "バーツ", "EUR": "ユーロ"}

# ================= PY-07a：グラフ =================
fig = plt.figure(figsize=(16, 9), dpi=120)
ax = fig.add_axes([0.03, 0.07, 0.94, 0.80]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-1.9, 1.9)
P = lambda k: np.array(pos[k])
ax.add_patch(Polygon([P("USD"), P("CNY"), P("EUR")], fc="#cfe3e3", ec="none", zorder=0))
ax.add_patch(Polygon([P("USD"), P("EUR"), P("THB")], fc="#f3dfb4", ec="none", zorder=0))
for a, b in [("USD", "EUR"), ("CNY", "EUR"), ("THB", "EUR")]:
    ax.plot(*zip(pos[a], pos[b]), color=GRAY, lw=3, zorder=1)
ax.plot(*zip(pos["CNY"], pos["THB"]), color=RED, lw=3, ls=(0, (4, 4)), zorder=1)
for a, b in [("USD", "CNY"), ("CNY", "AED"), ("AED", "THB"), ("THB", "USD")]:
    ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=30, color=GOLD, lw=6, shrinkA=34, shrinkB=34, zorder=2))
for k, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), 0.30, color=NAVY, zorder=4))
    ax.text(x, y, name[k], ha="center", va="center", color="white", fontsize=17 if len(name[k]) < 5 else 14, weight="bold", zorder=5)
ax.text(*((P("USD") + P("CNY") + P("EUR")) / 3), f"三角形１\n{T1:+d} bp", ha="center", va="center", fontsize=20, weight="bold", color=TEAL)
ax.text(*((P("USD") + P("EUR") + P("THB")) / 3 + np.array([0.08, 0.0])), f"三角形２\n{T2:+d} bp", ha="center", va="center", fontsize=20, weight="bold", color="#8a6a20")
ax.text(0.85, -0.35, "穴", ha="center", va="center", fontsize=44, weight="bold", color=RED)
ax.text(0.85, -0.85, "三角形で埋められない", ha="center", va="center", fontsize=15, color=RED)
ax.text(-0.45, -0.95, "人民元―バーツ（切断されたまま）", ha="center", va="center", fontsize=14, color=RED, rotation=41)
fig.text(0.5, 0.955, "大きな輪のずれを、三角形ごとに分ける", ha="center", va="top", fontsize=28, weight="bold", color=NAVY)
fig.text(0.5, 0.04, "金色の矢印：大きな輪（ドル→人民元→ディルハム→バーツ→ドル）　　塗った三角形：３辺がそろっている部分　　赤の点線：切断された経路",
         ha="center", fontsize=14, color="#444")
fig.text(0.985, 0.01, "数値は架空のもの。bp（ベーシスポイント）＝0.01%", ha="right", fontsize=11.5, color="#777")
fig.savefig("mucam_py07a_big_loop_graph.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)

# ================= PY-07b：内訳の表 =================
fig = plt.figure(figsize=(16, 9), dpi=120)
tx = fig.add_axes([0.08, 0.05, 0.84, 0.82]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1); tx.set_autoscale_on(False)
fig.text(0.5, 0.955, "大きな輪のずれの内訳", ha="center", va="top", fontsize=29, weight="bold", color=NAVY)
rows = [("三角形１（ドル・人民元・ユーロ）", T1, TEAL, "どの通貨の組が出どころか、分かる"),
        ("三角形２（ドル・ユーロ・バーツ）", T2, "#8a6a20", "どの通貨の組が出どころか、分かる"),
        ("三角形ごとのずれの合計", T1 + T2, NAVY, ""),
        ("穴のまわりに残るずれ", H, RED, "１つの通貨の組のせいにはできない")]
y = 0.84
for lab, v, c, note in rows:
    tx.text(0.0, y, lab, fontsize=22, weight="bold" if "合計" in lab or "穴" in lab else "normal", color=c, va="center")
    tx.text(0.62, y, f"{v:+d} bp", fontsize=26, weight="bold", color=c, va="center")
    if note: tx.text(0.0, y - 0.065, note, fontsize=15, color="#555", va="center")
    y -= 0.17
tx.plot([0, 1], [y + 0.07, y + 0.07], color="#999", lw=1.5)
tx.text(0.0, y - 0.02, "大きな輪の一周のずれ", fontsize=24, weight="bold", color=NAVY, va="center")
tx.text(0.62, y - 0.02, f"{L:+d} bp", fontsize=30, weight="bold", color=NAVY, va="center")
tx.text(0.0, y - 0.12, f"＝ 三角形ごとの合計（{T1+T2:+d}）＋ 穴のまわりのずれ（{H:+d}）", fontsize=19, color="#333", va="center")
fig.text(0.985, 0.01, "数値は架空のもの。bp（ベーシスポイント）＝0.01%", ha="right", fontsize=11.5, color="#777")
fig.savefig("mucam_py07b_loop_breakdown.jpg", dpi=120, pil_kwargs={"quality": 92}); plt.close(fig)
print(f"三角形１ {T1}、三角形２ {T2}、穴 {H}、大きな輪 {L}")
