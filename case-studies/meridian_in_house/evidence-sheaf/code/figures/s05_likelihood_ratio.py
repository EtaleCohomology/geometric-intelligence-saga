# s05 尤度比と、確率の取り違え（第7章「尤度比」）
# 説明のためのおもちゃの模型である。
#
# 観察 E：作業記録アプリのある区間の時刻だけが d 秒ずれ、同じ端末の基本ソフトの記録はずれていない。
# 仮説１（書き換え）：人が入力し直す。ずれの大きさは 0〜600 秒のどれでもありうる → 密度 1/600。
# 仮説２（時計の故障）：同じ時計を使う２つの記録は、ふつう一緒にずれる。
#   アプリだけが別にずれる故障が、割合 ε で起きると考え、そのときのずれは −600〜600 秒のどれか → 密度 ε/1200。
# 尤度比 LR = (1/600) / (ε/1200) = 2/ε。ε を見積もる模型を変えると LR が変わる。
from common import plt, savefig, NAVY, CYAN, AMBER

models = {"控えめな模型": 1 / 750, "中くらいの模型": 1 / 2000, "大胆な模型": 1 / 4500}
LR = {}
for name, eps in models.items():
    LR[name] = (1 / 600) / (eps / 1200)
    print(f"{name}: ε = 1/{round(1/eps)} → 尤度比 {LR[name]:,.0f}")

# 欧州の鑑識の指針（ENFSI、2015年）の言葉の目安
scale = [(1, 10, "弱い支持"), (10, 100, "中程度の支持"), (100, 1000, "中程度に強い支持"),
         (1000, 10000, "強い支持"), (10000, 1e6, "とても強い支持"), (1e6, float("inf"), "きわめて強い支持")]
for name, v in LR.items():
    word = next(w for lo, hi, w in scale if lo <= v < hi)
    print(f"  {name} の {v:,.0f} → {word}")

# 尤度比は「書き換えられた確率」ではない。前もっての見積もり（事前の見込み）がないと確率は出ない
print("\n事前の見込みを変えたときの、書き換えの確率（尤度比 4,000 のとき）")
for prior in [0.5, 0.01, 0.0001]:
    odds = prior / (1 - prior) * 4000
    print(f"  事前の見込み {prior:>7}: 観察のあとの確率 {odds/(1+odds):.4f}")
print("  → 『4,000分の3,999』は、事前の見込みが半々のときだけ成り立つ")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(list(LR.keys()), list(LR.values()), color=[CYAN, NAVY, AMBER])
ax.axvspan(1000, 10000, color="#eeeeee", zorder=0)
ax.text(10100, -0.45, "灰色：強い支持（1,000〜10,000）", fontsize=8, ha="right")
for k, v in enumerate(LR.values()):
    ax.text(v + 120, k, f"{v:,.0f}", va="center", fontsize=9)
ax.set_xlim(0, 11000); ax.set_xlabel("尤度比")
ax.set_title("時計の揺れの模型を変えても、尤度比は 1,000〜10,000 に入る", fontsize=10)
savefig(fig, "s05_likelihood_ratio.png")
