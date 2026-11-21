"""05 記録を残せた人の歴史だけを見る偏り（生存者バイアス）と、選び方が作る見かけ（バークソンの偏り）（本編第4章）。

すべて架空の模型。数字は、偏りの向きと大きさを確かめるためだけのもの。
1. 生存者バイアス
   一族ごとに「早く離れたか」と「行き先の安全さ」を乱数で決める。早く離れた一族の一部は、
   行き先がのちに危うくなり、助からない（行き先の安全さが低いほど）。
   記録（手紙、日記）が残るのは、ほとんど生き延びた一族だけ。
   記録だけを見たときの「早く離れた一族の助かった割合」と、本当の割合を比べる。
2. バークソンの偏り
   「早く気づいた（直感が鋭い）」と「資産を国の外に持っていた」を、互いに無関係に決める。
   どちらかがあれば生き延びやすい。生き延びた一族（記録が残った一族）だけを見ると、
   ２つのあいだに、本当はない負の相関が現れる。
"""
import numpy as np

rng = np.random.default_rng(5)
N = 200_000

# ---- 1. 生存者バイアス -----------------------------------------------------
early = rng.random(N) < 0.35                          # 早く離れた一族
dest_safe = rng.random(N)                             # 行き先の安全さ（0〜1）
p_survive = np.where(early, 0.35 + 0.6 * dest_safe,   # 早く離れても、行き先しだいで助からない
                     0.12)
survived = rng.random(N) < p_survive
p_record = np.where(survived, 0.6, 0.03)             # 消された一族の記録は、ほとんど残らない
record = rng.random(N) < p_record

def rate(mask):
    return survived[mask].mean()

print("1. 生存者バイアス")
print(f"   本当の割合：早く離れた一族の助かった割合 {rate(early):.3f}、残った一族 {rate(~early):.3f}")
print(f"   記録だけ  ：早く離れた一族の助かった割合 {rate(early & record):.3f}、残った一族 {rate(~early & record):.3f}")
print(f"   記録が残った一族のうち、助からなかった一族の割合 {1 - survived[record].mean():.3f}（本当は {1 - survived.mean():.3f}）")
print("   記録だけを見ると、どちらの群も多くが助かったように見え、離れたかどうかによる差は縮んで見え、行き先の危うさはほとんど見えなくなる。")
print("   記録だけで推定した割合は、おもに記録が残る確率の差（生き延びた一族 0.6、助からなかった一族 0.03）で決まっていて、本当の割合を表していない。")

# 行き先の安全さ別に見る：記録だけでは、危うい行き先の失敗がほとんど見えない
print("\n   行き先の安全さ別の、早く離れた一族の助かった割合")
for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
    m = early & (dest_safe >= lo) & (dest_safe < hi)
    print(f"   安全さ {lo:.1f}〜{hi:.1f}：本当 {rate(m):.3f}  記録だけ {rate(m & record):.3f}  記録の数 {int((m & record).sum())}")

# ---- 2. バークソンの偏り ---------------------------------------------------
sharp = rng.random(N) < 0.3          # 早く気づいた
abroad = rng.random(N) < 0.3         # 資産を国の外に持っていた
p2 = 0.1 + 0.45 * sharp + 0.45 * abroad
p2 = np.minimum(p2, 0.95)
surv2 = rng.random(N) < p2

def corr(a, b):
    return np.corrcoef(a.astype(float), b.astype(float))[0, 1]

print("\n2. バークソンの偏り")
print(f"   全員での相関（早く気づいた, 資産が国の外）      {corr(sharp, abroad):+.3f}")
print(f"   生き延びた一族だけでの相関                    {corr(sharp[surv2], abroad[surv2]):+.3f}")
print(f"   生き延びた一族だけで見た、資産が国の外にあった割合：早く気づいた一族 {abroad[surv2 & sharp].mean():.3f}、"
      f"気づかなかった一族 {abroad[surv2 & ~sharp].mean():.3f}")
print("   生き延びた一族だけを見ると、『早く気づいた一族ほど、資産を外に持っていなかった』という見かけの関係が現れる。")
print("   生き延びたことが、２つの原因の共通の結果（合流点。２本の矢印が流れ込む点）になっていて、そこで選ぶと見かけの相関が生まれる。")
