# s04 保全の順番：スミスの規則と、期限（第3章「重さと、かかる時間と、期限」）
# 数はすべて説明のための架空の値である。
from itertools import permutations

# 名前: (大事さの重さ, 写すのにかかる時間[時間], 上書きされるまでの期限[時間])
jobs = {
    "ネットワーク機器の記録": (3, 1.0, 1.5),
    "AI エージェントの記録": (4, 1.0, 72),
    "区画ＡとＢのメモリ": (4, 3.0, 100),
    "クラウドの記録": (2, 2.0, 168),
}


def cost(order):
    t = 0; total = 0; late = []
    for name in order:
        w, p, due = jobs[name]
        t += p; total += w * t
        if t > due: late.append(name)
    return total, late


smith = sorted(jobs, key=lambda n: -jobs[n][0] / jobs[n][1])
print("スミスの規則（重さ ÷ 時間 の大きい順）:", " → ".join(smith))
print("  重さ付きの完了時刻の合計:", cost(smith)[0], " 期限に遅れるもの:", cost(smith)[1] or "なし")

best = min((o for o in permutations(jobs) if not cost(o)[1]), key=lambda o: cost(o)[0])
print("期限を守る並べ方のうち、合計がいちばん小さいもの:", " → ".join(best))
print("  重さ付きの完了時刻の合計:", cost(best)[0])

# 交換子：保全(P) と 切り離し(I) の順番を入れ替えると、残るものが変わる
memory = {"メモリの中身"}
after_PI = memory | {"写し"}              # 保全してから切り離す：写しが残る
after_IP = set()                           # 切り離して再起動してから保全：写すものがない
print("P のあとに I:", after_PI, " / I のあとに P:", after_IP or "{}",
      " → 差（交換子）は０ではない")
