"""
敵の多様体 補助文書03 コード12：ベイズ説得 ── 嘘をつかずに、判断を動かす
ある省の状態は「危機あり」（事前確率 0.6）か「危機なし」。中央は、危機ありの事後確率が 0.5 以上なら介入する。
地方（送り手）は介入を避けたい。地方は報告の方針（各状態で「平穏」「警報」をどの確率で出すか）を先に決め、
中央はその方針を知ったうえで報告を読む。地方は嘘をつかない（方針は公開されている）。数値は架空のもの。
"""
import numpy as np
prior, th = 0.6, 0.5
best = None
for a in np.linspace(0, 1, 201):          # 危機ありのとき「平穏」と報告する確率
    for b in np.linspace(0, 1, 201):      # 危機なしのとき「平穏」と報告する確率
        p_calm = prior * a + (1 - prior) * b; p_alarm = 1 - p_calm
        post_calm = prior * a / p_calm if p_calm > 0 else 0.0
        post_alarm = prior * (1 - a) / p_alarm if p_alarm > 0 else 0.0
        act = (p_calm if post_calm >= th else 0) + (p_alarm if post_alarm >= th else 0)
        if best is None or act < best[0] - 1e-12: best = (act, a, b, post_calm, post_alarm, p_calm)
act, a, b, pc, pa, p_calm = best
if pc > pa:   # 「平穏」を、介入につながらない側の報告として表示する
    a, b, pc, pa, p_calm = 1 - a, 1 - b, pa, pc, 1 - p_calm
print(f"報告がなければ、中央は事前確率 {prior} で判断し、必ず介入する（介入の確率 1.00）")
print(f"正直な報告（状態をそのまま伝える）なら、介入の確率 = {prior:.2f}")
print(f"最適な見せ方：危機ありのとき確率 {a:.2f}、危機なしのとき確率 {b:.2f} で『平穏』と報告")
print(f"→ 介入の確率 = {act:.2f}。『平穏』を聞いた中央の事後確率 = {pc:.3f}（閾値 0.5 をわずかに下回るように設計されている）")
print(f"事後確率の平均 = {p_calm*pc + (1-p_calm)*pa:.2f}（事前確率と同じ。信念は平均では偏らないが、行動は大きく動く）")
