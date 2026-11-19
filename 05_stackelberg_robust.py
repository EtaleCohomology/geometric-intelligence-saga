"""05 小さなシュタッケルベルク・ゲームと、ロバストな選び方、順番の効果（第3章・第8章）。

守る側（凪浦）と、事実だけを選んで流す相手の、架空の小さなゲーム。
守る側の手：何もしない／業種ごとの数字を公表する／公表して同じ日に銀行と保険会社に説明する
相手の手（仮定した範囲）：決算の夜に選んだ事実を流す／業界について流す／銀行の担当者の目に触れる所に流す
仮定しなかった手（違法な手口、事実と違う情報）には、この計算は何も保証しない。
受け手の見立て（危うさの見込み）を状態とし、2つの手を状態に作用させる。
"""
import itertools
import numpy as np

true_risk = 0.30          # 本当の危うさ（架空）


def breach_prob(perceived):
    """受け手の見立てから、線に届く確率への架空の写し方（銀行と保険会社の反応を含む）。"""
    return 1.0 / (1.0 + np.exp(-(perceived - 0.40) / 0.06)) * 0.7 + 0.1


def disclose(state, explain=False):
    """全体を見せる：見立てを本当の危うさに近づけ、全体の見え方 κ を上げる。"""
    mu, kappa = state
    w = 0.8 if explain else 0.6
    return (w * true_risk + (1 - w) * mu, max(kappa, 0.95 if explain else 0.85))


def post(state, strength):
    """事実だけを選んで流す：全体が見えていない分だけ、見立てを悪い側にずらす。"""
    mu, kappa = state
    return (mu + strength * (1 - kappa), kappa)


start = (0.32, 0.10)      # 公表の前：受け手は全体をほとんど見ていない
strength = {"決算の夜に流す": 0.30, "業界について流す": 0.22, "銀行の担当者に流す": 0.26}

# --- 順番の効果（リー括弧が 0 でないこと） ---
s = strength["決算の夜に流す"]
before = post(disclose(start, True), s)     # 公表と説明を先に、投稿があと
after = disclose(post(start, s), True)      # 投稿が先に、公表と説明があと
print("公表と説明を、投稿の前に行う:", f"{breach_prob(before[0]):.0%}")
print("公表と説明を、投稿の後に行う:", f"{breach_prob(after[0]):.0%}")
print("2つの順番の、行き着く先の見立ての差（交換子にあたる量）:", round(before[0] - after[0], 3))

# --- ロバストな選び方（仮定した範囲の中の最悪の場合を、いちばん小さくする） ---
defs = {"何もしない": lambda st: st,
        "公表する": lambda st: disclose(st, False),
        "公表して、同じ日に説明する": lambda st: disclose(st, True)}
print("\n守る側が先に手を決めて約束する場合（シュタッケルベルク・ゲームの先手）")
table = {}
for dn, d in defs.items():
    worst = max(breach_prob(post(d(start), v)[0]) for v in strength.values())
    typical = breach_prob(d(start)[0])
    table[dn] = worst
    print(f"  {dn:16s} ふつうの場合 {typical:4.0%}  相手が範囲の中で最も悪い手を打つ場合 {worst:4.0%}")
best = min(table, key=table.get)
print("  → 最悪の場合をいちばん小さくする手:", best)

# 先手を取らない（相手が先に打ち、守る側があとから答える）場合
late = {dn: max(breach_prob(d(post(start, v))[0]) for v in strength.values()) for dn, d in defs.items()}
print("\n相手が先に打ち、守る側があとから答える場合の最悪の場合")
for dn, v in late.items():
    print(f"  {dn:16s} {v:4.0%}")
print("注意：数字はすべて架空。この計算が守れるのは、仮定した相手の手の範囲の中だけである。")
