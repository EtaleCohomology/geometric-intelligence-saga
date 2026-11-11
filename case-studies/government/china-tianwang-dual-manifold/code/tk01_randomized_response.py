"""
敵の多様体 補助文書03 コード1：ランダム化回答法（Warner 型の一種、強制回答方式）
2,850人の地方の幹部が、「公式の統計に載っていない暴動や債務があるか」に答える。
硬貨の表（確率 p）なら正直に答え、裏なら必ず「はい」と答える。
「はい」の割合 λ から、本当の割合 π を π_hat = (λ − (1 − p)) / p で推定する。数値は架空のもの。
"""
import numpy as np
rng = np.random.default_rng(2033)
N, pi_true, p = 2850, 0.30, 0.5
truth = rng.random(N) < pi_true            # 本当の答え（集める側には見えない）
heads = rng.random(N) < p                   # 硬貨（集める側には見えない）
answer = np.where(heads, truth, True)       # 表なら正直、裏なら「はい」
lam = answer.mean()
pi_hat = (lam - (1 - p)) / p
se = np.sqrt(lam * (1 - lam) / N) / p       # 推定の標準誤差
# 「はい」と答えた人が、本当に「はい」である確率（１人１人の秘密がどれだけ守られるか）
post = (pi_true * 1) / (pi_true + (1 - pi_true) * (1 - p))
print(f"本当の割合 {pi_true:.3f}  推定 {pi_hat:.3f}  95%区間 [{pi_hat-1.96*se:.3f}, {pi_hat+1.96*se:.3f}]")
print(f"「はい」と答えた人が本当に「はい」である確率 {post:.3f}（１人の答えからは断定できない）")
print(f"直接に尋ねた場合と比べた標準誤差の倍率 {se/np.sqrt(pi_true*(1-pi_true)/N):.2f}（秘密を守る代わりに、推定の精度は下がる）")
