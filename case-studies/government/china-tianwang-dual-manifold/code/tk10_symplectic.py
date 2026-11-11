"""
敵の多様体 補助文書03 コード10：シンプレクティック積分 ── 長期の計算で、エネルギーが崩れないこと
単振動 H = (p² + q²)/2 を、オイラー法とストルメール＝ヴェルレ法（シンプレクティック積分）で 20,000 ステップ計算し、
エネルギー H のずれを比べる。シンプレクティック積分が保証するのは「計算が崩れないこと」で、「予測が当たること」ではない。
"""
import numpy as np
h, N = 0.05, 20000
def euler(q, p):
    for _ in range(N): q, p = q + h * p, p - h * q
    return q, p
def verlet(q, p):
    for _ in range(N):
        p = p - 0.5 * h * q; q = q + h * p; p = p - 0.5 * h * q
    return q, p
H0 = 0.5
for name, f in [("オイラー法", euler), ("ストルメール＝ヴェルレ法", verlet)]:
    q, p = f(1.0, 0.0); H = 0.5 * (p * p + q * q)
    print(f"{name}: 最初のエネルギー {H0:.3f} → {N} ステップ後 {H:.4g}（比 {H/H0:.4g}）")
