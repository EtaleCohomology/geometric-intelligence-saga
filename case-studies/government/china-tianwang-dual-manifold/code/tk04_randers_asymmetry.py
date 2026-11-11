"""
敵の多様体 補助文書03 コード4：ランダース計量 ── 行きと帰りで長さの違う定規
F(v) = |v| + b·v（|b| < 1）。背景が平らで b が一定なら、２点を結ぶ最短の道は直線で、
A→B の長さは |AB| + b·(B−A)、B→A の長さは |AB| − b·(B−A) になる。
b を「緊張を高める方向」と逆向きに置くと、緊張を高める道は短く（易しく）、元に戻す道は長く（難しく）なる。数値は架空のもの。
"""
import numpy as np
A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0])   # A：平時、B：緊張の高い状態
for bx in [0.0, -0.3, -0.6]:
    b = np.array([bx, 0.0])
    assert np.linalg.norm(b) < 1, "ランダース計量になるには |b| < 1 が必要"
    d_ab = np.linalg.norm(B - A) + b @ (B - A)
    d_ba = np.linalg.norm(A - B) + b @ (A - B)
    print(f"b={bx:+.1f}: 緊張を高める（A→B）{d_ab:.2f}   緊張を下げる（B→A）{d_ba:.2f}   比 {d_ba/d_ab:.2f}")
print("b=0 ならリーマン計量（行きと帰りは同じ）。|b| が大きいほど、行きと帰りの違いが大きい。")
