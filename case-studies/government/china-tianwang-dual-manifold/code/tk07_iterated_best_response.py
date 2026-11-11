"""
敵の多様体 補助文書03 コード7：反復ベストレスポンスと、均衡の安定性（ヘッセ行列の固有値）
２人のプレイヤーの戦略 x, y（各2次元）。利得は２次関数で、相手の戦略に依存する。
最適応答 BR_x(y) = A y + a、BR_y(x) = B x + b。交互に応答を繰り返し、変化量が 1e-4 を下回ったら収束とみなす。
写像のスペクトル半径が 1 未満（縮小写像）なら収束し、1 を超えると発散・振動する。数値は架空のもの。
"""
import numpy as np
def run(A, B, a, b, tol=1e-4, maxit=500):
    x, y = np.zeros(2), np.zeros(2)
    for it in range(1, maxit + 1):
        x_new = A @ y + a
        y_new = B @ x_new + b
        step = np.linalg.norm(np.r_[x_new - x, y_new - y]); x, y = x_new, y_new
        if step < tol: return it, x, y, True
    return maxit, x, y, False
a, b = np.array([1.0, 0.5]), np.array([-0.5, 1.0])
for label, s in [("縮小写像になる場合", 0.6), ("縮小写像にならない場合", 1.4)]:
    A = s * np.array([[0.8, -0.3], [0.2, 0.7]]); B = s * np.array([[0.6, 0.4], [-0.5, 0.9]])
    rho = max(abs(np.linalg.eigvals(B @ A)))
    it, x, y, ok = run(A, B, a, b)
    print(f"{label}: 合成写像のスペクトル半径 {rho:.3f} → {'収束' if ok else '収束せず'}（反復 {it} 回）")
# 均衡点の安定性：利得関数のヘッセ行列の固有値
H = np.array([[-2.0, 0.3], [0.3, 0.8]])   # 均衡点での、ある政策の組の利得のヘッセ行列（架空）
ev = np.linalg.eigvalsh(H)
kind = "安定（すべて負）" if all(ev < 0) else ("鞍点型（正と負が混在）" if any(ev > 0) and any(ev < 0) else "不安定")
print(f"ヘッセ行列の固有値 {np.round(ev, 3)} → {kind}")
print("→ 鞍点型の均衡は、ある方向の小さな揺れには戻るが、別の方向の揺れには崩れうる（本編の台湾海峡の均衡）。")
