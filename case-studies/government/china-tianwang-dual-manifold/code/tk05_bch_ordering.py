"""
敵の多様体 補助文書03 コード5：政策の順序の効果 ── リー括弧と、三重のリー括弧（BCH の公式）
３つの政策を、小さな行列 X, Y, Z の指数で表し、続けて実行した結果 exp(X)exp(Y)exp(Z) を、
log をとった「まとめた効果」として計算する。
近似を、(1) 和だけ、(2) リー括弧（２次）まで、(3) 三重のリー括弧（３次）まで、と順に上げ、誤差を比べる。
"""
import numpy as np
from scipy.linalg import expm, logm
rng = np.random.default_rng(31)
s = 0.15
X, Y, Z = (s * rng.standard_normal((4, 4)) for _ in range(3))
br = lambda a, b: a @ b - b @ a
exact = logm(expm(X) @ expm(Y) @ expm(Z)).real
# BCH を２回使う：log(e^X e^Y) を３次まで、その結果と Z を、さらに３次まで
def bch3(A, B):
    return A + B + br(A, B) / 2 + (br(A, br(A, B)) + br(B, br(B, A))) / 12
o1 = X + Y + Z
W2 = X + Y + br(X, Y) / 2
o2 = W2 + Z + br(W2, Z) / 2
o3 = bch3(bch3(X, Y), Z)
for name, approx in [("和だけ（順序を無視）", o1), ("リー括弧（２次）まで", o2), ("三重のリー括弧（３次）まで", o3)]:
    print(f"{name}: 誤差 {np.linalg.norm(exact - approx):.2e}")
print("順序の入れ替え：exp(X)exp(Y)exp(Z) と exp(Z)exp(Y)exp(X) の差", f"{np.linalg.norm(expm(X)@expm(Y)@expm(Z) - expm(Z)@expm(Y)@expm(X)):.2e}")
