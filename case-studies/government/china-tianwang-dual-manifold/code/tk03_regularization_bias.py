"""
敵の多様体 補助文書03 コード3：正則化が曲率に入れる偏り
球面の計量 ds² = du² + sin²u dv²（曲率はどこでも 1）は、u → 0 で sin²u → 0 となり、計量がつぶれかける。
計量に小さな正の値 ε を足して ds² = du² + (sin²u + ε) dv² とすると、数値は安定するが、曲率に偏りが入る。
F=0 の２次元の計量 E=1, G=G(u) のガウス曲率は K = −(√G)'' / √G で計算できる。
"""
import numpy as np
u = np.array([0.05, 0.1, 0.3, 0.6, 1.0, 1.5])
h = 1e-4
def K(u, eps):
    g = lambda x: np.sqrt(np.sin(x) ** 2 + eps)
    d2 = (g(u + h) - 2 * g(u) + g(u - h)) / h ** 2
    return -d2 / g(u)
print("u（極からの角度）:", u)
for eps in [0.0, 1e-4, 1e-2]:
    print(f"ε={eps:<7} 曲率 K:", np.round(K(u, eps), 3))
print("正しい曲率はどこでも 1。ε を足すと、計量がつぶれかけていた極の近くほど、曲率が大きく偏る。")
