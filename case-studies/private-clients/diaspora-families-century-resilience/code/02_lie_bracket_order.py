"""02 出来事の順番の効果を、リー括弧で測る（本編第3章「靴下と靴」）。

状態は２つの座標（すべて架空）。
  x = 暮らしの厳しさ、 y = 少数派の置かれた立場の危うさ
２つの出来事を、地図の上の流れ（ベクトル場）として表す。
  A「暮らしが悪くなる」  ：F(x, y) = (1, 0)
  B「政権が揺らぐ」      ：G(x, y) = (0, x)  … 揺らぎが少数派の立場に及ぼす大きさが、そのときの暮らしの厳しさに比例する
A を時間 t だけ進めてから B を t だけ進めた行き先と、順番を入れ替えた行き先のずれは、
t が小さいとき t^2 [F, G] に近い。[F, G] = DG·F − DF·G がリー括弧（D はヤコビ行列）。
1. 上の簡単な例で、ずれが t^2 [F, G] に一致することを確かめる
2. 非線形のベクトル場で、ずれ ÷ t^2 が、t → 0 でリー括弧に近づくことを数値で確かめる
3. リー括弧が 0 になる例（順番に意味がない例）も確かめる
"""
import numpy as np
from scipy.integrate import solve_ivp

def flow(field, p, t):
    """ベクトル場 field に沿って、点 p から時間 t だけ進めた行き先。"""
    sol = solve_ivp(lambda _, z: field(z), (0, t), p, rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]

def jac(field, p, h=1e-6):
    p = np.asarray(p, float)
    J = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = h
        J[:, k] = (field(p + e) - field(p - e)) / (2 * h)
    return J

def bracket(F, G, p):
    return jac(G, p) @ F(p) - jac(F, p) @ G(p)

def gap(F, G, p, t):
    ab = flow(G, flow(F, p, t), t)      # A のあとに B
    ba = flow(F, flow(G, p, t), t)      # B のあとに A
    return ab - ba

p0 = np.array([0.0, 0.0])

# ---- 1. 簡単な例 ---------------------------------------------------------
F = lambda z: np.array([1.0, 0.0])
G = lambda z: np.array([0.0, z[0]])
print("1. 簡単な例  [F, G] =", bracket(F, G, p0))
for t in [0.5, 0.2, 0.1]:
    d = gap(F, G, p0, t)
    print(f"   t = {t:4.2f}  ずれ（A→B と B→A の差） = ({d[0]:+.6f}, {d[1]:+.6f})   t^2 [F, G] = (0, {t * t:.6f})")

# ---- 2. 非線形の例 -------------------------------------------------------
F2 = lambda z: np.array([1.0 + 0.3 * z[1] ** 2, 0.2 * np.sin(z[0])])
G2 = lambda z: np.array([0.1 * z[1], 0.5 * z[0] + 0.4 * z[0] * z[1]])
p1 = np.array([0.4, 0.3])
b = bracket(F2, G2, p1)
print("\n2. 非線形の例  点 (0.4, 0.3) でのリー括弧 =", np.round(b, 6))
for t in [0.2, 0.1, 0.05, 0.02]:
    d = gap(F2, G2, p1, t)
    print(f"   t = {t:4.2f}  ずれ ÷ t^2 = ({d[0] / t**2:+.6f}, {d[1] / t**2:+.6f})")

# ---- 3. 順番に意味がない例 ------------------------------------------------
F3 = lambda z: np.array([1.0, 0.0])
G3 = lambda z: np.array([0.0, 1.0])
print("\n3. 互いに無関係な２つの出来事  [F, G] =", bracket(F3, G3, p0), " ずれ =", np.round(gap(F3, G3, p0, 0.3), 10))
print("\n注意：リー括弧が 0 でないことは、順番に意味があるという「形の上の事実」である。")
print("      どの順番が少数派にとって危ういかは、F と G を、どのデータで、どう推定したかに依存する。")
