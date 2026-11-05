"""連載第3回 実験2：縦列駐車とLie括弧（JAX）
状態 q = (x, y, θ)。前進の流れ f = (cosθ, sinθ, 0)、回転の流れ g = (0, 0, 1)。
「前進 → 回転 → 後退 → 逆回転」を、それぞれ時間 ε だけ行うと、
正味の動きは ε^2 × [f, g] に近づく（[f, g] は真横の向き）。"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
f = lambda q: jnp.array([jnp.cos(q[2]), jnp.sin(q[2]), 0.0])   # 前進
g = lambda q: jnp.array([0.0, 0.0, 1.0])                        # その場で回転
def flow(v, q, t, n=2000):                                      # 流れに沿って時間 t だけ進む（ルンゲ・クッタ法）
    h = t / n
    for _ in range(n):
        k1 = v(q); k2 = v(q + h/2*k1); k3 = v(q + h/2*k2); k4 = v(q + h*k3)
        q = q + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return q
def lie_bracket(f, g, q):                                       # [f, g] = (∂g/∂q) f − (∂f/∂q) g
    return jax.jacfwd(g)(q) @ f(q) - jax.jacfwd(f)(q) @ g(q)
q0 = jnp.array([0.0, 0.0, 0.0])
print("Lie括弧 [f, g](q0) =", lie_bracket(f, g, q0))
for eps in [0.2, 0.1, 0.05]:
    q = flow(f, q0, eps); q = flow(g, q, eps); q = flow(f, q, -eps); q = flow(g, q, -eps)
    d = q - q0
    print(f"ε={eps:<5} 正味の動き = ({d[0]:+.6f}, {d[1]:+.6f}, {d[2]:+.6f})   動き/ε^2 = ({d[0]/eps**2:+.4f}, {d[1]/eps**2:+.4f}, {d[2]/eps**2:+.4f})")
