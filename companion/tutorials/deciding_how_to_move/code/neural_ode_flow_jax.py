"""連載第3回 実験3：Neural ODE で「流れ」を学び直す（JAX）
答えの流れ：地形 h(x, y) = (x² − 0.5)² + y² を下る流れ v = −∇h（谷底は (±0.707, 0)、間に峠 (0, 0)）。
この流れに沿った軌跡のデータだけから、ニューラルネットワークの流れ f_θ を学び、
新しい出発点から、学んだ流れで谷底にたどり着けるかを確かめる。"""
import numpy as np, jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
h = lambda p: (p[0]**2 - 0.5)**2 + p[1]**2
v_true = lambda p: -jax.grad(h)(p)
def rk4(f, p, dt, n):                     # ルンゲ・クッタ法で、流れに沿って進む
    out = [p]
    for _ in range(n):
        k1 = f(p); k2 = f(p + dt/2*k1); k3 = f(p + dt/2*k2); k4 = f(p + dt*k3)
        p = p + dt/6*(k1 + 2*k2 + 2*k3 + k4); out.append(p)
    return jnp.stack(out)
rng = np.random.default_rng(0)
starts = jnp.array(rng.uniform([-1.2, -0.8], [1.2, 0.8], (64, 2)))
dt, n = 0.05, 20
trajs = jax.vmap(lambda p: rk4(v_true, p, dt, n))(starts)            # 学習用の軌跡（64本 × 21時点）
def init(key, sizes):
    ps = []
    for m, k_ in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (m, k_))*np.sqrt(1/m), jnp.zeros(k_)))
    return ps
def mlp(ps, x):
    for W, b in ps[:-1]: x = jnp.tanh(x @ W + b)
    W, b = ps[-1]; return x @ W + b
params = init(jax.random.PRNGKey(0), [2, 64, 64, 2])
def loss(ps):                              # 学んだ流れで同じ出発点から進め、軌跡全体のずれを小さくする
    pred = jax.vmap(lambda p: rk4(lambda q: mlp(ps, q), p, dt, n))(starts)
    return jnp.mean((pred - trajs)**2)
opt = optax.adam(3e-3); st = opt.init(params)
@jax.jit
def step(ps, s):
    l, g = jax.value_and_grad(loss)(ps); u, s = opt.update(g, s); return optax.apply_updates(ps, u), s, l
for i in range(1500):
    params, st, l = step(params, st)
print(f"学習の終わりの、軌跡のずれ（平均二乗誤差） = {float(l):.2e}")
# 新しい出発点から、長い時間（学習の5倍）進めて、どこにたどり着くか
tests = jnp.array([[-1.0, 0.6], [1.1, -0.5], [-0.2, 0.7], [0.3, -0.6]])
for p0 in tests:
    end_true = rk4(v_true, p0, dt, 100)[-1]
    end_learn = rk4(lambda q: mlp(params, q), p0, dt, 100)[-1]
    print(f"出発点 ({float(p0[0]):+.1f}, {float(p0[1]):+.1f})  答えの流れの行き先 = ({float(end_true[0]):+.3f}, {float(end_true[1]):+.3f})"
          f"  学んだ流れの行き先 = ({float(end_learn[0]):+.3f}, {float(end_learn[1]):+.3f})")
# 学んだ流れの「止まる点」（流れがゼロの点）と、その種類
for z0 in [jnp.array([0.6, 0.1]), jnp.array([-0.6, -0.1]), jnp.array([0.05, 0.05])]:
    z = z0
    for _ in range(50):
        Jf = jax.jacfwd(lambda q: mlp(params, q))(z); z = z - jnp.linalg.solve(Jf, mlp(params, z))
    ev = np.linalg.eigvals(np.array(jax.jacfwd(lambda q: mlp(params, q))(z)))
    kind = "谷底（すべての向きから流れ込む）" if np.all(ev.real < 0) else "峠（流れ込む向きと、流れ出る向きがある）"
    print(f"学んだ流れが止まる点 ({float(z[0]):+.3f}, {float(z[1]):+.3f})：{kind}")
