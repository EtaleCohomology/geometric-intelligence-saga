"""01 会計の恒等式を課したデコーダの出力を、正しい決算書の領域へ射影する（第2章・第4章）。

数字はすべて架空。単位は億円。
1. 等式（資産 − 負債 − 純資産 = 0）だけの平面へ、内積 W で射影する。
   内積の選び方で、純資産の直し方が変わり、財務制限条項の線（392億円）の上にも下にもなる。
2. 負になれない勘定科目の条件を加えた、角のある領域（凸多面体）へ、Dykstra の交互射影で射影する。
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---- 1. 等式の平面への射影 ------------------------------------------------
x0 = jnp.array([1000.0, 600.0, 390.0])      # デコーダの出力：資産、負債、純資産
a = jnp.array([1.0, -1.0, -1.0])            # 資産 − 負債 − 純資産 = 0


def project_plane(x, a, w):
    """内積 <u, v>_W = Σ w_i u_i v_i で、平面 a·x = 0 にまっすぐ落とす。"""
    winv = 1.0 / w
    return x - winv * a * (a @ x) / (a @ (winv * a))


line = 392.0                                # 架空の純資産維持条項の線
w_equal = jnp.ones(3)                       # 3つを同じ重さで扱う内積
s = jnp.array([1000.0, 600.0, 400.0])       # ふだんの大きさ
w_scaled = 1.0 / s**2                       # 誤差が大きさに比例すると仮定した内積

for name, w in [("同じ重さ", w_equal), ("大きさで重さを変える", w_scaled)]:
    x = project_plane(x0, a, w)
    side = "線の上" if x[2] >= line else "線の下"
    print(f"{name:12s} 資産 {x[0]:8.2f} 負債 {x[1]:7.2f} 純資産 {x[2]:7.2f} "
          f"等式のずれ {float(a @ x):+.1e}  {side}")

# ---- 2. 角のある領域（凸多面体）への射影 -----------------------------------
# 勘定科目：現金、在庫、負債、純資産。決まり：現金 + 在庫 − 負債 − 純資産 = 0
# 現金、在庫、負債は 0 以上。純資産は負になりうる（債務超過）ので条件を付けない。
b = jnp.array([1.0, 1.0, -1.0, -1.0])
nonneg = jnp.array([True, True, True, False])


def project_box(x):
    # 対角の重さの内積なら、0 以上への射影は、その成分の切り捨てでよい
    return jnp.where(nonneg, jnp.maximum(x, 0.0), x)


@jax.jit
def project_region(x, w):
    """Dykstra の交互射影：平面と、0 以上の条件の共通部分への射影。"""
    def body(_, state):
        y, p, q = state
        z = project_plane(y + p, b, w)
        p = y + p - z
        y_new = project_box(z + q)
        q = z + q - y_new
        return (y_new, p, q)
    y, _, _ = jax.lax.fori_loop(0, 2000, body, (x, jnp.zeros_like(x), jnp.zeros_like(x)))
    return y


x_bad = jnp.array([-6.0, 120.0, 80.0, 30.0])   # 現金が負で、等式も 4 ずれている
w4 = jnp.ones(4)
x_plane = project_plane(x_bad, b, w4)
x_reg = project_region(x_bad, w4)
print("等式だけ    ", jnp.round(x_plane, 2), "ずれ", round(float(b @ x_plane), 6))
print("角のある領域", jnp.round(x_reg, 2), "ずれ", round(float(b @ x_reg), 6))
assert abs(float(b @ x_reg)) < 1e-8 and float(x_reg[0]) >= -1e-9
