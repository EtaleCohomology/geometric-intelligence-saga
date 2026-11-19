"""03 有限期間の可制御性・可観測性のグラミアンと、経験グラミアン（第3章・第5章）。

架空の、4つの状態を持つ月次の模型。
  状態 x：0 回収の遅れ（月）、1 大口の顧客への依存、2 新しい取引先の残高、3 純資産の余裕
  入力 u：外から加える小さな力（取引条件の変更、契約の付け替え、取引の拡大）
  観測 y：公開の決算書から見える数字（四半期の合計の数字だけ）
非線形の模型を、JAX の自動微分で現在地のまわりの線形の近似（ヤコビ行列）に置き換えて、
期間 N=24 か月のグラミアンを計算する。次に、近似を使わずに、元の模型を何度も動かして
経験グラミアンを作り、2つの方法の答えが近いかを確かめる。
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
N = 24

# 状態は、すべて「ふだんの大きさ」で割って、単位と桁をそろえた座標で書く。
# そろえないと、どの方向が動かしやすいかの比較が、勘定科目の単位に引きずられる。


def step(x, u):
    """1か月ぶんの、架空の非線形の模型（スケールをそろえた座標で書く）。"""
    A = jnp.array([[0.95, 0.00, 0.00, 0.0],
                   [0.00, 0.97, 0.00, 0.0],
                   [0.00, 0.00, 0.96, 0.0],
                   [-0.01, -0.01, -0.01, 0.9]])
    B = jnp.array([[0.30, 0.00, 0.00],
                   [0.00, 0.30, 0.00],
                   [0.00, 0.00, 0.30],
                   [0.00, 0.00, 0.00]])
    return A @ x + B @ u + 0.02 * jnp.tanh(x) ** 3


def observe(x):
    """公開の四半期の決算書から見える数字：売掛金の合計と純資産だけ（明細は見えない）。"""
    Cm = jnp.array([[0.05, 0.0, 0.05, 0.0],    # 回収の遅れと新しい取引先は、合計の中に薄く埋もれる
                    [0.0, 0.02, 0.0, 1.0]])    # 大口依存は、開示の線（10％）の下では、ほとんど映らない
    return Cm @ x


x_now = jnp.zeros(4)
u_now = jnp.zeros(3)
A_lin = jax.jacfwd(step, argnums=0)(x_now, u_now)
B_lin = jax.jacfwd(step, argnums=1)(x_now, u_now)
C_lin = jax.jacfwd(observe)(x_now)


def finite_gramians(A, B, C, N):
    Wc = jnp.zeros((4, 4)); Wo = jnp.zeros((4, 4))
    Ak = jnp.eye(4)
    for _ in range(N):
        Wc = Wc + Ak @ B @ B.T @ Ak.T
        Wo = Wo + Ak.T @ C.T @ C @ Ak
        Ak = A @ Ak
    return Wc, Wo


Wc, Wo = finite_gramians(A_lin, B_lin, C_lin, N)

# 経験グラミアン（Lall, Marsden, Glavaški, 2002 の考え方を簡単にしたもの）
def empirical_wc(eps=0.1):
    Wc = np.zeros((4, 4))
    for j in range(3):
        for sgn in (+1, -1):
            x = jnp.zeros(4)
            traj = []
            u = jnp.zeros(3).at[j].set(sgn * eps)
            x = step(x, u)                         # 最初の月にだけ力を加える
            for _ in range(N):
                traj.append(np.array(x))
                x = step(x, jnp.zeros(3))
            X = np.stack(traj)
            Wc += X.T @ X / (2 * eps**2)
    return Wc

Wc_emp = empirical_wc(eps=1.0)   # 大きめの力で、非線形の部分の影響も含める

names = ["回収の遅れ", "大口の顧客への依存", "新しい取引先の残高", "純資産の余裕"]
print("方向ごとの、動かしやすさ（可制御性）と、見えやすさ（可観測性）  期間 24 か月")
print(f"{'方向':16s} {'動かしやすさ':>10s} {'経験グラミアン':>12s} {'見えやすさ':>10s}")
for i, n in enumerate(names):
    print(f"{n:16s} {float(Wc[i,i]):10.3f} {Wc_emp[i,i]:12.3f} {float(Wo[i,i]):10.4f}")

# 線形の近似と、経験グラミアンの、いちばん動かしやすい方向の向きの近さ
v1 = np.linalg.eigh(np.array(Wc))[1][:, -1]
v2 = np.linalg.eigh(Wc_emp)[1][:, -1]
print("\n2つの方法の、最も動かしやすい方向のなす角の余弦:", round(abs(float(v1 @ v2)), 4))
print("（1 に近ければ、現在地のまわりでは、線形の近似が大きく外れていない）")

ratio = np.diag(np.array(Wc)) / (np.diag(np.array(Wo)) + 1e-12)
print("\n『動かしやすいのに見えにくい』順（動かしやすさ ÷ 見えやすさ）")
for i in np.argsort(-ratio):
    print(f"  {names[i]:16s} {ratio[i]:10.1f}")
print("注意：この結論は、現在地のまわりの、24か月という有限の期間での、局所の結論である。")
