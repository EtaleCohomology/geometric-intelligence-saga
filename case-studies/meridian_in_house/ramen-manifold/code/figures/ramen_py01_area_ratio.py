"""第３章：斜めに写った丼の縁の楕円と、チャーシューの面積の比。

- 縁の円とスープの面のチャーシューは、平行な２つの平面の上にある。
- アフィン変換（遠近を無視した近似）では、平行な平面の上の図形の面積の比は変わらない。
- 本物のカメラは射影変換（ホモグラフィー）なので、遠近の分だけ、比がずれる。
- 深さの違いと遠近のせいで、面積の比そのものは、本当の値からずれる。
- けれど、同じ距離と角度で撮った写真どうしなら、ずれはほぼ同じ割合で掛かるので、
  「前と比べて何％減ったか」は、本当の値に近く出る。

数値は、物語の中の架空の値である。
"""
import numpy as np
import matplotlib.pyplot as plt
import jpfont

R_RIM = 10.0      # 丼の縁の半径 [cm]
R_CHASHU = 4.0    # チャーシューを円で近似したときの半径 [cm]
DEPTH = 3.0       # 縁の平面から、スープの面までの深さ [cm]
OFFSET = np.array([2.0, 1.0])  # チャーシューの中心の、丼の中心からのずれ [cm]


def circle(center, r, z, n=2000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t), np.full(n, z)], 1)


def project(X, tilt_deg, dist, perspective=True):
    """丼の中心を、距離 dist の位置から、真上から tilt_deg だけ傾けて撮る。"""
    a = np.deg2rad(tilt_deg)
    # カメラの座標へ：x 軸まわりに回し、カメラを原点から dist 離す
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Y = X @ Rx.T
    Z = dist - Y[:, 2]  # カメラまでの奥行き
    if perspective:
        return np.stack([Y[:, 0] / Z, Y[:, 1] / Z], 1)
    return np.stack([Y[:, 0] / dist, Y[:, 1] / dist], 1)  # 遠近を無視した近似（アフィン）


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def axis_ratio(P):
    C = np.cov((P - P.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(C))
    return np.sqrt(ev[0] / ev[1])


def main():
    jpfont.setup()
    true_ratio = (R_CHASHU / R_RIM) ** 2
    rim = circle([0, 0], R_RIM, 0.0)
    ch = circle(OFFSET, R_CHASHU, -DEPTH)

    # 検算１：アフィン近似では、どの角度でも比は変わらない
    for t in (0, 30, 60):
        r_aff = area(project(ch, t, 40, False)) / area(project(rim, t, 40, False))
        assert abs(r_aff / true_ratio - 1) < 1e-4, r_aff
    print("検算１：アフィン近似では、面積の比は角度によらず一定 ->", round(true_ratio, 4))

    # 検算２：射影変換の自由度は 3x3 行列の 9 個から、全体の定数倍の 1 個を引いた 8
    print("検算２：ホモグラフィーの自由度 = 9 - 1 = 8、楕円の手がかり = 5、アフィンの自由度 = 6")

    tilts = np.linspace(0, 70, 71)
    shrink = 0.95  # チャーシューの面積を 5% 減らした丼
    ch_small = circle(OFFSET, R_CHASHU * np.sqrt(shrink), -DEPTH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for dist, col in ((35, "#d9822b"), (60, "#2b7bb9")):
        ar, err, dec = [], [], []
        for t in tilts:
            pr = project(rim, t, dist)
            m0 = area(project(ch, t, dist)) / area(pr)
            m1 = area(project(ch_small, t, dist)) / area(pr)
            ar.append(axis_ratio(pr))
            err.append((m0 / true_ratio - 1) * 100)
            dec.append((1 - m1 / m0) * 100)
        axes[0].plot(ar, err, color=col, lw=2, label=f"カメラまで {dist} cm")
        axes[1].plot(ar, dec, color=col, lw=2, label=f"カメラまで {dist} cm")
        print(f"距離 {dist} cm：面積の比そのもののずれ 真上 {err[0]:+.2f}%、"
              f"短軸/長軸 0.5 で {np.interp(0.5, ar[::-1], err[::-1]):+.2f}%")
        print(f"            5% 減らした丼との比べ：推定した減り方 {min(dec):.3f}〜{max(dec):.3f}%")
    for ax in axes:
        ax.axvline(0.5, color="gray", ls="--")
        ax.invert_xaxis()
        ax.set_xlabel("縁の楕円の、短軸と長軸の比（1 = 真上から）")
        ax.legend(loc="lower left")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("面積の比の、本当の値からのずれ [%]")
    axes[0].set_title("面積の比そのもの：深さと遠近の分だけ、ずれる")
    axes[1].axhline(5, color="black", lw=0.8)
    axes[1].set_ylim(4, 6)
    axes[1].set_ylabel("推定した、チャーシューの面積の減り方 [%]")
    axes[1].set_title("同じ撮り方どうしの比べ：本当の 5% に近い")
    jpfont.save(fig, "ramen_py01_area_ratio.jpg")


if __name__ == "__main__":
    main()
