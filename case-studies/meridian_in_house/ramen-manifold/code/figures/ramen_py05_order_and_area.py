"""第６章・第７章：順番で答えが変わる部分。混合差異、5.26%、レヴィ面積。

1. 原価計算の差異分析（第７章のナイドゥさんの計算）
   どちらを先に入れ替えて数えるかで、価格差異と数量差異の大きさが変わる。
   その差が、両方の変化の掛け算（混合差異）である。
2. 量を 5% 減らして値段を据え置くと、1 グラムあたりの値上げは 5.26%。
3. ２つの量の道が囲む、向きつきの面積（レヴィ面積）。
   「右へ 1、上へ 1」と「上へ 1、右へ 1」で、+1/2 と -1/2。
   シグネチャの２段目と、チェンの恒等式で検算する。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jpfont


def variance_analysis():
    p0, p1 = 7.00, 8.40      # 1kg あたりの標準の値段と実際の値段 [ポンド]
    q0, q1 = 60.0, 57.0      # 1 週間の標準の量と実際の量 [kg]
    dp, dq = p1 - p0, q1 - q0
    total = p1 * q1 - p0 * q0
    a = ((p1 - p0) * q1, (q1 - q0) * p0)   # 値段を先に（決まりの式）
    b = ((p1 - p0) * q0, (q1 - q0) * p1)   # 量を先に
    three = (dp * q0, dq * p0, dp * dq)     # 値段だけ、量だけ、混合差異
    print(f"合計 {total:+.2f}")
    print(f"値段が先：価格差異 {a[0]:+.2f}、数量差異 {a[1]:+.2f}")
    print(f"量が先　：価格差異 {b[0]:+.2f}、数量差異 {b[1]:+.2f}")
    print(f"３つに分ける：値段だけ {three[0]:+.2f}、量だけ {three[1]:+.2f}、混合差異 {three[2]:+.2f}")
    for parts in (a, b, three):
        assert abs(sum(parts) - total) < 1e-9
    assert abs((a[0] - b[0]) - dp * dq) < 1e-9
    return p0, p1, q0, q1


def per_gram():
    price, g0, g1 = 16.50, 60, 57
    up = (price / g1) / (price / g0) - 1
    print(f"1 グラムあたりの値上げ = 60/57 - 1 = {up*100:.2f}%")
    assert abs(up - (1 / 0.95 - 1)) < 1e-12


def signature_level2(path):
    """折れ線の道のシグネチャを、２段目まで計算する（チェンの恒等式で区間をつなぐ）。"""
    d = path.shape[1]
    S1 = np.zeros(d); S2 = np.zeros((d, d))
    for k in range(len(path) - 1):
        inc = path[k + 1] - path[k]
        seg2 = np.outer(inc, inc) / 2   # 直線の区間のシグネチャの２段目
        S2 = S2 + np.outer(S1, inc) + seg2   # チェンの恒等式：S(X*Y) = S(X) ⊗ S(Y)
        S1 = S1 + inc
    levy = (S2[0, 1] - S2[1, 0]) / 2
    return S1, S2, levy


def main():
    jpfont.setup()
    p0, p1, q0, q1 = variance_analysis()
    per_gram()
    right_up = np.array([[0, 0], [1, 0], [1, 1]], float)
    up_right = np.array([[0, 0], [0, 1], [1, 1]], float)
    for name, P in (("右へ、それから上へ", right_up), ("上へ、それから右へ", up_right)):
        S1, S2, levy = signature_level2(P)
        print(f"{name}：全体の増え方 {S1}、レヴィ面積 {levy:+.2f}")
    assert np.isclose(signature_level2(right_up)[2], 0.5)
    assert np.isclose(signature_level2(up_right)[2], -0.5)

    # 値上がりが先で、量の減り方があと、という道（第６章の図の考え方）
    s = np.linspace(0, 1, 200)
    price = s
    shrink = s ** 4
    _, _, levy = signature_level2(np.column_stack([price, shrink]))
    print(f"値上がりが先、量の減り方があとの道：レヴィ面積 {levy:+.3f}（正なら、値段が先に動いた）")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), q1, p0, fc="#cfe3f2", ec="#2b7bb9"))
    ax.add_patch(Rectangle((0, p0), q1, p1 - p0, fc="#f6dcc0", ec="#d9822b"))
    ax.add_patch(Rectangle((q1, p0), q0 - q1, p1 - p0, fc="#f2b880", ec="#b35c00", hatch="//"))
    ax.add_patch(Rectangle((q1, 0), q0 - q1, p0, fc="white", ec="gray", ls="--"))
    ax.text(q1 / 2, p0 / 2, "もとの代金のうち、\n今も払っている分", ha="center", va="center")
    ax.text(q1 / 2, p0 + 0.7, "値段だけのせいで増えた分", ha="center", va="center")
    ax.text(q1 + 1.5, p0 + 0.7, "混合\n差異", ha="center", va="center", fontsize=8)
    ax.text(q1 + 1.5, p0 / 2, "量だけの\nせいで\n減った分", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, q0 + 6); ax.set_ylim(0, p1 + 1)
    ax.set_xlabel("量 [kg]"); ax.set_ylabel("1kg あたりの値段 [ポンド]")
    ax.set_title("値段の変化と、量の変化（混合差異は、角の小さな長方形）")

    ax = axes[1]
    ax.plot(price, shrink, color="#2b7bb9", lw=2, label="値上がりが先、量があと")
    ax.plot(shrink, price, color="#d9822b", lw=2, label="量が先、値上がりがあと")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", label="弦（同時に動く）")
    ax.fill_between(price, shrink, price, color="#2b7bb9", alpha=0.15)
    ax.set_xlabel("豚の値段（変化の割合）"); ax.set_ylabel("チャーシューの減り方（変化の割合）")
    ax.set_title(f"レヴィ面積：符号が、どちらが先かを表す（{levy:+.2f} と {-levy:+.2f}）")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    jpfont.save(fig, "ramen_py05_order_and_area.jpg")


if __name__ == "__main__":
    main()
