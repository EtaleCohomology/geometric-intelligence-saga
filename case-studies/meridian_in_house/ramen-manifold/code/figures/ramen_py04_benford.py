"""第５章：ベンフォードの法則を、当てはめてはいけないデータに当てはめると。

ベンフォードの法則：先頭の数字が d になる割合 = log10(1 + 1/d)。
これが成り立つのは、数が何桁にもまたがって、広く散らばっているときである。

ラーメン店のお品書きの値段は、10 ポンドから 20 ポンド前後に集まる。
だから、先頭が 1 の値段が多いのは当然で、何かを隠している証拠にはならない。
数値は、物語の中の架空の値である。
"""
import numpy as np
import matplotlib.pyplot as plt
import jpfont

rng = np.random.default_rng(312)


def first_digit(x):
    x = np.asarray(x, float)
    return (x / 10 ** np.floor(np.log10(x))).astype(int)


def menu_prices(n=312):
    # ラーメン（主役）、ご飯もの・つまみ、飲みもの、を混ぜた架空のお品書き
    kinds = rng.choice(3, size=n, p=[0.44, 0.31, 0.25])
    p = np.where(kinds == 0, rng.uniform(13.0, 19.9, n),
        np.where(kinds == 1, rng.uniform(4.5, 12.5, n), rng.uniform(3.0, 7.5, n)))
    return np.round(p * 2) / 2  # 50 ペンス刻み


def main():
    jpfont.setup()
    d = np.arange(1, 10)
    benford = np.log10(1 + 1 / d)
    assert abs(benford.sum() - 1) < 1e-12

    prices = menu_prices()
    fd = first_digit(prices)
    share = np.array([(fd == k).mean() for k in d])
    print(f"お品書き {len(prices)} 件：先頭が 1 の割合 {share[0]*100:.1f}%（ベンフォードなら {benford[0]*100:.1f}%）")

    # 何桁にもまたがるデータ（例：店の年商、対数正規分布）
    wide = rng.lognormal(mean=13, sigma=2.0, size=5000)
    fd2 = first_digit(wide)
    share2 = np.array([(fd2 == k).mean() for k in d])
    print(f"何桁にもまたがるデータ：先頭が 1 の割合 {share2[0]*100:.1f}%")
    print(f"お品書きの値段の、最大 ÷ 最小 = {prices.max()/prices.min():.1f} 倍（1 桁にも満たない）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sh, title in ((axes[0], share, "お品書きの値段（3〜20 ポンド）"),
                          (axes[1], share2, "何桁にもまたがるデータ（店の年商など）")):
        ax.bar(d, sh, color="#2b7bb9", alpha=0.8, label="データ")
        ax.plot(d, benford, "o-", color="#d9822b", label="ベンフォードの法則")
        ax.set_xticks(d)
        ax.set_xlabel("先頭の数字")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("割合")
    fig.suptitle("ベンフォードの法則が成り立つデータと、成り立たないデータ")
    jpfont.save(fig, "ramen_py04_benford.jpg")


if __name__ == "__main__":
    main()
