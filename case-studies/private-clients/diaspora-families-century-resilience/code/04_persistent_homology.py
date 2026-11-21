"""04 長く残る輪を見つける（パーシステント・ホモロジー）（本編第2章「人づての話」）。

点の集まり（架空）を、結ぶ距離の物差し r を少しずつ広げながら眺め、
輪（１次元の穴）が生まれる r と、ふさがって消える r を記録する（ヴィートリス・リップス複体。ripser を使う）。
1. 輪の形に並んだ点に雑音を加えたもの：長く残る輪が１つ、すぐ消える輪がいくつか
2. 輪のない、ばらばらの点：長く残る輪はない
3. 点の数を減らしたり、雑音を大きくしたりすると、長く残る輪が見えにくくなることを確かめる
輪の「長さ」（消えた r − 生まれた r）が長いものを、本当にある形の候補とみなす。
輪が何を意味するか（悪いことか、ふつうの商売か）は、数学は決めない。
"""
import numpy as np
from ripser import ripser

rng = np.random.default_rng(4)

def ring(n, noise):
    th = rng.uniform(0, 2 * np.pi, n)
    return np.c_[np.cos(th), np.sin(th)] + rng.normal(0, noise, (n, 2))

def blob(n):
    return rng.normal(0, 0.6, (n, 2))

def bars(X):
    d1 = ripser(X, maxdim=1)["dgms"][1]
    life = d1[:, 1] - d1[:, 0]
    order = np.argsort(-life)
    return d1[order], life[order]

for name, X in [("輪の形の点（120点、雑音 0.10）", ring(120, 0.10)),
                ("ばらばらの点（120点）", blob(120))]:
    d, life = bars(X)
    print(f"{name}：輪の数 {len(d)}")
    for k in range(min(4, len(d))):
        print(f"   生まれた r {d[k, 0]:.3f}  消えた r {d[k, 1]:.3f}  長さ {life[k]:.3f}")
    second = life[1] if len(life) > 1 else 0.0
    print(f"   いちばん長い輪 ÷ ２番目に長い輪 = {life[0] / max(second, 1e-9):.1f}\n")

print("点の数と雑音を変えたときの、いちばん長い輪と２番目の輪の長さ（各20回の平均）")
for n in [120, 40, 15]:
    for noise in [0.05, 0.15, 0.30]:
        a, b = [], []
        for _ in range(20):
            _, life = bars(ring(n, noise))
            a.append(life[0] if len(life) > 0 else 0.0); b.append(life[1] if len(life) > 1 else 0.0)
        print(f"   点 {n:3d}  雑音 {noise:.2f}  いちばん長い {np.mean(a):.3f}  ２番目 {np.mean(b):.3f}")
print("点が少なく雑音が大きいほど、いちばん長い輪が短くなり、ばらばらの点からも生まれる長さ 0.1 前後の輪との差が小さくなる。")
print("長さの比べ方は、輪のないデータ（比べる相手）を用意して決める。")
