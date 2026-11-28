# s11 数が p 個しかない時計の中で y² = x³ − x を解く（第2章「翌朝、もう一度」）
# ハッセの上限、フロベニウス写像、x² = 2 のガロア群の確認
import math
from common import plt, savefig, NAVY, CYAN, AMBER


def solutions(p):
    return [(x, y) for x in range(p) for y in range(p) if (y * y - (x**3 - x)) % p == 0]


print("p=5 の解:", solutions(5))
rows = []
for p in [3, 5, 7, 11, 13, 17]:
    n = len(solutions(p))
    rows.append((p, n, n - p, 2 * math.sqrt(p)))
    print(f"p = {p:2d}: 解の数 {n:2d}, 差 {n-p:+d}, ハッセの上限 2√p = {2*math.sqrt(p):.2f}")

# フロベニウス写像：p 個の時計では、どの数も p 乗すると自分に戻る（フェルマーの小定理）
print("５つの数の時計で x⁵ = x:", all(pow(x, 5, 5) == x for x in range(5)))

# x² = 2 のガロア群：a + b√2 を (a, b) の組で表して計算する
def mul(u, v):
    return (u[0] * v[0] + 2 * u[1] * v[1], u[0] * v[1] + u[1] * v[0])


def swap(u):           # √2 と −√2 を入れ替える
    return (u[0], -u[1])


u, v = (1, 1), (3, 2)
print("(1+√2)(3+2√2) =", mul(u, v), " / 入れ替えてから掛ける:", mul(swap(u), swap(v)),
      " / 掛けてから入れ替える:", swap(mul(u, v)))
print("有理数の解（y² = x³ − x, y = 0 のもの）:", [x for x in (-1, 0, 1) if x**3 - x == 0])


def primes(n):
    return [p for p in range(3, n) if all(p % d for d in range(2, int(p**0.5) + 1))]


ps = primes(200)
ns = [len(solutions(p)) for p in ps]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.fill_between(ps, [p - 2 * math.sqrt(p) for p in ps], [p + 2 * math.sqrt(p) for p in ps],
                color="#e3eef5", label="ハッセの上限の内側")
ax.plot(ps, ps, color=NAVY, lw=1, label="時計の数の個数 p")
ax.plot(ps, ns, "o", ms=3, color=AMBER, label="y² = x³ − x の解の数")
ax.set_xlabel("素数 p"); ax.set_ylabel("個数"); ax.legend(fontsize=8)
ax.set_title("解の数は p のまわりで揺れ、2√p より外へは出ない", fontsize=10)
savefig(fig, "s11_hasse.png")
