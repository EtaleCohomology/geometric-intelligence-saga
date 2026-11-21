"""11 秘密を見せずに、正しいことだけを示す（ゼロ知識証明）のおもちゃの例（本編第2章・第6章）。

注意：これは、考え方を確かめるための、とても小さな数のおもちゃである。実際の秘密を守るためには使えない。
シュノアの識別の手順（Schnorr, 1991）を使う。
  公開：素数 p = 2039、q = 1019（p = 2q + 1）、位数 q の元 g、公開の値 y = g^x mod p
  秘密：x（証明する側だけが知っている）
  主張：「私は、y = g^x となる x を知っている」
  1. 証明する側：乱数 r を選び、t = g^r を送る
  2. 確かめる側：乱数 c を送る
  3. 証明する側：s = r + c x mod q を送る
  4. 確かめる側：g^s = t · y^c mod p なら受け入れる
確かめること
  (a) 完全性：x を知っていれば、必ず受け入れられる
  (b) 健全性：同じ t に、２つの違う c で正しく答えられる者は、x を計算できる（だから、知らない者は、ほぼ c を当てるしかない）
  (c) ゼロ知識：x を知らなくても、(t, c, s) の記録と同じ分布の記録を作れる（記録から、x について何も分からない）
  (d) 限界：示せるのは、あらかじめ決めた主張だけ。y がどの数字から作られたかが正しいかどうかは、示さない
"""
import random
from collections import Counter

# 出力を毎回同じにするため、種を固定した乱数を使う。本物の仕組みでは、暗号用の乱数（secrets など）を使う。
rng = random.Random(11)

def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))

p, q = 2039, 1019
assert is_prime(p) and is_prime(q) and p == 2 * q + 1
g = 4                                  # 2 の２乗。位数 q の部分群を生成する
assert pow(g, q, p) == 1 and g != 1

x = rng.randrange(q - 1) + 1       # 秘密
y = pow(g, x, p)                       # 公開の値

def prove_round(x, c_choice=None):
    r = rng.randrange(q)
    t = pow(g, r, p)
    c = rng.randrange(q) if c_choice is None else c_choice
    s = (r + c * x) % q
    return t, c, s, r

def verify(t, c, s):
    return pow(g, s, p) == (t * pow(y, c, p)) % p

# (a) 完全性
ok = sum(verify(*prove_round(x)[:3]) for _ in range(2000))
print(f"(a) 完全性：2000回のうち、受け入れられた回数 {ok}")

# (b) 健全性：同じ r（同じ t）で、２つの c に答えた記録から x を取り出す
r = rng.randrange(q); t = pow(g, r, p)
c1, c2 = 17, 503
s1, s2 = (r + c1 * x) % q, (r + c2 * x) % q
x_recovered = ((s1 - s2) * pow(c1 - c2, -1, q)) % q
print(f"(b) 健全性：２つの答えから取り出した値が秘密と一致するか {x_recovered == x}")

# x を知らない者が、c を当てずっぽうにして通る確率（c を予想して t を細工する）
def cheat_round():
    c_guess = rng.randrange(q); s = rng.randrange(q)
    t = (pow(g, s, p) * pow(pow(y, c_guess, p), -1, p)) % p
    c = rng.randrange(q)
    return verify(t, c, s)
cheats = sum(cheat_round() for _ in range(20000))
print(f"    x を知らない者が通った回数：20000回のうち {cheats}回（1 回あたり、およそ 1/{q} の確率）")

# (c) ゼロ知識：x を使わずに記録を作る（s と c を先に選び、t を逆算する）
def simulate():
    c = rng.randrange(q); s = rng.randrange(q)
    t = (pow(g, s, p) * pow(pow(y, c, p), -1, p)) % p
    return t, c, s
sim_ok = sum(verify(*simulate()) for _ in range(2000))
real_s = Counter(prove_round(x)[2] % 10 for _ in range(50000))
sim_s = Counter(simulate()[2] % 10 for _ in range(50000))
print(f"(c) ゼロ知識：x を使わずに作った記録 2000件のうち、確かめる側の式を満たした件数 {sim_ok}")
print("    s の下１桁の出現回数（本物の記録 / 作った記録）：",
      [(d, real_s[d], sim_s[d]) for d in range(5)], "…")
print("    本物の記録と、秘密なしで作った記録は、同じ分布になる。だから、記録から秘密は分からない。")
print("    （ここで示したのは、確かめる側が c を正直に乱数で選ぶ場合のゼロ知識である）")

# (d) 限界
print("(d) 限界：この手順が示すのは『y = g^x となる x を知っている』ことだけである。")
print("    『どの国にも資産の半分より多くは置いていない』のような主張には、範囲の証明など、別の仕組みがいる。")
print("    また、y を作るもとの数字が偽りなら、偽りの数字について、正しく証明するだけである。")
print("    安全さは、離散対数（g^x から x を求めること）が現実の時間では解けないという仮定に支えられている。")
print("    この例の p は小さいので、x は総当たりですぐに分かる：",
      next(k for k in range(q) if pow(g, k, p) == y) == x)
