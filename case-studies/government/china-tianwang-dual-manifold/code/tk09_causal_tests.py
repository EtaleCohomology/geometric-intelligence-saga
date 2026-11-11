"""
敵の多様体 補助文書03 コード9：移転エントロピーと、偶然の一致の検定（循環ずらしの置換検定）
X（ロシアの関与が疑われる動き）が、1期遅れて Y（中国の演習）に影響する架空の時系列を作る。
(1) 移転エントロピー TE(X→Y) と TE(Y→X) を、2値化した系列で推定し、情報の流れの向きを比べる。
(2) Y を循環的にずらして（系列の性質を保ったまま）同時発生の数を数え直し、偶然でこれ以上重なる確率を求める。
数値は架空のもの。いずれも「意図」は示さない。
"""
import numpy as np
rng = np.random.default_rng(5)
T = 3000
x = (rng.random(T) < 0.2).astype(int)
y = np.zeros(T, int)
for t in range(1, T):
    p = 0.6 if x[t - 1] == 1 else 0.08
    y[t] = int(rng.random() < p)
def te(src, dst):
    # TE(src→dst) = Σ p(d1, d0, s0) log p(d1|d0,s0)/p(d1|d0)
    d1, d0, s0 = dst[1:], dst[:-1], src[:-1]; tot = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                p_abc = np.mean((d1 == a) & (d0 == b) & (s0 == c))
                if p_abc == 0: continue
                p_a_bc = p_abc / np.mean((d0 == b) & (s0 == c))
                p_a_b = np.mean((d1 == a) & (d0 == b)) / np.mean(d0 == b)
                tot += p_abc * np.log2(p_a_bc / p_a_b)
    return tot
print(f"移転エントロピー TE(X→Y) = {te(x, y):.4f} ビット   TE(Y→X) = {te(y, x):.4f} ビット")
obs = np.sum(x[:-1] & y[1:])                       # X の直後に Y が起きた回数
null = [np.sum(x[:-1] & np.roll(y, s)[1:]) for s in range(50, T - 50, 3)]
p = (np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)
print(f"重なりの回数 {obs}、ずらした系列での平均 {np.mean(null):.1f}、偶然でこれ以上重なる確率 p = {p:.4f}")
print("→ 情報の流れの向きと、偶然では説明しにくいことは示せる。合意か便乗か（意図）は、データだけでは区別できない。")
