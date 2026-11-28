# s06 シャノンのエントロピーは「調べる順番をつける網」（第9章）
# 暗号化そのものは行わない。でたらめな並び（os.urandom）を、暗号化されたファイルの代わりに使う。
import os, zlib, hashlib, math, random
from collections import Counter
from common import plt, savefig, NAVY, CYAN, AMBER, RED


def entropy(b):
    n = len(b); c = Counter(b)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def chi2(b):
    n = len(b); c = Counter(b); e = n / 256
    return sum((c.get(k, 0) - e) ** 2 / e for k in range(256))


random.seed(7)
words = ["荷物", "仕分け", "ライン", "伝票", "区画", "出荷", "到着", "埼玉", "設定", "時刻", "倉庫"]
text = ("、".join(random.choice(words) for _ in range(6000)) + "。").encode("utf-8")
samples = {
    "文章（設定のファイルの代わり）": text,
    "圧縮した文章（写真や書庫の代わり）": zlib.compress(text, 9),
    "でたらめな並び（暗号化の代わり）": os.urandom(len(zlib.compress(text, 9))),
}
for name, b in samples.items():
    print(f"{name}: エントロピー {entropy(b):.2f} ビット/バイト, カイ二乗 {chi2(b):,.0f}")
print("上限は log2(256) =", math.log2(256))

# 照らし合わせて決める（第9章の数）
total, same = 380_000, 369_000
changed = total - same
enc_text, photos, photo_mismatch, new_files, new_high = 8_200, 2_600, 700, 200, 140
assert enc_text + photos + new_files == changed
print(f"\n変わったファイル {changed:,} = 8,200 + 2,600 + 200")
print(f"暗号化された見込みが高い: {enc_text:,} + {photo_mismatch:,} = {enc_text + photo_mismatch:,}")
print(f"決められない: {new_high}")

# ハッシュ値：１ビット変えると、まったく別の値になる
b = bytearray(text[:64]); h1 = hashlib.sha256(b).hexdigest(); b[0] ^= 1
print("\nSHA-256:", h1[:16], "→ 1ビット変えると", hashlib.sha256(b).hexdigest()[:16])

fig, ax = plt.subplots(figsize=(6.5, 2.8))
vals = [entropy(b) for b in samples.values()]
ax.barh(["文章", "圧縮した文章", "でたらめな並び"], vals, color=[CYAN, AMBER, RED])
for k, v in enumerate(vals):
    ax.text(v + 0.05, k, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 8.8); ax.axvline(8, color=NAVY, ls="--", lw=1)
ax.set_xlabel("エントロピー（ビット/バイト。上限は８）")
ax.set_title("圧縮したファイルも、暗号化に見える", fontsize=10)
savefig(fig, "s06_entropy.png")
