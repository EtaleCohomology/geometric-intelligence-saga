# s08 ハッシュの入れ子（マークルの木）と、ゼロ知識証明のまぐれの見込み（第10章）
import hashlib, math


def h(b):
    return hashlib.sha256(b).digest()


def merkle_root(leaves):
    level = [h(x) for x in leaves]
    while len(level) > 1:
        level = [h(level[k] + level[k + 1]) for k in range(0, len(level), 2)]
    return level[0]


def proof(leaves, idx):
    """ある写しと根を結ぶ道の上の、隣のハッシュ値の一覧"""
    level = [h(x) for x in leaves]; path = []
    while len(level) > 1:
        sib = idx ^ 1
        path.append((level[sib], sib < idx))
        level = [h(level[k] + level[k + 1]) for k in range(0, len(level), 2)]
        idx //= 2
    return path


def verify(leaf, path, root):
    cur = h(leaf)
    for sib, left in path:
        cur = h(sib + cur) if left else h(cur + sib)
    return cur == root


copies = [f"保全した写し {k}".encode() for k in range(8)]
root = merkle_root(copies)
print("根の値:", root.hex()[:24], "...")
changed = copies.copy(); changed[5] += b" "
print("写し５を１文字変えた根:", merkle_root(changed).hex()[:24], "...")

big = [f"写し {k}".encode() for k in range(1024)]
r = merkle_root(big); p = proof(big, 777)
print("写しが 1,024 個のとき、道の上のハッシュ値の数:", len(p), "= log2(1024) =", int(math.log2(1024)))
print("道だけで写し 777 を確認できるか:", verify(big[777], p, r))

# ゼロ知識証明の洞窟のたとえ：扉を開けられない人が、20回続けて言われた側から出てくる見込み
print("まぐれの見込み: 1/2^20 =", f"1/{2**20:,}", "≒ 100万分の１")
