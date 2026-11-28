# s13 結合写像の力学系：２つの倉庫の仕分けの数（第5章「大きすぎる道具」）
def step(x, normal):
    """足りない個数が毎日半分になる写像"""
    return normal - (normal - x) / 2


print("倉庫１つ：", end=" ")
x = 0
for day in range(4):
    print(f"{day}日後 {x:.0f}個", end=" / ")
    x = step(x, 1000)
print()

a, b = 0, 600                  # 倉庫甲（ふだん1,000個）、倉庫乙（ふだん600個）
fa, fb = step(a, 1000), step(b, 600)
a2 = 0.9 * fa + 0.1 * fb
b2 = 0.9 * fb + 0.1 * fa
print(f"つながりを考えない明日：甲 {fa:.0f}, 乙 {fb:.0f}")
print(f"相手の様子が１割入り込む明日：甲 {a2:.0f}, 乙 {b2:.0f}")
