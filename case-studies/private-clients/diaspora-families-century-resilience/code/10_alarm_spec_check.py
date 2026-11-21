"""10 合図の計算が、紙に書いた約束（仕様）どおりかを確かめる（本編第6章）。

本編のダニエルの友人が話す「形式証明」は、定理証明支援系（Lean など）で、プログラムが仕様を満たすことを証明する方法である。
このスクリプトは証明ではない。小さな整数の範囲で、すべての入力を調べ尽くして食い違いを探す「検査」である。
検査は、調べた範囲の外の入力については、何も言わない。

仕様（一族と一緒に紙に書く約束を、数式に書き直したもの。値は架空）
  点数 s は 0〜1000 の整数。基準は c1 = 300（身の回りを確かめる）、c2 = 550（書類を整える）、c3 = 800（資産の一部を動かす）
  段階 = （s ≥ c1）+（s ≥ c2）+（s ≥ c3）。ただし、点数の裏付けになった、互いに独立した公開の情報源の数が２つ未満なら、段階は２を超えない
  性質１（単調）：同じ情報源の数なら、点数が高いほど、段階は下がらない
  性質２（上限）：段階は 0〜3。人が移るかどうかを表す段階は、仕様に存在しない
"""
import itertools

C1, C2, C3 = 300, 550, 800

def spec(s, n_src):
    stage = (s >= C1) + (s >= C2) + (s >= C3)
    return min(stage, 2) if n_src < 2 else stage

def impl_ok(s, n_src):
    if s >= C3 and n_src >= 2:
        return 3
    if s >= C2:
        return 2
    if s >= C1:
        return 1
    return 0

def impl_bug_strict(s, n_src):           # 「以上」を「より大きい」と取り違えた実装
    if s > C3 and n_src >= 2:
        return 3
    return 2 if s > C2 else 1 if s > C1 else 0

def impl_bug_source(s, n_src):           # 情報源の数の条件を書き忘れた実装
    return (s >= C1) + (s >= C2) + (s >= C3)

def check(impl):
    bad = [(s, k) for s, k in itertools.product(range(0, 1001), range(0, 4)) if impl(s, k) != spec(s, k)]
    return bad

def monotone(f):
    for k in range(4):
        prev = f(0, k)
        for s in range(1, 1001):
            cur = f(s, k)
            if cur < prev:
                return (s, k)
            prev = cur
    return None

print("仕様そのものの性質の検査：単調でない点", monotone(spec),
      "／ 段階の値の範囲", sorted({spec(s, k) for s in range(1001) for k in range(4)}))
for name, f in [("正しい実装", impl_ok), ("『以上』を取り違えた実装", impl_bug_strict), ("情報源の条件を忘れた実装", impl_bug_source)]:
    bad = check(f)
    print(f"{name:20s} 食い違い {len(bad):4d} 件  例 {bad[:3]}")
print("\n検査で見つかるのは、仕様とプログラムの食い違いまでである。")
print("仕様（紙に書いた約束）が、一族が本当に守りたいことを表しているかどうかは、人が読んで確かめる。")
