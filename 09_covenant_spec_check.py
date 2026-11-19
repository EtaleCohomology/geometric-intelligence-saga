"""09 財務制限条項の計算が、契約書の文言を書き直した仕様と一致するかを確かめる（第6章・第8章）。

杉田が使うのは、定理証明支援系による「証明」である。このスクリプトは証明ではなく、
小さな整数の範囲で、すべての入力を調べ尽くして食い違いを探す「検査」である。
証明と検査の違い：検査は、調べた範囲の外の入力については何も言わない。
仕様（契約書の文言を、数式に書き直したもの。第6章の架空の条項）
  単体の純資産 ≥ max(0.75 × 直前の期末の単体の純資産, 0.75 × 2032年3月期末の単体の純資産)
  判定は、各年度の決算期末だけ
"""
from fractions import Fraction as F
import itertools


def spec(na, prev, base):
    """仕様。分数で計算し、浮動小数点の丸めを持ち込まない。"""
    return na >= max(F(3, 4) * prev, F(3, 4) * base)


def impl_ok(na, prev, base):
    return 4 * na >= 3 * max(prev, base)          # 整数だけで同じ判定をする実装


def impl_bug_min(na, prev, base):
    return 4 * na >= 3 * min(prev, base)          # 「高い方」を「低い方」と取り違えた実装


def impl_bug_strict(na, prev, base):
    return 4 * na > 3 * max(prev, base)           # 「以上」を「より大きい」と取り違えた実装


def impl_bug_consolidated(na_single, na_consol, prev_single, prev_consol, base):
    """単体で判定すべきところを、連結の数字で判定してしまう実装（第6章の相手のモデルの誤り）。"""
    return 4 * na_consol >= 3 * max(prev_consol, base)


R = range(0, 41)
for name, f in [("正しい実装", impl_ok), ("高い方と低い方の取り違え", impl_bug_min),
                ("以上とより大きいの取り違え", impl_bug_strict)]:
    bad = [(a, b, c) for a, b, c in itertools.product(R, R, R) if f(a, b, c) != spec(a, b, c)]
    print(f"{name:16s} 食い違い {len(bad):6d} 件" + (f"  例 {bad[0]}" if bad else ""))

# 第6章の玄冬の数字（億円）：単体 640 → 495、連結 820 → 590、2032年3月期末の単体 600
print("\n玄冬の判定（契約書の定義は単体）")
print("  契約書どおり（単体）:", "線の上" if spec(495, 640, 600) else "線の下")
print("  連結の数字で判定   :", "線の上" if impl_bug_consolidated(495, 590, 640, 820, 600) else "線の下")
print("注意：プログラムが仕様と一致しても、仕様が契約書の法律の上の意味と一致しているかは、人が読んで確かめる。")
