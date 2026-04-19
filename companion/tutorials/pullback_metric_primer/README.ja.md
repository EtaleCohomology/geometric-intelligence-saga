# Pullback Metric Primer

VAE(Variational Autoencoder)の潜在空間におけるリーマン計量(引き戻し計量)の入門的解説。

## 対象読者

- 数学初心者のデータサイエンティスト
- 企業内の研修教育・勉強会
- GI(Geometric Intelligence)理論に興味があり、その数学的基盤を直感的に理解したい方

## 内容

VAE のデコーダが作る潜在空間は、一見平坦なベクトル空間に見えるが、実はデータ空間への写像によって伸び縮みする曲がった空間(多様体)である。本稿は、この歪みを正しく測るための**引き戻し計量**(pullback metric)の考え方を、数式・図解・Python 実装の 3 方向から解説する。

主要トピック:

- 100 次元のデータを 2 次元の戦略地図に凝縮する VAE の構造
- ユークリッド距離の限界(「地図上の直線が遠回り」)
- ヤコビアン J_f と計量テンソル g = J_f^T J_f の導出
- 数学的背景: 引き戻し計量(pullback metric)とは何か
- PyTorch による計量の実装例
- ユークリッド距離 vs リーマン距離 vs 情報幾何距離

## 文書

📄 [pullback_metric_primer_JA.pdf](./pullback_metric_primer_JA.pdf)

## 前提知識

- 高校数学(偏微分、行列の基本)
- Python の基礎(PyTorch を読める程度)
- 線形代数の基本(ベクトル、内積)

## 関連教材

- [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — GI 理論の体系的な入門教材(48 ページ、6 章)

## ライセンス

CC BY 4.0
