[English](./README.md) | [日本語](./README.ja.md)

# python_implementations/

GI理論のプロトタイプ実装とベンチマーク。

## 目的

本サブディレクトリには動作するコードを置く。探索的な内容である:特定の数学的構造の実装、手法間のベンチマーク、`mathematical_foundations/`の理論的分析を支える小実験。

ここに置くコードはプロダクション用ではない。理解のための計装であり、デプロイ用ではない。プロトタイプが再利用可能なツールへと成熟した段階で、`companion/`に移行する。

## スコープ

典型的な内容:

- PyTorch autograd(`create_graph=True`)によるLie微分計算
- 訓練済みVAEデコーダからの引き戻し計量構築
- 計量計算時の条件数モニタリング
- Christoffel記号、スカラー曲率、測地線積分
- `torchdiffeq`によるニューラルODEと多様体ダイナミクス
- 架空データでの枠組み間の比較ベンチマーク

## ファイル

プロトタイプの進展に従って追加する。Pythonスクリプト(`.py`)とJupyter notebook(`.ipynb`)の両方を想定する。

## 規約

- Python 3.10以上
- 自動微分にはPyTorch
- ニューラルODEには`torchdiffeq`
- 古典的数値計算にはNumPyとSciPy
- 可視化にはMatplotlib
- 再現性が重要な箇所ではランダムシードを固定
- 各ファイルは目的、入力、出力、制限を記すdocstringで開始

## 依存関係

最初の実行可能スクリプトをコミットする際、`requirements.txt`を追加する。

## ライセンス

文書類はCC BY 4.0。実装にはMITまたは同等の寛容なライセンスを適用することがある。各ファイルはヘッダに自らのライセンスを明示する。
