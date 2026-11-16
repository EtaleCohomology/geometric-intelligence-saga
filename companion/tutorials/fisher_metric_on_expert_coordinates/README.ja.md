[English](./README.md) | [日本語](./README.ja.md)

# 専門家の座標の上の、フィッシャー計量の階数（GI理論 番外編）

Qiita の記事「【GI理論 番外編】Gluzman氏の定規（計量）のないStrategic Manifoldは、GI理論の経営環境多様体に組み込めるか？ ── フィッシャー計量で人間の専門家の座標を測る」のコードです。

架空のデータで、次の4つの実験を行います。

- `toy`：手で計算できる小さな例（専門家の座標2個 → 指標3個）で、ヤコビ行列と引き戻し計量を確かめる
- `rank`：結果が1つの二値だけのとき、フィッシャー計量の階数が1になる
- `two_rulers`：引き戻し計量とフィッシャー計量を同じ点で比べ、食い違う向きを探す
- `sigma_test`：安定性の式 Σ の形を、入れ子のモデルの比較（尤度比検定）で確かめる

## 資料

📄 [fisher_metric_on_expert_coordinates_JA.pdf](./fisher_metric_on_expert_coordinates_JA.pdf)（日本語、99ページ。Qiitaの記事「【GI理論 番外編】Gluzman氏の定規（計量）のないStrategic Manifoldは、GI理論の経営環境多様体に組み込めるか？ ── フィッシャー計量で人間の専門家の座標を測る」を、画像を入れてPDFにしたもの。記事は https://qiita.com/etale_cohomology/items/7c083c3021b0b826c892 ）

記事の中の対話篇に登場する人物と場面は、本稿執筆者が書いたフィクションです。

## 実行方法

    pip install -r code/requirements.txt
    python code/gi_expert_rulers_jax.py            # すべての実験
    python code/gi_expert_rulers_jax.py two_rulers # 1つの実験だけ

数値はすべて架空のものです。考え方を確かめるための教材です。

## 著者

Étale Cohomology

## ライセンス

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
