[English](./README.md) | [日本語](./README.ja.md)

# 速度計と地形 ── 地図から得た経営インテリジェンスと、既存の指標を並べて読む（Geometric Intelligence論への招待 第5回）

Qiita の連載「Geometric Intelligence論への招待」第5回のコードです。

地図とは、社会経済のデータから構築した可微分多様体です。架空のデータで、既存の指標（VaR、期待ショートフォール、KRI）と、地図から得た経営インテリジェンス（崖までの近さ）を並べて読みます。

| 実験 | 内容 |
|---|---|
| `compare` | VaR は落ち着いているのに崖に近づいている局面と、VaR は跳ねるのに谷の底で安定している局面。地図を推定し直したときの幅 |
| `signals` | 「早く気づける」かの確かめ（当たり・見逃し・空振り・先行期間・ノイズ対シグナル比。学習に使っていない期間で評価） |
| `onepage` | 架空の会社「エタール重工株式会社」の、経営会議の1枚の数値 |

記事の図（地形と崖の領域、VaR と崖までの近さ、当たり・見逃し・空振り）は、`code/make_figures.py` で、同じ架空のデータから描けます（`figs/` に保存。日本語のフォントが必要）。

## 実行方法

    pip install -r code/requirements.txt
    python code/gi_kgi_kpi_jax.py compare
    cd code && python make_figures.py

数値はすべて架空のものです。エタール重工株式会社は架空の会社です。考え方を確かめるための教材です。

## 著者

Étale Cohomology

## ライセンス

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
