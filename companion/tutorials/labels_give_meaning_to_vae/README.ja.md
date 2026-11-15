# ラベルで VAE の潜在空間に意味を与える

Qiita の記事「ラベルで VAE の潜在空間に意味を与える ── 半教師つき・条件つき VAE の代表的な4本の論文」のコードです。

架空のデータで、次の実験を行います。

- `labels`：ラベルの数（0件、20件、200件）を変え、情報量（潜在空間の全体で要因を説明できる割合）と、並び（軸ごとに要因を説明できる割合）を比べる
- `compare`：同じ20件のラベルで、同時に学習する方法と、後から条件づける方法（意味の方向）を、乱数の種を5通りに変えて比べる
- `manipulate`：軸や意味の方向を動かしたとき、ねらった要因だけが動くかを、真の要因の読み取り役で確かめる
- `correlated`：要因どうしに相関があるデータで学習し、相関のないデータで確かめる

## 実行方法

    pip install -r code/requirements.txt
    python code/gi_few_labels_jax.py            # すべての実験（CPU で数分）
    python code/gi_few_labels_jax.py compare    # 1つの実験だけ

数値はすべて架空のものです。考え方を確かめるための教材です。

## 著者

Étale Cohomology

## ライセンス

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
