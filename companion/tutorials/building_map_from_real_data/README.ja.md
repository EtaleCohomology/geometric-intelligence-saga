[English](./README.md) | [日本語](./README.ja.md)

# 本物の社会経済データから「地図」を作るまでの道のり

本物の社会経済データには、欠測、外れ値、単位と桁の違い、粒度の違いなどの乱れがあります。データを選び、乱れを手当てし、3つの条件に合格する地図（VAEの潜在空間）を作るまでの道のりを紹介する入門のチュートリアルです。Qiitaの連載「ジオメトリック・インテリジェンス」第4回のコードを置いています。

## 想定読者

- 企業や官庁で働く、Python機械学習エンジニア
- 経営コンサルタント
- 政策コンサルタント
- 機械学習（AI）とAIエージェントに明るい、国会議員の政策スタッフ・秘書
- 高度な数学を、経営や政策の分析に使える道具として応用することに関心のある研究者・Research Scientist

## 内容

- 地図に入れるデータを、分析の問いから逆算して選ぶ考え方
- 実データの問題（名寄せ、粒度、単位、欠測、外れ値、更新の遅れと改定、構造の変化、外に出せないデータ）と手当て
- 3つの条件のどれに落ちたかで、原因の見当をつけて直す方法
- 業界別の地図の設計の案（金融、製造、エネルギー）
- 試行錯誤の記録の残し方
- 答えが分かっている形を、わざと乱れたデータに変え、前処理の違いが3つの条件をどう変えるかを確かめるPythonのコード（JAX）

## コード

- [code/gi_real_data_preprocessing_jax.py](./code/gi_real_data_preprocessing_jax.py)

```
pip install -r code/requirements.txt
python code/gi_real_data_preprocessing_jax.py clean 0.05               # 参考：乱れのないデータ
python code/gi_real_data_preprocessing_jax.py naive 0.05               # (A) 0で埋めるだけ
python code/gi_real_data_preprocessing_jax.py zscore 0.05              # (B) 標準化だけ
python code/gi_real_data_preprocessing_jax.py careful 0.05             # (C) 手当て一式
python code/gi_real_data_preprocessing_jax.py careful_mask 0.05        # (D) (C)＋欠測の印
python code/gi_real_data_preprocessing_jax.py careful_quarterly 0.05   # (E) (D)＋粒度をそろえる
python code/gi_real_data_preprocessing_jax.py careful_quarterly 0.05 3 # 3つ目の引数：学習の乱数の種（省略すると1）
```

記事の中の数値は、このコードを本稿執筆者の環境で実行して得たものです。

## 読む順番

1. [第1回](../vae_to_socioeconomic_map/)：地図を確かめる（定規を当ててよいか）
2. [第2回](../reading_current_position_shape/)：いま立っている場所のかたちを読む
3. [第3回](../deciding_how_to_move/)：どう動くかを決める
4. 第4回（本チュートリアル）：本物のデータから地図を作る
5. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf)：GI理論の体系的な入門

## 関連資料

- 業界別・政策領域別のデータの制約と、条件に合格するまでの試行錯誤（架空のケーススタディ）：[gi_practice_materials](../gi_practice_materials/)
- 記事の中で紹介した架空の設定（業界別・政策領域別のケーススタディ・ノベル）：[case-studies](../../../case-studies/)

## 注意

記事の中で紹介したケーススタディの内容は、筆者が書いたフィクションのストーリーの設定です。実在する企業や政府機関の実データを使って得られた知見ではありません。業界別の地図の設計は、筆者が考えた案であり、実在の企業で使われていることを示すものではありません。

## ライセンス

CC BY 4.0
