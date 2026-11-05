[English](./README.md) | [日本語](./README.ja.md)

# 社会経済の地図の上で「どう動くか」を決める

条件に合格した地図（VAEの潜在空間）の上で、最も無理のない道筋（測地線）を選び、施策の順番の効果（Lie括弧）を測り、施策を流れへの介入として扱い（Neural ODE、最適制御）、道筋の確からしさを確かめる方法を、地図のたとえで紹介する入門のチュートリアルです。Qiitaの連載「ジオメトリック・インテリジェンス」第3回のコードを置いています。

## 想定読者

- 企業や官庁で働く、Python機械学習エンジニア
- 経営コンサルタント
- 政策コンサルタント
- 機械学習（AI）とAIエージェントに明るい、国会議員の政策スタッフ・秘書
- 高度な数学を、経営や政策の分析に使える道具として応用することに関心のある研究者・Research Scientist

## コード

| ファイル | 内容 |
|---|---|
| [code/geodesic_hill_jax.py](./code/geodesic_hill_jax.py) | 実験1：丘のある地形で、測地線が丘を回り込むことを確かめる |
| [code/lie_bracket_parking_jax.py](./code/lie_bracket_parking_jax.py) | 実験2：縦列駐車で、Lie括弧を確かめる |
| [code/neural_ode_flow_jax.py](./code/neural_ode_flow_jax.py) | 実験3：Neural ODEで、流れと地形の骨組み（谷底と峠）を学び直す |

```
pip install -r code/requirements.txt
python code/geodesic_hill_jax.py 0.05      # 1つ目の引数：KLの重み β
python code/geodesic_hill_jax.py 0.05 3    # 2つ目の引数：学習の乱数の種（省略すると1）
python code/lie_bracket_parking_jax.py
python code/neural_ode_flow_jax.py
```

記事の中の数値は、このコードを筆者の環境で実行して得たものです。

## 読む順番

1. [第1回](../vae_to_socioeconomic_map/)：地図を確かめる（定規を当ててよいか）
2. [第2回](../reading_current_position_shape/)：いま立っている場所のかたちを読む
3. 第3回（本チュートリアル）：どう動くかを決める
4. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf)：GI理論の体系的な入門

## 関連資料

- 記事の中で紹介した架空の設定（業界別・政策領域別のケーススタディ・ノベル）：[case-studies](../../../case-studies/)

## 注意

記事の中で紹介したケーススタディの内容は、筆者が書いたフィクションのストーリーの設定です。実在する企業や政府機関の実データを使って得られた知見ではありません。

## ライセンス

CC BY 4.0
