[English](./README.md) | [日本語](./README.ja.md)

# 社会経済の地図で「いま立っている場所」のかたちを読む

条件に合格した地図（VAEの潜在空間）の上で、本当の近さを測り、曲率の符号で「崖」を見つけ、モース理論で谷底・峠・山頂を見つけ、パーシステント・ホモロジーと層（sheaf）で地図の「つながり方」と「矛盾」を読む方法を、地図のたとえで紹介する入門のチュートリアルです。Qiitaの連載「ジオメトリック・インテリジェンス」第2回のコードを置いています。

## 想定読者

- 企業や官庁で働く、Python機械学習エンジニア
- 経営コンサルタント
- 政策コンサルタント
- 機械学習（AI）とAIエージェントに明るい、国会議員の政策スタッフ・秘書

## 内容

- リーマン計量で測る「本当の近さ」
- スカラー曲率と、その符号の意味
- モース理論：臨界点（谷底・峠・山頂）、退化した臨界点、水位のたとえ、谷が浅くなるときの転換点
- パーシステント・ホモロジーと層（sheaf）、ネットワーク解析との関係
- 物理学と工学での、曲がった空間の上の計算
- 答えが分かっている形（球の表面、馬の鞍の形）で、条件の確認から曲率・要所の検出までを確かめるPythonのコード（JAX）

## 資料

📄 [reading_current_position_shape_JA.pdf](./reading_current_position_shape_JA.pdf)（日本語。Qiitaの記事「社会経済の地図で「いま立っている場所」のかたちを読む」をPDFにしたもの。記事は https://qiita.com/etale_cohomology/items/940ecb65278977f3d7fd ）

## コード

- [code/gi_manifold_check_curvature_morse_jax.py](./code/gi_manifold_check_curvature_morse_jax.py)

```
pip install -r code/requirements.txt
python code/gi_manifold_check_curvature_morse_jax.py sphere 0.001
python code/gi_manifold_check_curvature_morse_jax.py sphere 0.05
python code/gi_manifold_check_curvature_morse_jax.py saddle 0.05
python code/gi_manifold_check_curvature_morse_jax.py saddle 0.05 3   # 3つ目の引数：学習の乱数の種（省略すると1）
```

記事の結果の表と、谷底・峠の位置は、このコードを筆者の環境で実行して得たものです。

## 読む順番

1. [第1回](../vae_to_socioeconomic_map/)：地図を確かめる（定規を当ててよいか）
2. 第2回（本チュートリアル）：いま立っている場所のかたちを読む
3. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf)：GI理論の体系的な入門

## 関連資料

- 記事の中で紹介した架空の設定（業界別・政策領域別のケーススタディ・ノベル）：[case-studies](../../../case-studies/)

## 注意

記事の中で紹介したケーススタディの内容は、筆者が書いたフィクションのストーリーの設定です。実在する企業や政府機関の実データを使って得られた知見ではありません。

## ライセンス

CC BY 4.0
