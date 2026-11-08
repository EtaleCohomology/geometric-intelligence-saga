# Image Credits — 投資ファンド業界編「曲率の裁定」

本作品「曲率の裁定 ── 平坦な確率空間が崩壊するとき」に含まれる画像（計49点）の作り方を、透明に開示する。

## 画像生成AIで生成した画像（39点）

Google Gemini により、Étale Cohomology の指示のもとで生成した。

- キャラクター肖像：12点
  - メリディアン・ジオメトリック・インテリジェンス社：リーム・ストリーム、リルル先生、倉田教授、カイト、リン、ユキ
  - Math Universe Capital Asset Management（MUCAM）：ヴィクター・ナッシュ（CEO 兼 CIO）、エレナ・チェン（CRO）、ドミトリ・コズロフ（クオンツ・チーム統括）、サラ・パテル（デジタル決済部門長）、杉田 純（インターンの大学院生）、長谷部 健（非常勤の数学顧問）
- 表紙と場面の挿絵・概念図：27点

すべての登場人物は架空の存在であり、実在の人物との類似は偶然である。

## Pythonで描いた図（10点）

数値や文字を正確に描く必要がある図は、Python（Matplotlib）で描いた。描くコードは code/figures/ にある。数値は、物語の中の架空の数値である。

- mucam_py01a_currency_graph.jpg、mucam_py01b_loop_sum.jpg：通貨のグラフを一周したときの、両替のずれ（離散的なホロノミー）
- mucam_py02_fisher_vs_euclid.jpg：平穏な市場と危機の市場の、ユークリッド距離とフィッシャー・ラオ距離
- mucam_py03_wasserstein_flat.jpg：何の距離で見るかで、空間の形が変わる（フィッシャー情報計量とワッサースタイン距離）
- mucam_py04_tail_and_es.jpg：損失の裾の形が変わる ── VaR と期待ショートフォール
- mucam_py05a_k_forms.jpg、mucam_py05b_collapse.jpg：k形式が測るもの、同じ矢印を２本並べると面積は0
- mucam_py06_volume_collapse.jpg：相関が1に近づくと、散らばりの「体積」がつぶれる
- mucam_py07a_big_loop_graph.jpg、mucam_py07b_loop_breakdown.jpg：大きな輪のずれを、三角形ごとに分ける

## 画像のファイル形式とメタデータ

公開にあたり、すべての画像を JPEG 形式にそろえ、撮影情報（EXIF）、XMP、コメント、画像の出どころの記録（C2PA）などの付加情報を取り除いた。
