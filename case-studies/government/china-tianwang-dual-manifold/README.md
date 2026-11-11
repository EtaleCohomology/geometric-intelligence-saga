# 敵の多様体

**── もし中国が同じ技術を持っていたら ── 同じ数学、正反対の測地線**

ジオメトリック・インテリジェンス ケーススタディ・ノベル

## あらすじ

2031年4月、ロンドン、カナリー・ワーフ。Eight Eyes（UKUSA 協定の５か国と日本・フランス・ドイツ）の多様体が稼働して半年、メリディアン・ジオメトリック・インテリジェンス社（Meridian Geometric Intelligence）の分析室で、リンが１つの異常に気づく。中国の行動から、人間の意思決定に必ずあるはずのノイズが消えている。まるで、中国もまた、世界を曲がった空間として描き、その上で最適な手を計算しているかのように。

リルル教授には、心当たりがあった。30年前、ケンブリッジの研究室にいた天才、陳偉明。陳偉明は、党の巨額の予算と、国家が動員した数百人の若い数学者とともに、中国の多様体「天網」を作り上げていた。フィンスラー計量、三重のリー括弧、外積代数、有限体、構成主義の数学 ── 天網の数学は、メリディアン社のものより精緻だった。そして、天網の改良版は、北極海での Eight Eyes の協調行動を察知した中国の情報機関が持ち出した、西側の設計の断片の上に立っていた。

しかし、天網には致命的な弱点があった。検閲されたデータである。地方の幹部は本当の数字を報告できず、天網は、正しいが選ばれた事実だけを映す鏡になっていく。同じ数学、正反対の目的関数。「正直な鏡は、嘘をつく鏡に長期的に勝つ」── 本書は、その命題を、数学の言葉で描く。

本書は、『ジオメトリック・インテリジェンス』理論を、政府の対外政策と国内政策の立案に役立てるための使用方法を、ケース・スタディ・ストーリー形式で示すことを目的とする。ケース・スタディ・ストーリー中に登場する数学とデータ処理・データ解析技術・人工知能技術は、すべて実在するものである。ストーリー展開は架空の物語であり、登場する人物・組織も架空の存在である。本書は、中国を敵とみなして囲い込む構図を描くことを目的としない。

## 想定読者

- 外交、安全保障、インテリジェンスの実務家と研究者、政策立案者
- 統計の信頼性、データの正直さ、メカニズムデザインに関心のある研究者
- 情報幾何学・微分幾何学・位相幾何学・代数幾何学・ゲーム理論を社会の問題に使いたいデータサイエンティスト・機械学習エンジニア
- 数学を社会の問題に使うことに関心のある学生

## ファイル構成

### 日本語版 ([ja/](ja/))

| ファイル | 内容 |
|---------|------|
| ja/00_story_main.pdf | 本編ストーリー |
| ja/01_casestudy_GI_theory_mapping.pdf | 補助文書01：本編と GI理論の対応。本書の位置づけ、GI理論の基本の流れ（10のステップ、命題1.1の３条件）、本作の課題、本編の各章と GI理論の対応、教材パートの実務手順、本作で GI理論に加わったもの、本作が示す GI理論の限界、シリーズの時系列 |
| ja/02_mathematics.pdf | 補助文書02：本編で使用された数学の解説。GI理論の基礎、命題1.1と検閲の影響、計量・測地線・曲率・フィンスラー計量、ベクトル場とリー括弧・BCH の公式、パーシステント・ホモロジー・モース理論・手術理論の言葉・層、テンソル代数と離散数学、ナッシュ均衡と多様体均衡（クリスターイ、平均場ゲーム、変分的合理性、min-max 最適化、BKK の定理）、メカニズムデザインとベイズ説得、因果推論と情報理論、シンプレクティック積分、暗号と形式検証、abc 予想と IUT 理論 |
| ja/03_python_scripts.pdf | 補助文書03：本編の数学を確かめる Python のコード13本の全文と実行結果 |
| ja/04_industry_knowledge.pdf | 補助文書04：政治・経済・社会の背景と、現実とフィクションの境界、出典。中国の統治機構と総体国家安全観、統計の信頼性、地方財政と隠れ債務、不動産不況と若年層の雇用、台湾海峡と危機管理、西側の情報共同体とクリアランス、反射統制、IUT 理論、銭学森、船舶位置の偽装と非協調的な観測 |

### コード ([code/](code/))

| ファイル | 内容 |
|---------|------|
| code/tk01_randomized_response.py | ランダム化回答法 |
| code/tk02_benford.py | ベンフォードの法則 |
| code/tk03_regularization_bias.py | 正則化が曲率にもたらす偏り |
| code/tk04_randers_asymmetry.py | ランダース計量（行きと帰りで長さの異なる定規） |
| code/tk05_bch_ordering.py | 政策の順序効果（リー括弧と三重のリー括弧） |
| code/tk06_manifold_conditions.py | 検閲が近傍保存率と最小特異値に与える影響 |
| code/tk07_iterated_best_response.py | 反復ベストレスポンスの収束と、ヘッセ行列による安定性の判定 |
| code/tk08_information.py | 相互情報量とデータ処理不等式、KL 情報量とワッサースタイン距離 |
| code/tk09_causal_tests.py | 移転エントロピーと偶然の一致の検定 |
| code/tk10_symplectic.py | シンプレクティック積分 |
| code/tk11_beta0_bridge.py | パーシステント・ホモロジーの β0（２つのかたまりがつながる） |
| code/tk12_bayesian_persuasion.py | ベイズ説得 |
| code/tk13_n_person_dilemma.py | N人囚人のジレンマとメカニズムデザイン |
| code/requirements.txt | 必要なライブラリ（numpy、scipy） |

    pip install -r code/requirements.txt
    python code/tk01_randomized_response.py

## 画像について

本編に含まれる画像（計184点）は、本編の PDF に収めるとともに、単独のファイルとして [images/](images/) に公開している。ファイル名は「本編での掲載順の番号（3桁）＋画像の内容を表す英語」とした（例：011_character_chen_weiming.jpg）。

| 掲載順の番号 | 内容 |
|---|---|
| 001 | 表紙 |
| 002〜018 | 登場人物の肖像 |
| 019〜184 | 本編の場面の挿絵と概念図（プロローグからエピローグ、付録まで） |

画像の作り方は、[IMAGE_CREDITS.md](IMAGE_CREDITS.md) に記した。

## 関連作品

- 政府編「Eight Eyesの多様体」: [../eight-eyes-intelligence-integration/](../eight-eyes-intelligence-integration/)
- 政府編「白夜の臨界」: [../arctic-lng-insurance-criticality/](../arctic-lng-insurance-criticality/)
- 投資ファンド業界編「曲率の裁定」: [../../corporate/hedge-fund-industry/](../../corporate/hedge-fund-industry/)
- 半導体製造装置業界編「光稜の巡回差」: [../../corporate/semiconductor-equipment-industry-2/](../../corporate/semiconductor-equipment-industry-2/)
- ジオメトリック・インテリジェンス論文: [theory/](../../../theory/)
- ジオメトリック・インテリジェンス手引書: [companion/](../../../companion/)
- 研究ノート「GI 理論の数学的基盤と代替アプローチの比較検討」: [research/mathematical_foundations/](../../../research/mathematical_foundations/)

## ライセンス

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## 著者

Étale Cohomology
