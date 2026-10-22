# Step 2: Python によるデータ分析入門

**Excel 卒業生のためのデータ分析入門 — pandas・seaborn・networkx で学ぶ実用データサイエンス (全 12 回相当)**

GI (Geometric Intelligence) 理論への学習ロードマップにおける **Step 2**。Step 1 (Tableau) でドラッグ&ドロップによる可視化に慣れたエンジニアが、コードによる自動化・繰り返し分析の世界に踏み出すための教材。

## 概要

経済安全保障・地政学リスク・サイバーセキュリティの実務データ ── OFAC 制裁リスト、米国 Entity List、CISA KEV、財務省貿易統計、World Bank WGI など ── を、Python のデータサイエンスエコシステム (pandas / seaborn / networkx / matplotlib / scikit-learn) で分析する方法を、Google Colaboratory のハンズオン形式で学ぶ研修教材。

Tableau 版 ([../01_tableau-economic-security/](../01_tableau-economic-security/)) と同じ経済安全保障の文脈・同じ公開データソースを使い、ドラッグ&ドロップではなくコードで分析する方法を扱う設計になっている。

## 対象読者

- Excel で関数 (VLOOKUP, SUMIF, IF など) を使ったことがある実務家
- プログラミング言語に触ったことがない方
- 統計学やデータ分析の専門教育を受けたことがない方
- 経済安全保障・地政学リスク・サプライチェーンに業務で関わる方
- 制裁対応・取引先審査・与信判断などで、データを使った判断を求められる方
- Step 1 (Tableau) を学習済みで、自動化・繰り返し分析に進みたい方

## 学習目標

本教材を完了すると、以下ができるようになる:

- Google Colaboratory 上で Python コードを実行できる
- pandas で CSV / Excel / Web API からデータを読み込める
- pandas でデータの前処理・クリーニング・集計ができる
- seaborn で箱ひげ図・ヒストグラム・散布図・ヒートマップを描画できる
- 記述統計と相関分析がコードで実行できる
- networkx で取引先のサプライチェーン関係を可視化できる
- 時系列データに地政学イベントを参照線として重ねて分析できる
- scikit-learn で PCA とクラスタリングを使った多変量解析ができる
- 複数データソースを統合した実務的なリスク評価レポートを作成できる

## 構成 (全 12 回)

### 前半編 (基礎・入門)
- 第 1 回: Google Colaboratory と Python のはじめの一歩
- 第 2 回: pandas でデータを読み込む — Excel からの卒業
- 第 3 回: データの前処理と加工 — pandas でクリーニング
- 第 4 回: データの分布を可視化する — seaborn で美しく
- 第 5 回: 記述統計 — 平均・分散・相関を Python で
- 第 6 回: データ集計と前半編まとめ

### 後半編 (発展・応用)
- 第 7 回: ネットワーク分析 — networkx で関係性を描く
- 第 8 回: 時系列解析 — 地政学イベントの影響を見る
- 第 9 回: 多変量解析 — カントリーリスクの複合評価
- 第 10 回: サイバーセキュリティデータの分析
- 第 11 回: 統合分析レポート — 複数データを組み合わせる
- 第 12 回: 総合演習と次へのステップ

### 付録
- 付録 A: 公開データソース一覧
- 付録 B: Python コードクイックリファレンス
- 付録 C: よくあるエラーと対処法

## Tableau 版との関係

本教材は Step 1 ([../01_tableau-economic-security/](../01_tableau-economic-security/)) と同じデータセット・同じ経済安全保障の文脈で構成されている。

- **Tableau 版**: ドラッグ&ドロップで可視化に慣れたい方向け
- **Python 版** (本教材): 自動化・繰り返しの分析を見据えている方向け
- **両方学ぶ**: 業務で重宝されるデータ分析者になる王道。多くの実務家が両方使っている

## 使用するデータソース

すべて無料で利用できる公開データのみ。本教材は **教育目的** で各データソースを参照する方法を示すもので、データ自体を再配布するものではない。各データソースは、演習の指示に従って公式サイトから直接ダウンロードする。

| データソース | 提供元 | ライセンス |
| --- | --- | --- |
| OFAC SDN リスト | 米国財務省 | 米国政府著作物 (自由に利用可能) |
| Entity List | 米国商務省 | 米国政府著作物 (自由に利用可能) |
| CISA KEV | 米国 CISA | 米国政府著作物 (自由に利用可能) |
| NVD CVE | 米国 NIST | 米国政府著作物 (自由に利用可能) |
| World Bank WGI | 世界銀行 | CC BY 4.0 |
| OECD Country Risk | OECD | 教育目的での参考使用が可能 (再配布不可) |
| 財務省貿易統計 | 日本財務省 | 公開統計として一般利用可能 |
| OpenSanctions | OSS / NGO | 非商用利用は無料、商用利用は別途ライセンス |

業務で実際に活用される際は、各データソースの最新の利用規約を必ずご確認ください。商用ツール・サービスへの組み込みには、別途ライセンス契約が必要な場合があります。

## ファイル構成

### 日本語版 ([ja/](ja/))

- **python_economic_security_training.pdf** — 全 12 回 + 付録の完全版教材

### 英語版 (en/) — 公開準備中

### フランス語版 (fr/) — 公開準備中

## 必要な環境

- **Google Colaboratory** (無料): https://colab.research.google.com/
- Google アカウント
- ブラウザ (Chrome, Edge, Firefox など)
- インターネット接続

PC へのインストールは一切不要。すべてブラウザ上で完結する。

## 学習ロードマップにおける位置づけ
Step 1: Tableau (ドラッグ&ドロップ可視化)

↓

Step 2: Python (pandas, seaborn, networkx) ←── あなたはここ

↓

Step 3: 多様体プログラミング (Python)

↓

Step 4: GI 理論 (経営環境多様体、AI エージェント)

Step 2 を終えたら、Step 3 (多様体プログラミング) に進むことで、平らなユークリッド空間ではなく、曲がったリーマン多様体上での計算を Python で実装する技術を学べる。

## 関連リソース

- 前ステップ: [../01_tableau-economic-security/](../01_tableau-economic-security/) — Step 1: Tableau 経済安全保障研修
- 上位ステップ: 03_manifold-programming/ — Step 3: 多様体プログラミング (公開準備中)
- 理論的背景: [../../theory/](../../theory/) — GI 理論 Vol.1, Vol.2
- 応用例: [../../case-studies/](../../case-studies/) — GI 理論を実問題に適用したケーススタディ

## ライセンス

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## 著者

Étale Cohomology
