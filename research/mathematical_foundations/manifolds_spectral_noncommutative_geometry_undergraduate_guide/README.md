# 多様体、スペクトル幾何学、非可換幾何学 ── 学部生のための見取り図

> **Undergraduate guide** — 多様体論と、スペクトル幾何学・非可換幾何学との関係を、GI 理論の地図を例にしてやさしく解説する

![status](https://img.shields.io/badge/status-draft-orange) ![license](https://img.shields.io/badge/license-CC--BY--4.0-blue) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![level](https://img.shields.io/badge/level-undergraduate-green)

**著者**: Étale Cohomology
**最終更新**: 2026 年 11 月
**前提にする知識**: 線形代数（固有値、行列の積が一般に交換しないこと）、微分積分（偏微分、2 階微分）、複素数 $e^{i\theta}$
**図解 PDF（画像入り）**: [`three_geometries_illustrated_undergrad.pdf`](./three_geometries_illustrated_undergrad.pdf)（『多様体、スペクトル幾何学、非可換幾何学 ── 3 つの関係』。図 20 枚入り、28 ページ）
**一緒に動かすコード**: [`small_examples_undergraduate.py`](./small_examples_undergraduate.py)（numpy のみ、1 秒以内）

## この解説で分かること

- **多様体**とは何か。GI 理論の地図が、なぜ多様体なのか
- **スペクトル幾何学**: 多様体の「音」（ラプラシアンの固有値）から、形について何が分かり、何が分からないか
- **非可換幾何学**: 多様体を「点」ではなく「関数の代数」と「作用素」で書き直すと、何ができるようになるか
- 3 つの関係と、GI 理論で使うときの注意

証明は省き、定義と例と、その結果が何を言っているかを中心に書く。

---

## 目次

- [1. 多様体](#1-多様体)
- [2. 多様体の上のラプラシアン](#2-多様体の上のラプラシアン)
- [3. スペクトル幾何学: 音から形を聞く](#3-スペクトル幾何学-音から形を聞く)
- [4. 非可換幾何学: 点の代わりに関数で考える](#4-非可換幾何学-点の代わりに関数で考える)
- [5. 3 つの関係の見取り図](#5-3-つの関係の見取り図)
- [6. GI 理論の地図では](#6-gi-理論の地図では)
- [7. よくある誤解](#7-よくある誤解)
- [8. 練習問題](#8-練習問題)
- [参考文献](#参考文献)

---

## 1. 多様体

### 1.1 局所的には平らな空間

地球の表面は丸いが、町の地図は平らな紙に描ける。地球全体を 1 枚の紙に、ゆがみも切れ目もなく描くことはできないが、何枚かの地図を重なりを持たせて貼り合わせれば、地球全体を覆える。

**$m$ 次元多様体**とは、このように、どの点の近くも $\mathbb{R}^m$ の一部と同じように座標で表せる空間である。

- 1 枚 1 枚の地図を**座標近傍**（チャート）と呼ぶ
- 2 枚の地図が重なる部分では、片方の座標からもう片方の座標への変換が決まる。この変換が何回でも微分できるとき、**滑らかな多様体**と呼ぶ

| 例 | 次元 | 1 枚の地図で覆えるか |
|---|---|---|
| 円 $S^1$ | 1 | 覆えない（角度 $\theta$ は 1 周でつながらない）。2 枚で覆える |
| 球面 $S^2$ | 2 | 覆えない。2 枚で覆える（北極側と南極側） |
| トーラス（ドーナツの表面） | 2 | 覆えない |
| 円筒 $S^1 \times [0, 3]$ | 2 | 覆えない（縁のある多様体） |

### 1.2 リーマン計量: 長さを測る道具

多様体の各点には、その点で動ける向きを集めた**接空間**（その点に接する平面のようなもの）がある。接空間に内積 $g$ を与え、それが点とともに滑らかに変わるとき、$g$ を**リーマン計量**と呼ぶ。リーマン計量があると、

- 曲線の長さ、2 点の間の最短の道（**測地線**）と距離
- 面積や体積
- 曲がり具合（**曲率**）

が計算できる。リーマン計量を持つ多様体を**リーマン多様体**と呼ぶ。

### 1.3 GI 理論の地図は多様体

GI 理論は、VAE のデコーダ $f: U \subset \mathbb{R}^d \to \mathbb{R}^n$ を使ってデータの地図を作る。第 1 巻の命題 2.1 は、$U$ がコンパクト、$f$ が滑らか、ヤコビ行列 $J$ がどこでもフルランク、$f$ が単射なら、$f(U)$ は多様体（埋め込み部分多様体）になり、

$$g = J^\top J$$

がリーマン計量になることを述べる（引き戻し計量）。つまり、GI 理論の地図は、コンパクトなリーマン多様体である。以下では、この地図を $M$ と書く。

## 2. 多様体の上のラプラシアン

$\mathbb{R}^m$ では、関数 $u$ に対して

$$\Delta u = -\left( \frac{\partial^2 u}{\partial x_1^2} + \cdots + \frac{\partial^2 u}{\partial x_m^2} \right)$$

を**ラプラシアン**と呼ぶ（この解説では、固有値が 0 以上になるように、マイナス符号をつける流儀を使う）。リーマン多様体の上でも、計量 $g$ を使って同じ働きの作用素を定義できる。これを**ラプラス＝ベルトラミ作用素**と呼ぶ。

$M$ がコンパクトでつながっているとき、$\Delta u = \lambda u$ を満たす固有値は

$$0 = \lambda_0 < \lambda_1 \le \lambda_2 \le \cdots \to \infty$$

のように、0 から始まって、とびとびに無限に並ぶ。この並びを $M$ の**スペクトル**と呼ぶ。

**例（円）**: 半径 1 の円の上では $\Delta u = -u''(\theta)$。$\cos k\theta$ と $\sin k\theta$ が固有関数で、固有値は $k^2$:

$$0,\ 1,\ 1,\ 4,\ 4,\ 9,\ 9,\ \dots$$

（コードの Example 1 で、差分の行列から数値で確認できる。）

## 3. スペクトル幾何学: 音から形を聞く

太鼓の膜は、決まった振動数でしか鳴らない。振動数の 2 乗が、ラプラシアンの固有値にあたる。**スペクトル幾何学**は、「多様体の音（スペクトル）から、形についてどこまで分かるか」を調べる分野である。Kac（1966）の「太鼓の音を聞いて形が分かるか」という問いで広く知られるようになった。

### 3.1 音から分かること

| 分かること | 結果 | やさしく言うと |
|---|---|---|
| つながった塊の数 | 固有値 0 の個数 | 塊ごとに一定の関数は、揺れない（固有値 0） |
| 次元と体積 | ワイルの法則（Weyl 1911） | 高い音の増え方が、次元と大きさで決まる |
| 曲率の積分 | 熱核の展開（Minakshisundaram–Pleijel 1949; McKean–Singer 1967） | 熱の広がり方の細部に、曲がり具合の合計が出る |
| 穴の数 | ホッジの定理 | 微分形式のラプラシアンの固有値 0 の個数が、穴の数 |
| くびれ | チーガーの不等式（Cheeger 1970） | くびれた図形は、低い音（小さい $\lambda_1$）を持つ |
| 曲がり具合の下限 | リクネロヴィッツの定理（Lichnerowicz 1958） | 一様に正に曲がった図形は、低い音を持てない |

**ワイルの法則.** $\lambda$ 以下の固有値の個数を $N(\lambda)$ とすると、$m$ 次元のコンパクトなリーマン多様体では

$$N(\lambda) \sim \frac{\omega_m \, \mathrm{Vol}(M)}{(2\pi)^m} \, \lambda^{m/2} \qquad (\lambda \to \infty)$$

が成り立つ（$\omega_m$ は $m$ 次元の単位球の体積）。円（$m = 1$、長さ $2\pi$、$\omega_1 = 2$）では右辺は $2\sqrt{\lambda}$ で、実際、$k^2 \le \lambda$ を満たす整数 $k$ の個数は約 $2\sqrt{\lambda}$ である（Example 2）。増え方の指数 $m/2$ から次元が、係数から体積が読める。

**熱核の展開.** 図形の上で熱が広がる様子は、固有値を使って $\sum_k e^{-\lambda_k t}$（熱跡）で表せる。短い時間 $t \to 0$ では

$$\sum_k e^{-\lambda_k t} \sim \frac{1}{(4\pi t)^{m/2}} \left( \mathrm{Vol}(M) + \frac{t}{6} \int_M R \, dV + O(t^2) \right)$$

となる（$R$ はスカラー曲率）。左辺はスペクトルだけで決まるので、体積と、**スカラー曲率の積分**がスペクトルから分かる。閉じた曲面では、ガウス＝ボネの定理と組み合わせると、オイラー標数（向きのつけられる閉曲面なら、穴の数）が分かる。ただし、分かるのは**積分した量**であって、「どこがどれだけ曲がっているか」という場所ごとの情報ではない。

**チーガーの不等式.** 図形を 2 つに切り分けるとき、切り口の大きさを、小さいほうの体積で割った値の最小値を $h$（チーガー定数）とすると、$\lambda_1 \ge h^2 / 4$。逆向きの評価（Buser 1982）もあり、細いくびれのある図形（$h$ が小さい）では $\lambda_1$ が小さくなる。

**リクネロヴィッツの定理.** $m$ 次元のコンパクトなリーマン多様体で、リッチ曲率が $(m-1)k > 0$ 以上なら、$\lambda_1 \ge m k$。等号は球面で成り立つ（Obata 1962）。

### 3.2 音から分からないこと

Milnor（1964）は、16 次元で、形が違うのにスペクトルがまったく同じ 2 つの平坦なトーラスを見つけた。Sunada（1985）は、このような例を系統的に作る方法を与え、Gordon–Webb–Wolpert（1992）は、平面の上で、同じ音で形の違う 2 つの太鼓を作った。

> 音が違えば形は違う。しかし、音が同じでも形が同じとは限らない。

### 3.3 グラフの上のスペクトル

データは図形そのものではなく点の集まりなので、実際の計算では、近い点どうしを結んだ**グラフ**のラプラシアン $L = D - W$ を使う（$W$ は重みの行列、$D$ は重みの合計を対角に並べた行列）。点が図形の上にたくさんあり、重みを適切に選べば、グラフのスペクトルの小さいほうは、図形のスペクトルに近づく（Belkin–Niyogi 2007）。GI 理論の地図の形の点検は、これを使う（[`../manifold_shape_diagnostics_undergraduate_guide/`](../manifold_shape_diagnostics_undergraduate_guide)）。

## 4. 非可換幾何学: 点の代わりに関数で考える

### 4.1 空間と、関数の代数は同じ情報を持つ

空間 $X$ の上の連続関数 $f, g$ は、足したり掛けたりできる。しかも、掛け算は交換する: $fg = gf$。こうした関数を全部集めたもの $C(X)$ は、**可換な代数**である。

驚くべきことに、コンパクトな空間 $X$ は、この代数 $C(X)$ から完全に復元できる（**ゲルファント＝ナイマルクの定理**、1943 年）。点 $x$ は、「関数に $x$ での値を対応させる操作」$f \mapsto f(x)$ として、代数の中から見つけ出せる。

**例（2 点の空間）**: 2 点 $\{p, q\}$ の上の関数は、値の組 $(a, b)$ で決まる。掛け算は成分ごと $(a, b)(c, d) = (ac, bd)$ で、交換する。この代数は $\mathbb{C}^2$ である。

> **空間を調べることと、その上の関数の代数を調べることは、同じことである。**

### 4.2 非可換にすると

それなら、交換しない代数（たとえば行列の代数。一般に $AB \ne BA$）を、「何かの空間の上の関数」とみなしたらどうか。これが**非可換幾何学**（Connes 1994）の出発点である。

- 量子力学では、位置と運動量が交換しない（$xp - px = i\hbar$）。このような世界は、ふつうの意味の点の集まりとしては描けない
- $n \ge 2$ の行列の代数には、$f \mapsto f(x)$ にあたる操作（積を保つ $\mathbb{C}$ への写像）がない。つまり**点がない**。それでも、代数を「空間」とみなして幾何学を進める
- 有名な例に、$uv = e^{2\pi i \theta} vu$ を満たす $u, v$ から作る**非可換トーラス**がある。$\theta = 0$ なら、ふつうのトーラスの関数の代数に戻る

### 4.3 距離を取り戻す: スペクトル三つ組

点がないと、距離をどう測るかが問題になる。コンヌの答えは、代数 $A$ に、ヒルベルト空間 $H$ と、ディラック作用素と呼ぶ自己共役作用素 $D$ を組にした**スペクトル三つ組** $(A, H, D)$ を使うことだった。

**円の例**: $A$ を円の上の滑らかな関数、$H$ を円の上の 2 乗可積分な関数の空間、$D = -i \dfrac{d}{d\theta}$ とする。関数 $f$ を「掛ける作用素」とみなすと、

$$[D, f] \, u = D(fu) - f(Du) = -i f' u$$

なので、交換子 $[D, f]$ の大きさは $\sup \lvert f' \rvert$（**関数の傾きの最大値**）になる。そこで、2 点 $x, y$ の距離を

$$d(x, y) = \sup \{\, \lvert f(x) - f(y) \rvert : \lVert [D, f] \rVert \le 1 \,\}$$

と定める（**コンヌの距離**）。「傾きが 1 以下の関数で、$x$ と $y$ の値の差をどこまで大きくできるか」を測っていて、答えは円の上の 2 点の弧の長さ（測地距離）に一致する。点そのものを使わず、**関数と作用素だけで距離を測った**ことになる。

この定義は、点がない非可換な代数にもそのまま使える（点の代わりに「状態」と呼ぶものの間の距離になる）。

**スペクトルとの関係**: 円の $D = -i \, d/d\theta$ の固有値は整数 $k$、$D^2 = -d^2/d\theta^2$ はラプラシアンで、固有値は $k^2$。**スペクトル三つ組の $D$ は、ラプラシアンの「平方根」にあたる**。スペクトル幾何学は $D$ の固有値を調べ、非可換幾何学は、それに代数 $A$ を組み合わせる。

**一般の多様体では**: ディラック作用素を定義するには、**スピン構造**という追加の構造が必要になる。スピン構造を仮定しない代わりに、微分形式の上の $d + d^*$（ホッジ＝ド・ラーム作用素）を使っても、スペクトル三つ組が作れる。

**再構成定理**: Connes（2008 年に公開、2013 年に掲載）は、**可換な**スペクトル三つ組がいくつかの条件を満たせば、それは多様体から来ていることを示した。つまり、多様体は、スペクトル三つ組の言葉で完全に書き直せる。

### 4.4 2 点の空間の距離

2 点の空間 $A = \mathbb{C}^2$、$H = \mathbb{C}^2$ に

$$D = \begin{pmatrix} 0 & m \\ m & 0 \end{pmatrix} \qquad (m \ne 0 \text{ は実数})$$

を置く。関数 $f = (a, b)$ を対角行列 $\mathrm{diag}(a, b)$ とみなすと、$\lVert [D, f] \rVert = \lvert m \rvert \, \lvert a - b \rvert$ なので、コンヌの距離は

$$d(p, q) = \sup \{ \lvert a - b \rvert : \lvert m \rvert \, \lvert a - b \rvert \le 1 \} = \frac{1}{\lvert m \rvert}$$

になる（Example 3）。2 点の間には「道」がないのに、$D$ が距離を決めている。

多様体 $M$ とこの 2 点の空間を掛け合わせた $M \times \{p, q\}$ は、「同じ形の 2 枚の写し」で、2 枚の間の距離が $1/\lvert m \rvert$ になる。素粒子物理学の標準模型は、時空とある有限の非可換な空間との積として書き直せる（Chamseddine–Connes 1997 のスペクトル作用の原理）。

### 4.5 有限の解像度では、関数が交換しなくなる

実際の計算では、固有関数を無限に使えず、低いほうから有限個だけを使う。円なら、フーリエモード $e^{in\theta}$（$n = -N, \dots, N$）の範囲に切り捨てる。

関数 $e^{i\theta}$ を掛けると、モード $n$ は $n + 1$ に移る。切り捨てた範囲でこれを行列 $S$ で表すと、いちばん上のモードは範囲の外に出て消える。$e^{-i\theta}$ を掛ける行列は $S^\top$ になる。関数としては $e^{i\theta} e^{-i\theta} = e^{-i\theta} e^{i\theta} = 1$ で交換するのに、

$$S S^\top - S^\top S = \mathrm{diag}(-1, 0, \dots, 0, 1) \ne 0$$

となる（Example 4）。範囲の両端で食い違いが出る。

つまり、**もとの空間が可換でも、有限の解像度で見ると関数どうしが交換しなくなる**。Connes–van Suijlekom（2021）は、このように切り捨てた世界を、代数ではなく**作用素系**という枠組みで扱う理論を作った。データから有限個の固有関数しか得られない状況と、自然につながる話題である。

## 5. 3 つの関係の見取り図

| | 多様体論 | スペクトル幾何学 | 非可換幾何学 |
|---|---|---|---|
| 主役 | 点と座標近傍、リーマン計量 | ラプラシアン（またはディラック作用素）の固有値 | 関数の代数 $A$ と作用素 $D$（スペクトル三つ組） |
| 問い | 長さ、測地線、曲率は | 音から形がどこまで分かるか | 点がなくても幾何学ができるか |
| 距離 | 曲線の長さの最小値 | 直接は扱わない（固有値から形の性質を読む） | $\lVert [D, f] \rVert \le 1$ の関数で測る |
| 可換か | 関数の代数は可換 | 可換（多様体の上の作用素） | 非可換も扱う。可換の場合は多様体に戻る |

関係を一言で言うと、

- **スペクトル幾何学は、多様体を「音」で調べる**。点と座標を使わず、作用素の固有値だけを見る
- **非可換幾何学は、多様体を「代数と作用素」で書き直し、それを非可換の場合に広げる**。スペクトル三つ組の $D$ は、スペクトル幾何学の作用素そのものなので、非可換幾何学はスペクトル幾何学を土台の 1 つにしている
- 可換な場合には、3 つは同じものを別の言葉で語っている（ゲルファント＝ナイマルクの定理と、コンヌの再構成定理）

## 6. GI 理論の地図では

GI 理論の地図 $M$ はコンパクトなリーマン多様体なので、スペクトル三つ組 $(C^\infty(M), L^2(\Lambda^\bullet M), d + d^*)$ を持つ。

**スペクトル幾何学が役に立つところ**: 地図の形の点検（島の数、低い固有値の並び、熱核の署名、ワイルの法則による次元の目安）。データの側と地図の側の「音」を比べ、地図が切れたり折れたりしていないかを見る。これは可換な場合の道具で足りる。

**非可換幾何学の言葉で読み直せるところ**:

- 地図の上の指標（関数）$f$ について、$\lVert [D, f] \rVert$ は、その指標がいちばん急に変わる場所での傾き（リプシッツ定数）に等しい
- コンヌの距離を、点ではなく**分布**（たとえば、ある年の国々の散らばり方）の間に使うと、可換な場合には、最適輸送の 1 次のワッサースタイン距離に一致する（D'Andrea–Martinetti 2010）
- データから得られるのは有限個の固有関数なので、第 4.5 節の切り捨ての話が、そのまま当てはまる

**比喩にとどまるところ**: 「2 つの局面（好況と不況など）を、第 4.4 節の 2 枚の写しとして扱う」という考えは魅力的だが、$m$ がデータから決まる量として定義されていないかぎり、数学の主張にはならない。局面の切り替わりを扱うには、確率で切り替えるマルコフ局面転換モデル（Hamilton 1989）など、すでに確立した方法がある。

## 7. よくある誤解

| 誤解 | 正しくは |
|---|---|
| スペクトル幾何学と「スペクトル系列」は同じ分野 | スペクトル系列（spectral sequence）はホモロジー代数の計算道具で、固有値の話ではない |
| 音から形が必ず分かる | 分からない例がある（第 3.2 節） |
| 熱核の展開から、どこがどれだけ曲がっているかが分かる | 分かるのは積分した量。場所ごとの情報を取り出すには、関数 $f$ を掛けた $\mathrm{Tr}(f \, e^{-t\Delta})$ を使う |
| どんな多様体にもディラック作用素が置ける | スピン構造が必要。なければ $d + d^*$ を使う |
| 非可換幾何学を使えば、データの非可換な構造が見える | GI 理論の地図は可換。非可換が自然に現れるのは、有限の解像度で切り捨てたとき（第 4.5 節）などに限られる |
| ゲルファント＝ナイマルクの定理は 1960 年代 | 1943 年 |

## 8. 練習問題

**問 1.** 半径 $r$ の円のラプラシアンの固有値を求め、ワイルの法則の右辺 $\omega_1 \cdot (2\pi r) \cdot \lambda^{1/2} / (2\pi)$ と、実際の $N(\lambda)$ を比べよ。

**問 2.** 第 4.4 節で、$D = \begin{pmatrix} 0 & m \\ m & 0 \end{pmatrix}$ と $f = \mathrm{diag}(a, b)$ の交換子 $[D, f]$ を計算し、その大きさが $\lvert m \rvert \lvert a - b \rvert$ になることを示せ。

**問 3.** 第 4.5 節で、モードを $\{0, 1\}$ の 2 つだけに切り捨てたとき、$S$ と $S^\top$ を 2 × 2 の行列で書き、$S S^\top$ と $S^\top S$ を求めよ。

**問 4.** 半径 1 の球面（$m = 2$、面積 $4\pi$、スカラー曲率 $R = 2$）について、熱核の展開の括弧の中の第 2 項 $\frac{t}{6} \int R \, dV$ を計算せよ。これをガウス＝ボネの定理 $\int K \, dA = 2\pi \chi$（$K = R/2$ はガウス曲率）と比べ、オイラー標数 $\chi = 2$ と合うことを確認せよ。

<details>
<summary>答え</summary>

**問 1.** 弧の長さ $s = r\theta$ で 2 回微分するので、固有値は $k^2 / r^2$（$k = 0, \pm 1, \dots$）。$k^2 / r^2 \le \lambda$ を満たす整数 $k$ は約 $2 r \sqrt{\lambda}$ 個。右辺は $2 \cdot 2\pi r \cdot \sqrt{\lambda} / (2\pi) = 2 r \sqrt{\lambda}$ で一致する。円が大きいほど、同じ $\lambda$ 以下の音が多い。

**問 2.** $Df = \begin{pmatrix} 0 & mb \\ ma & 0 \end{pmatrix}$、$fD = \begin{pmatrix} 0 & ma \\ mb & 0 \end{pmatrix}$ なので、$[D, f] = m(b - a) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$。右の行列は回転で、大きさ（作用素ノルム）は 1。よって $\lVert [D, f] \rVert = \lvert m \rvert \lvert a - b \rvert$。

**問 3.** 基底を $e_0, e_1$ とすると、$S e_0 = e_1$、$S e_1 = 0$（$e_2$ は範囲外）。$S = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$、$S^\top = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$。$S S^\top = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$、$S^\top S = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ で、等しくない。

**問 4.** $\frac{t}{6} \int R \, dV = \frac{t}{6} \cdot 2 \cdot 4\pi = \frac{4\pi t}{3}$。一方、$\int K \, dA = 1 \cdot 4\pi = 2\pi \chi$ から $\chi = 2$。一般の閉曲面では $\int R \, dA = 2 \int K \, dA = 4\pi\chi$ なので、第 2 項は $\frac{2\pi\chi t}{3}$ となり、前の係数 $\frac{1}{4\pi t}$ を掛けると $\chi / 6$。熱跡の展開の定数項にオイラー標数が現れ、球面では $1/3$ になる。

</details>

## 用語の対応表

| この解説の言葉 | 英語 |
|---|---|
| 多様体、座標近傍 | manifold, chart |
| リーマン計量、測地線、曲率 | Riemannian metric, geodesic, curvature |
| ラプラス＝ベルトラミ作用素 | Laplace–Beltrami operator |
| スペクトル | spectrum |
| 熱跡、熱核 | heat trace, heat kernel |
| チーガー定数 | Cheeger constant |
| 等スペクトル | isospectral |
| 可換な代数、非可換な代数 | commutative algebra, noncommutative algebra |
| スペクトル三つ組 | spectral triple |
| ディラック作用素、スピン構造 | Dirac operator, spin structure |
| コンヌの距離 | Connes distance (spectral distance) |
| 作用素系 | operator system |

## 参考文献

**多様体とスペクトル幾何学**

- M. Belkin, P. Niyogi. Convergence of Laplacian eigenmaps. *NIPS*, 2007.
- P. Buser. A note on the isoperimetric constant. *Annales scientifiques de l'École Normale Supérieure*, 15(2), 1982.
- J. Cheeger. A lower bound for the smallest eigenvalue of the Laplacian. In *Problems in Analysis*, Princeton University Press, 1970.
- C. Gordon, D. Webb, S. Wolpert. One cannot hear the shape of a drum. *Bulletin of the AMS*, 27(1), 1992.
- M. Kac. Can one hear the shape of a drum? *American Mathematical Monthly*, 73(4), 1966.
- A. Lichnerowicz. *Géométrie des groupes de transformations*. Dunod, 1958.
- H. P. McKean, I. M. Singer. Curvature and the eigenvalues of the Laplacian. *Journal of Differential Geometry*, 1, 1967.
- J. Milnor. Eigenvalues of the Laplace operator on certain manifolds. *PNAS*, 51(4), 1964.
- S. Minakshisundaram, Å. Pleijel. Some properties of the eigenfunctions of the Laplace-operator on Riemannian manifolds. *Canadian Journal of Mathematics*, 1, 1949.
- M. Obata. Certain conditions for a Riemannian manifold to be isometric with a sphere. *Journal of the Mathematical Society of Japan*, 14(3), 1962.
- T. Sunada. Riemannian coverings and isospectral manifolds. *Annals of Mathematics*, 121(1), 1985.
- H. Weyl. Über die asymptotische Verteilung der Eigenwerte. *Nachrichten der Königlichen Gesellschaft der Wissenschaften zu Göttingen*, 1911.

**非可換幾何学**

- A. H. Chamseddine, A. Connes. The spectral action principle. *Communications in Mathematical Physics*, 186, 1997（arXiv 1996）.
- A. Connes. *Noncommutative Geometry*. Academic Press, 1994.
- A. Connes. On the spectral characterization of manifolds. *Journal of Noncommutative Geometry*, 7(1), 2013（arXiv:0810.2088, 2008）.
- A. Connes, W. D. van Suijlekom. Spectral truncations in noncommutative geometry and operator systems. *Communications in Mathematical Physics*, 383, 2021.
- F. D'Andrea, P. Martinetti. A view on optimal transport from noncommutative geometry. *SIGMA*, 6, 2010.
- I. Gelfand, M. Neumark. On the imbedding of normed rings into the ring of operators in Hilbert space. *Matematicheskii Sbornik*, 12(54), 1943.
- W. D. van Suijlekom. *Noncommutative Geometry and Particle Physics*. Springer, 2015（入門に向く）.

**その他**

- J. D. Hamilton. A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 1989.

---

## 関連

- GI の地図の形を固有値で点検する（学部生向け）: [`../manifold_shape_diagnostics_undergraduate_guide/`](../manifold_shape_diagnostics_undergraduate_guide)
- 研究ノート（数式と限界、非可換幾何学からの位置づけ）: [`../manifold_shape_diagnostics/`](../manifold_shape_diagnostics)

## License

本稿は CC BY 4.0 のもとで公開される。引用時は以下を推奨:

```
@misc{etale2026gi_geometry_ug,
  author = {Étale Cohomology},
  title  = {多様体、スペクトル幾何学、非可換幾何学 ── 学部生のための見取り図},
  year   = {2026},
  note   = {Undergraduate guide}
}
```

Issues, PRs welcome.
