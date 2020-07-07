# 「Pythonによるベイズ統計学入門」正誤表

## 2020年6月20日改定

### 誤植

#### 1ページ、下から11行目

+ （誤）集めれられた
+ （正）集められた

#### 28ページ、(2.9)式の直後の行

+ （誤）つまり，$p(D|q)$はデータ$D$が観測される平均的な可能性と解釈される．
+ （正）つまり，$p(D)$はデータ$D$が観測される平均的な可能性と解釈される．

#### 69ページ、(3.17)式の右側の逆ガンマ分布

+ （誤）$\sigma^2\sim$
+ （正）$\sigma^2|D\sim$

#### コード3.3 pybayes\_conjugate\_regression.py 第79行

+ （誤）`nu_star = n + nu0`
+ （正）`nu_star = y.size + nu0`

#### 111ページ、PyMCの仕様の変更に伴う注意点

バージョン3.7より仕様が変更されて`pm.traceplot()`はオプション`ax`を受け付けなくなった．したがって，バージョン3.7以降のPyMCを使用する際には最初の`plt.subplots()`の部分を削除し，続く`pm.traceplot()`内の`ax=ax1`も削除しなければならない．

また，`pm.plot_posterior()`もオプション`kde_plot`を無視するようになった．デフォルトでは滑らかなカーネル推定が描かれるが(`kind='kde'`)，ヒストグラムを描く際には引数に`kind='hist'`を指定しなければならない．

#### 120ページ、上から5行目

+ （誤）(4.2) 式のコーシー分布を
+ （正）(4.23) 式のコーシー分布を

#### 138ページ、上から3行目

+ （誤）Kitagawand Gersch (1984)
+ （正）Kitagawa and Gersch (1984)

### PyMCの仕様の変更に伴う修正

#### コード6.1`pybayes_gibbs_gaussian.py`とコード6.2`pybayes_gibbs_regression.py`において

+ `pm.mc_error()`を`pm.mcse()`へ
+ `pm.gelman_rubin()`を`pm.rhat()`へ

と置き換える．特に`pm.mcse()`は`pm.mc_error()`から用法も変更されているので注意．
