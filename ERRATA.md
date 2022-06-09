# 「Pythonによるベイズ統計学入門」正誤表

## 2022年2月21日改定

### PyMCの仕様変更に伴う修正

#### ArviZの必須化

バージョン3.11より，記述統計や作図に関連する関数はPyMCから[ArviZ](https://arviz-devs.github.io/arviz/index.html)に移された．そのためArviZを

```IPython
import arviz as az
```

と読み込み，ArviZの関数を呼び出す必要が生じる．

#### 111ページ

+ `pm.traceplot()`の代わりに`az.plot_trace()`を，`pm.plot_posterior()`の代わりに`az.plot_posterior()`を使わなければならない．
+ さらに，`pm.traceplot()`（つまり`az.plot_trace()`）はオプション`ax`を受け付けなくなった．したがって，最初の`plt.subplots()`の部分を削除し，続く`pm.traceplot()`内の`ax=ax1`も削除しなければならない．
+ また，`pm.plot_posterior()`（つまり`az.plot_posterior()`）もオプション`kde_plot`を無視するようになった．デフォルトでは滑らかなカーネル推定が描かれるが(`kind='kde'`)，ヒストグラムを描く際には引数に`kind='hist'`を指定しなければならない．

#### コード6.1`pybayes_gibbs_gaussian.py`とコード6.2`pybayes_gibbs_regression.py`において

+ `pm.mc_error()`を`az.mcse()`へ
+ `pm.hpd()`を`az.hdi()`へ
+ `pm.gelman_rubin()`を`az.rhat()`へ

と置き換える．そして、`pm.hpd()`（つまり`az.hdi()`）内の`1.0 - prob`を`prob`に変更する．さらに`az.mcse()`は`pm.mc_error()`から用法も変更されているので注意．

### 誤植

#### 1ページ、下から11行目

+ （誤）集めれられた
+ （正）集められた

#### 28ページ、(2.9)式の直後の行

+ （誤）つまり，$p(D|q)$はデータ$D$が観測される平均的な可能性と解釈される．
+ （正）つまり，$p(D)$はデータ$D$が観測される平均的な可能性と解釈される．

#### 54ページ、(2.38)式の下の2つの式

+ （誤）

$$\nabla_\delta R(\delta^*|D) = 2P(\delta^*|\mathbf{x}) - 1 = 0,\quad\nabla_\delta^2 R(\delta^*|D) = 2p(\delta^*|\mathbf{x}) > 0,$$

$$P(\delta^*|\mathbf{x}) = \frac12,$$

+ （正）

$$\nabla_\delta R(\delta^*|D) = 2P(\delta^*|D) - 1 = 0,\quad\nabla_\delta^2 R(\delta^*|D) = 2p(\delta^*|D) > 0,$$

$$P(\delta^*|D) = \frac12,$$

#### 55ページ、最後の2つの式

+ （誤）

$$\Pr\{q=q_0|D\}= \frac{q_0 p(D|q_0)}{q_0 p(D|q_0)+(1-q_0)\int_0^1p(D|q)f(q)dq}, \\ \Pr\{q\ne q_0|D\} = \frac{(1-q_0)\int_0^1 p(D|q)f(q)dq}{q_0 p(D|q_0)+(1-q_0)\int_0^1p(D|q)f(q)dq},$$

+ （正）

$$\Pr\{q=q_0|D\} = \frac{p_0 p(D|q_0)}{p_0 p(D|q_0)+(1-p_0)\int_0^1p(D|q)f(q)dq}, \\ \Pr\{q\ne q_0|D\} = \frac{(1-p_0)\int_0^1 p(D|q)f(q)dq}{p_0 p(D|q_0)+(1-p_0)\int_0^1p(D|q)f(q)dq},$$

#### 56ページ、(2.43)式

+ （誤）

$$\frac{\Pr\{q=q_0|D\}}{\Pr\{q\ne q_0|D\}} = \frac{q_0}{1-q_0} \times \frac{p(D|q_0)}{\int_0^1 p(D|q)f(q)dq},$$

+ （正）

$$\frac{\Pr\{q=q_0|D\}}{\Pr\{q\ne q_0|D\}} = \frac{p_0}{1-p_0} \times \frac{p(D|q_0)}{\int_0^1 p(D|q)f(q)dq},$$


#### 61ページ、(3.6)式の右辺の2行目

+ （誤）$x^{\alpha_0-1}$
+ （正）$\lambda^{\alpha_0-1}$

#### 69ページ、(3.17)式の右側の逆ガンマ分布

+ （誤）$\sigma^2\sim$
+ （正）$\sigma^2|D\sim$

#### コード3.3 pybayes\_conjugate\_regression.py 第79行

+ （誤）`nu_star = n + nu0`
+ （正）`nu_star = y.size + nu0`

#### 120ページ、上から5行目

+ （誤）(4.2) 式のコーシー分布を
+ （正）(4.23) 式のコーシー分布を

#### 138ページ、上から3行目

+ （誤）Kitagawand Gersch (1984)
+ （正）Kitagawa and Gersch (1984)

#### 176ページ、(6.8)式の2行目の積分内の関数

+ （誤）$f_t(x_{t-1})$
+ （正）$f_{t-1}(x_{t-1})$
