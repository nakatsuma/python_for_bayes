# 中妻照雄「Pythonによるベイズ統計学入門」

[朝倉書店ウェブサイト](https://www.asakura.co.jp/books/isbn/978-4-254-12898-7/ "朝倉書店ウェブサイト")

---

+ [正誤表](ERRATA.md)
+ [PythonとPyMCのインストール手順](#python%e3%81%a8pymc%e3%81%ae%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%bc%e3%83%ab%e6%89%8b%e9%a0%86)
  + [ステップ1: Anacondaのインストール](#%E3%82%B9%E3%83%86%E3%83%83%E3%83%971-Anaconda%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
  + [ステップ2: PyMCを実行する環境の設定](#%E3%82%B9%E3%83%86%E3%83%83%E3%83%972-PyMC%E3%82%92%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B%E7%92%B0%E5%A2%83%E3%81%AE%E8%A8%AD%E5%AE%9A)
+ [Jupyter Notebookを始める方法](#Jupyter-Notebook%E3%82%92%E5%A7%8B%E3%82%81%E3%82%8B%E6%96%B9%E6%B3%95)
  + [方法1: Anaconda NavigatorからJupyter Notebookを起動する方法](#%E6%96%B9%E6%B3%951-Anaconda-Navigator%E3%81%8B%E3%82%89Jupyter-Notebook%E3%82%92%E8%B5%B7%E5%8B%95%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95)
  + [方法2: CLIから起動する方法](#%E6%96%B9%E6%B3%952-CLI%E3%81%8B%E3%82%89%E8%B5%B7%E5%8B%95%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95)
+ [Pythonコード](#Python%E3%82%B3%E3%83%BC%E3%83%89)
  + [第2章](#%E7%AC%AC2%E7%AB%A0)
  + [第3章](#%E7%AC%AC3%E7%AB%A0)
  + [第4章](#%E7%AC%AC4%E7%AB%A0)
  + [第5章](#%E7%AC%AC5%E7%AB%A0)
  + [第6章](#%E7%AC%AC6%E7%AB%A0)

---

## 正誤表（2020年6月20日改定）

+ [ERRATA.md](ERRATA.md)

## PythonとPyMCのインストール手順

### ステップ1: Anacondaのインストール

1. 古いAnacondaがインストールされているときは、この[手順](https://docs.anaconda.com/anaconda/install/uninstall/)でアンインストールしておく。

2. Anacondaのインストーラー (Windows, macOS or Linux) を[ここ](https://www.anaconda.com/distribution/)から入手する.

3. ダウンロードしたインストーラーをダブルクリックして Anacondaのインストールを行う。

### ステップ2: PyMCを実行する環境の設定

`Anaconda Powershell Prompt` (Windows) あるいは`Terminal` (macos, Linux) を立ち上げて、

```IPython
(base) PS C:\Users\Thomas> conda create -n bayes jupyterlab seaborn spyder
```

とする。続けて

```IPython
(base) PS C:\Users\Thomas> conda activate bayes
```

とすると、以下のようにプロンプトが変わる。

```IPython
(bayes) PS C:\Users\Thomas>
```

続いてpipでPyMCをインストールする。

```IPython
(bayes) PS C:\Users\Thomas> pip install pymc3
```

最後に

```IPython
(bayes) PS C:\Users\Thomas> python -m ipykernel install --user --name bayes --display-name "Python (Bayes)"
```

とすれば、環境の設定が完了する。

---

## Jupyter Notebookを始める方法

### 方法1: Anaconda NavigatorからJupyter Notebookを起動する方法

`Anaconda Navigator`を`Start Menu` (Windows) か `Launchpad` (macOS) から起動する。 あるいは、`Anaconda Powershell Prompt` (Windows) か `Terminal` (macOS, Linux) を立ち上げて、

```IPython
(base) PS C:\Users\Thomas> anaconda-navigator
```

としてもよい。そして、`Anaconda Navigator`で`Jupyter Notebook`の`Launch`ボタンをクリックする。

### 方法2: CLIから起動する方法

`Anaconda Powershell Prompt` (Windows) か `Terminal` (macOS, Linux) を立ち上げて、

```IPython
(base) PS C:\Users\Thomas> conda activate bayes
(bayes) PS C:\Users\Thomas> jupyter notebook
```

とする。

方法1あるいは方法2を実行すると、規定のブラウザーが立ち上がり、Jupyter Notebookが起動する。その画面の右上にある`New`のプルダウンメニューの中にある`Python (Bayes)`を選んでNotebookを開始すればよい。

**注意:** `New`のプルダウンメニューの中にある`Python 3`を選んでNotebookを開始すると、PyMCを使用することができない。

## Pythonコード

### 第2章

+ コード2.1 ベルヌーイ分布の成功確率の事前分布: [pybayes\_beta\_prior.py](python/pybayes_beta_prior.py)
+ コード2.2 ベータ分布のグラフ: [pybayes\_beta\_distribution.py](python/pybayes_beta_distribution.py)
+ コード2.3 ベルヌーイ分布の成功確率の事後分布と事後統計量: [pybayes\_conjugate\_bernoulli.py](python/pybayes_conjugate_bernoulli.py)
+ コード2.4 損失関数と区間推定の図示: [pybayes\_posterior\_inference.py](python/pybayes_posterior_inference.py)

### 第3章

+ コード3.1 ポアソン分布の&lambda;の事後分布と事後統計量: [pybayes\_conjugate\_poisson.py](python/pybayes_conjugate_poisson.py)
+ コード3.2 正規分布の平均と分散の事後分布と事後統計量: [pybayes\_conjugate\_gaussian.py](python/pybayes_conjugate_gaussian.py)
+ コード3.3 回帰係数と誤差項の分散の事後分布と事後統計量: [pybayes\_conjugate\_regression.py](python/pybayes_conjugate_regression.py)
+ コード3.4 ポアソン分布とガンマ分布の例: [pybayes\_poisson\_gamma.py](python/pybayes_poisson_gamma.py)
+ コード3.5 正規分布の例: [pybayes\_gaussian\_distribution.py](python/pybayes_gaussian_distribution.py)
+ コード3.6 逆ガンマ分布とt分布の例: [pybayes\_invgamma\_t.py](python/pybayes_invgamma_t.py)

### 第4章

+ コード4.1 回帰モデルのベイズ分析(自然共役事前分布): [pybayes\_mcmc\_reg\_ex1.py](python/pybayes_mcmc_reg_ex1.py)
+ コード4.2 回帰モデルのベイズ分析(正規分布 + 逆ガンマ分布): [pybayes\_mcmc\_reg\_ex2.py](python/pybayes_mcmc_reg_ex2.py)
+ コード4.3 回帰モデルのベイズ分析(重回帰モデル): [pybayes\_mcmc\_reg\_ex3.py](python/pybayes_mcmc_reg_ex3.py)
+ コード4.4 回帰モデルのベイズ分析(ラプラス分布 + 半コーシー分布): [pybayes\_mcmc\_reg\_ex4.py](python/pybayes_mcmc_reg_ex4.py)
+ コード4.5 ロジット・モデルのベイズ分析: [pybayes\_mcmc\_logit.py](python/pybayes_mcmc_logit.py)
+ コード4.6 プロビット・モデルのベイズ分析: [pybayes\_mcmc\_probit.py](python/pybayes_mcmc_probit.py)
+ コード4.7 ポアソン回帰モデルのベイズ分析: [pybayes\_mcmc\_poisson.py](python/pybayes_mcmc_poisson.py)

### 第5章

+ コード5.1 ノイズを含むAR(1)過程: [pybayes\_mcmc\_ar1.py](python/pybayes_mcmc_ar1.py)
+ コード5.2 使用電力量のトレンドと季節変動: [pybayes\_mcmc\_decomp.py](python/pybayes_mcmc_decomp.py)
+ コード5.3 確率的ボラティリティ・モデル: [pybayes\_mcmc\_sv.py](python/pybayes_mcmc_sv.py)
+ コード5.4 時系列データのプロット: [pybayes\_timeseries\_data.py](python/pybayes_timeseries_data.py)
+ ドル円為替レート日次データ: [dollaryen.csv](python/dollaryen.csv)
+ 電灯電力需要実績月報・用途別使用電力量・販売電力合計・10社計: [electricity.csv](python/electricity.csv)

### 第6章

+ コード6.1 正規分布に対するギブズ・サンプラー: [pybayes\_gibbs\_gaussian.py](python/pybayes_gibbs_gaussian.py)
+ コード6.2 回帰モデルに対するギブズ・サンプラー: [pybayes\_gibbs\_regression.py](python/pybayes_gibbs_regression.py)
