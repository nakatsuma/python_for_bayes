# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのlinalgモジュールの読み込み
import scipy.linalg as la
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   Pandasの読み込み
import pandas as pd
#   ArviZの読み込み
import arviz as az
#   tqdmからプログレスバーの関数を読み込む
from tqdm import trange
#   MatplotlibのPyplotモジュールの読み込み
import matplotlib.pyplot as plt
#   日本語フォントの設定
from matplotlib.font_manager import FontProperties
import sys
if sys.platform.startswith('win'):
    FontPath = 'C:\\Windows\\Fonts\\meiryo.ttc'
elif sys.platform.startswith('darwin'):
    FontPath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
elif sys.platform.startswith('linux'):
    FontPath = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
else:
    print('このPythonコードが対応していないOSを使用しています．')
    sys.exit()
jpfont = FontProperties(fname=FontPath)
#%% ギブズ・サンプラーによる回帰モデルのパラメータに関するベイズ推論
#   回帰モデルの回帰係数と誤差項の分散のギブズ・サンプラー
def gibbs_regression(y, X, iterations, b0, A0, nu0, lam0):
    """
        入力
        y:          被説明変数
        X:          説明変数
        iterations: 反復回数
        b0:         回帰係数の事前分布（多変量正規分布）の平均
        A0:         回帰係数の事前分布（多変量正規分布）の精度行列
        nu0:        誤差項の分散の事前分布（逆ガンマ分布）の形状パラメータ
        lam0:       誤差項の分散の事前分布（逆ガンマ分布）の尺度パラメータ
        出力
        runs:   モンテカルロ標本
    """
    n, k = X.shape
    XX = X.T.dot(X)
    Xy = X.T.dot(y)
    b_ols = la.solve(XX, Xy)
    rss = np.square(y - X.dot(b_ols)).sum()
    lam_hat = rss + lam0
    nu_star = 0.5 * (n + nu0)
    A0b0 = A0.dot(b0)
    sigma2 = rss / (n - k)
    runs = np.empty((iterations, k + 1))
    for idx in trange(iterations):
        cov_b = la.inv(XX / sigma2 + A0)
        mean_b = cov_b.dot(Xy / sigma2 + A0b0)
        b = st.multivariate_normal.rvs(mean=mean_b, cov=cov_b)
        diff = b - b_ols
        lam_star = 0.5 * (diff.T.dot(XX).dot(diff) + lam_hat)
        sigma2 = st.invgamma.rvs(nu_star, scale=lam_star)
        runs[idx, :-1] = b
        runs[idx, -1] = sigma2
    return runs
#   モンテカルロ標本からの事後統計量の計算
def mcmc_stats(runs, burnin, prob, batch):
    """
        入力
        runs:   モンテカルロ標本
        burnin: バーンインの回数
        prob:   区間確率 (0 < prob < 1)
        batch:  乱数系列の分割数
        出力
        事後統計量のデータフレーム
    """
    traces = runs[burnin:, :]
    n = traces.shape[0] // batch
    k = traces.shape[1]
    alpha = 100 * (1.0 - prob)
    post_mean = np.mean(traces, axis=0)
    post_median = np.median(traces, axis=0)
    post_sd = np.std(traces, axis=0)
    mc_err = [az.mcse(traces[:, i].reshape((n, batch), order='F')).item(0) \
              for i in range(k)]
    ci_lower = np.percentile(traces, 0.5 * alpha, axis=0)
    ci_upper = np.percentile(traces, 100 - 0.5 * alpha, axis=0)
    hpdi = az.hdi(traces, prob)
    rhat = [az.rhat(traces[:, i].reshape((n, batch), order='F')).item(0) \
            for i in range(k)]
    stats = np.vstack((post_mean, post_median, post_sd, mc_err,
                       ci_lower, ci_upper, hpdi.T, rhat)).T
    stats_string = ['平均', '中央値', '標準偏差', '近似誤差',
                    '信用区間（下限）', '信用区間（上限）',
                    'HPDI（下限）', 'HPDI（上限）', '$\\hat R$']
    param_string = ['$\\beta_{0:<d}$'.format(i+1) for i in range(k-1)]
    param_string.append('$\\sigma^2$')
    return pd.DataFrame(stats, index=param_string, columns=stats_string)
#%% 回帰モデルからのデータの生成
n = 50
np.random.seed(99)
u = st.norm.rvs(scale=0.7, size=n)
x1 = st.uniform.rvs(loc=-np.sqrt(3.0), scale=2.0*np.sqrt(3.0), size=n)
x2 = st.uniform.rvs(loc=-np.sqrt(3.0), scale=2.0*np.sqrt(3.0), size=n)
y = 1.0 + 2.0 * x1 - x2 + u
X = np.stack((np.ones(n), x1, x2), axis=1)
#%% ギブズ・サンプラーの実行
k = X.shape[1]
b0 = np.zeros(k)
A0 = 0.2 * np.eye(k)
nu0 = 5.0
lam0 = 7.0
sd0 = np.sqrt(np.diag(la.inv(A0)))
prob = 0.95
burnin = 2000
samplesize = 20000
iterations = burnin + samplesize
np.random.seed(123)
runs = gibbs_regression(y, X, iterations, b0, A0, nu0, lam0)
#%% 事後統計量の計算
batch = 4
results = mcmc_stats(runs, burnin, prob, batch)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
fig, ax = plt.subplots(k+1, 2, num=1, figsize=(8, 1.5*(k+1)), facecolor='w')
for index in range(k+1):
    mc_trace = runs[burnin:, index]
    if index < k:
        x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.norm.pdf(x, loc=b0[index], scale=sd0[index])
        y_label = '$\\beta_{:<d}$'.format(index+1)
    else:
        x_min = 0.0
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.invgamma.pdf(x, 0.5*nu0, scale=0.5*lam0)
        y_label = '$\\sigma^2$'
        ax[index, 0].set_xlabel('乱数系列', fontproperties=jpfont)
        ax[index, 1].set_xlabel('パラメータの分布', fontproperties=jpfont)
    posterior = st.gaussian_kde(mc_trace).evaluate(x)
    ax[index, 0].plot(mc_trace, 'k-', linewidth=0.1)
    ax[index, 0].set_xlim(1, samplesize)
    ax[index, 0].set_ylabel(y_label, fontproperties=jpfont)
    ax[index, 1].plot(x, posterior, 'k-', label='事後分布')
    ax[index, 1].plot(x, prior, 'k:', label='事前分布')
    ax[index, 1].set_xlim(x_min, x_max)
    ax[index, 1].set_ylim(0, 1.1*posterior.max())
    ax[index, 1].set_ylabel('確率密度', fontproperties=jpfont)
    ax[index, 1].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_gibbs_regression.png', dpi=300)
plt.show()
