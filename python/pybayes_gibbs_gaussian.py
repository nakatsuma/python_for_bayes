# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   Pandasの読み込み
import pandas as pd
#   ArviZの読み込み
import arviz as az
#   MatplotlibのPyplotモジュールの読み込み
import matplotlib.pyplot as plt
#   tqdmからプログレスバーの関数を読み込む
from tqdm import trange
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
#%% ギブズ・サンプラーによる正規分布の平均と分散に関するベイズ推論
#   正規分布の平均と分散のギブズ・サンプラー
def gibbs_gaussian(data, iterations, mu0, tau0, nu0, lam0):
    """
        入力
        data:       データ
        iterations: 反復回数
        mu0:        平均の事前分布（正規分布）の平均
        tau0:       平均の事前分布（正規分布）の標準偏差
        nu0:        分散の事前分布（逆ガンマ分布）の形状パラメータ
        lam0:       分散の事前分布（逆ガンマ分布）の尺度パラメータ
        出力
        runs:       モンテカルロ標本
    """
    n = data.size
    sum_data = data.sum()
    mean_data = sum_data / n
    variance_data = data.var()
    inv_tau02 = 1.0 / tau0**2
    mu0_tau02 = mu0 * inv_tau02
    a = 0.5 * (n + nu0)
    c = n * variance_data + lam0
    sigma2 = variance_data
    runs = np.empty((iterations, 2))
    for idx in trange(iterations):
        variance_mu = 1.0 / (n / sigma2 + inv_tau02)
        mean_mu = variance_mu * (sum_data / sigma2 + mu0_tau02)
        mu = st.norm.rvs(loc=mean_mu, scale=np.sqrt(variance_mu))
        b = 0.5 * (n * (mu - mean_data)**2 + c)
        sigma2 = st.invgamma.rvs(a, scale=b)
        runs[idx, 0] = mu
        runs[idx, 1] = sigma2
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
    param_string = ['平均 $\\mu$', '分散 $\\sigma^2$']
    return pd.DataFrame(stats, index=param_string, columns=stats_string)
#%% 正規分布からのデータ生成
mu = 1.0
sigma = 2.0
n = 50
np.random.seed(99)
data = st.norm.rvs(loc=mu, scale=sigma, size=n)
#%% ギブズ・サンプラーの実行
mu0 = 0.0
tau0 = 1.0
nu0 = 5.0
lam0 = 7.0
prob = 0.95
burnin = 2000
samplesize = 20000
iterations = burnin + samplesize
np.random.seed(123)
runs = gibbs_gaussian(data, iterations, mu0, tau0, nu0, lam0)
#%% 事後統計量の計算
batch = 4
results = mcmc_stats(runs, burnin, prob, batch)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
fig, ax = plt.subplots(2, 2, num=1, figsize=(8, 3), facecolor='w')
labels = ['$\\mu$', '$\\sigma^2$']
for index in range(2):
    mc_trace = runs[burnin:, index]
    if index == 0:
        x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.norm.pdf(x, loc=mu0, scale=tau0)
    else:
        x_min = 0.0
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.invgamma.pdf(x, 0.5*nu0, scale=0.5*lam0)
        ax[index, 0].set_xlabel('乱数系列', fontproperties=jpfont)
        ax[index, 1].set_xlabel('周辺事後分布', fontproperties=jpfont)
    posterior = st.gaussian_kde(mc_trace).evaluate(x)
    ax[index, 0].plot(mc_trace, 'k-', linewidth=0.1)
    ax[index, 0].set_xlim(1, samplesize)
    ax[index, 0].set_ylabel(labels[index], fontproperties=jpfont)
    ax[index, 1].plot(x, posterior, 'k-', label='事後分布')
    ax[index, 1].plot(x, prior, 'k:', label='事前分布')
    ax[index, 1].set_xlim(x_min, x_max)
    ax[index, 1].set_ylim(0, 1.1*posterior.max())
    ax[index, 1].set_ylabel('確率密度', fontproperties=jpfont)
    ax[index, 1].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_gibbs_gaussian.png', dpi=300)
plt.show()
