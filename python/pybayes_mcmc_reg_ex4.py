# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   PyMCの読み込み
import pymc3 as pm
#   ArviZの読み込み
import arviz as az
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
#%% 回帰モデルからのデータ生成
n = 50
np.random.seed(99)
u = st.norm.rvs(scale=0.7, size=n)
x = st.uniform.rvs(loc=-np.sqrt(3.0), scale=2.0*np.sqrt(3.0), size=n)
y = 1.0 + 2.0 * x + u
#%% 回帰モデルの係数と誤差項の分散の事後分布の設定（ラプラス＋半コーシー分布）
b0 = np.zeros(2)
tau_coef = np.ones(2)
tau_sigma = 1.0
regression_laplace_halfcauchy = pm.Model()
with regression_laplace_halfcauchy:
    sigma = pm.HalfCauchy('sigma', beta=tau_sigma)
    a = pm.Laplace('a', mu=b0[0], b=tau_coef[0])
    b = pm.Laplace('b', mu=b0[1], b=tau_coef[1])
    y_hat = a + b * x
    likelihood = pm.Normal('y', mu=y_hat, sigma=sigma, observed=y)
#%% 事後分布からのサンプリング
n_draws = 5000
n_chains = 4
n_tune = 1000
with regression_laplace_halfcauchy:
    trace = pm.sample(draws=n_draws, chains=n_chains, tune=n_tune,
                      random_seed=123)
    print(az.summary(trace))
#%% 事後分布のグラフの作成
k = b0.size
param_names = ['a', 'b', 'sigma']
labels = ['$\\alpha$', '$\\beta$', '$\\sigma$']
fig, ax = plt.subplots(k+1, 2, num=1, figsize=(8, 1.5*(k+1)), facecolor='w')
for index in range(k+1):
    mc_trace = trace[param_names[index]]
    if index < k:
        x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.laplace.pdf(x, loc=b0[index], scale=tau_coef[index])
    else:
        x_min = 0.0
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.halfcauchy.pdf(x, scale=tau_sigma)
        ax[index, 0].set_xlabel('乱数系列', fontproperties=jpfont)
        ax[index, 1].set_xlabel('パラメータの分布', fontproperties=jpfont)
    ax[index, 0].plot(mc_trace, 'k-', linewidth=0.1)
    ax[index, 0].set_xlim(1, n_draws*n_chains)
    ax[index, 0].set_ylabel(labels[index], fontproperties=jpfont)
    posterior = st.gaussian_kde(mc_trace).evaluate(x)
    ax[index, 1].plot(x, posterior, 'k-', label='事後分布')
    ax[index, 1].plot(x, prior, 'k:', label='事前分布')
    ax[index, 1].set_xlim(x_min, x_max)
    ax[index, 1].set_ylim(0, 1.1*posterior.max())
    ax[index, 1].set_ylabel('確率密度', fontproperties=jpfont)
    ax[index, 1].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_mcmc_reg_ex4.png', dpi=300)
plt.show()
