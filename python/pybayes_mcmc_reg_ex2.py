# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   SciPyのLinalgモジュールの読み込み
import scipy.linalg as la
#   PyMCの読み込み
import pymc as pm
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
#%% 回帰モデルの係数と誤差項の分散の事後分布の設定（正規分布＋逆ガンマ分布）
b0 = np.zeros(2)
A0 = 0.2 * np.eye(2)
nu0 = 5.0
lam0 = 7.0
sd0 = np.sqrt(np.diag(la.inv(A0)))
regresssion_normal_invgamma = pm.Model()
with regresssion_normal_invgamma:
    sigma2 = pm.InverseGamma('sigma2', alpha=0.5*nu0, beta=0.5*lam0)
    a = pm.Normal('a', mu=0.0, sigma=sd0[0])
    b = pm.Normal('b', mu=0.0, sigma=sd0[1])
    y_hat = a + b * x
    likelihood = pm.Normal('y', mu=y_hat, sigma=pm.math.sqrt(sigma2),
                           observed=y)
#%% 事後分布からのサンプリング
n_draws = 5000
n_chains = 4
n_tune = 1000
with regresssion_normal_invgamma:
    trace = pm.sample(draws=n_draws, chains=n_chains, tune=n_tune,
                      random_seed=123)
    print(pm.summary(trace))
#%% 事後分布のグラフの作成
k = b0.size
param_names = ['a', 'b', 'sigma2']
labels = ['$\\alpha$', '$\\beta$', '$\\sigma^2$']
fig, ax = plt.subplots(k+1, 2, num=1, figsize=(8, 1.5*(k+1)), facecolor='w')
for index in range(k+1):
    mc_trace = trace.posterior[param_names[index]].values.flatten()
    if index < k:
        x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.norm.pdf(x, loc=b0[index], scale=sd0[index])
    else:
        x_min = 0.0
        x_max = mc_trace.max() + 0.2 * np.abs(mc_trace.max())
        x = np.linspace(x_min, x_max, 250)
        prior = st.invgamma.pdf(x, 0.5*nu0, scale=0.5*lam0)
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
plt.savefig('pybayes_fig_mcmc_reg_ex2.png', dpi=300)
plt.show()
