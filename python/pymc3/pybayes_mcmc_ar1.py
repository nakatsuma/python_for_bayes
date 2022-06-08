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
elif sys.platform.startswith('darwin' ):
    FontPath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
elif sys.platform.startswith('linux'):
    FontPath = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
else:
    sys.exit('このPythonコードが対応していないOSを使用しています．')
jpfont = FontProperties(fname=FontPath)
#%% ノイズを含むAR(1)過程からデータを生成
n = 500
np.random.seed(99)
x = np.empty(n)
x[0] = st.norm.rvs() # 定常分布の分散 = 0.19/(1 - 0.9**2) = 1.0
for t in range(1, n):
    x[t] = 0.9 * x[t-1] + st.norm.rvs(scale=np.sqrt(0.19))
y = x + st.norm.rvs(scale=0.5, size=n)
#%% 事後分布の設定
ar1_model = pm.Model()
with ar1_model:
    sigma = pm.HalfCauchy('sigma', beta=1.0)
    rho = pm.Uniform('rho', lower=-1.0, upper=1.0)
    omega = pm.HalfCauchy('omega', beta=1.0)
    ar1 = pm.AR('ar1', rho, sigma=omega, shape=n,
                init=pm.Normal.dist(sigma=omega/pm.math.sqrt(1 - rho**2)))
    observation = pm.Normal('y', mu=ar1, sigma=sigma, observed=y)
#%% 事後分布からのサンプリング
n_draws = 5000
n_chains = 4
n_tune = 1000
with ar1_model:
    trace = pm.sample(draws=n_draws, chains=n_chains, tune=n_tune,
                      random_seed=123)
    param_names = ['sigma', 'rho', 'omega']
    print(az.summary(trace, var_names=param_names))
#%% 事後分布のグラフの作成
labels = ['$\\sigma$', '$\\rho$', '$\\omega$']
k = len(labels)
fig, ax = plt.subplots(k, 2, num=1, figsize=(8, 1.5*k), facecolor='w')
for index in range(k):
    mc_trace = trace[param_names[index]]
    x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
    x_max =  mc_trace.max() + 0.2 * np.abs(mc_trace.max())
    x = np.linspace(x_min, x_max, 250)
    posterior = st.gaussian_kde(mc_trace).evaluate(x)
    ax[index, 0].plot(mc_trace, 'k-', linewidth=0.1)
    ax[index, 0].set_xlim(1, n_draws*n_chains)
    ax[index, 0].set_ylabel(labels[index], fontproperties=jpfont)
    ax[index, 1].plot(x, posterior, 'k-')
    ax[index, 1].set_xlim(x_min, x_max)
    ax[index, 1].set_ylim(0, 1.1*posterior.max())
    ax[index, 1].set_ylabel('確率密度', fontproperties=jpfont)
ax[k-1, 0].set_xlabel('乱数系列', fontproperties=jpfont)
ax[k-1, 1].set_xlabel('周辺事後分布', fontproperties=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_mcmc_ar1.png', dpi=300)
plt.show()
