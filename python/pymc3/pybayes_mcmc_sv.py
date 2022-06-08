# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   Pandasの読み込み
import pandas as pd
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
#   PandasからMatplotlibへのコンバーター
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%% ドル円為替レート日次データの読み込み
"""
    The Pacific Exchange Rate Serviceより入手
    http://fx.sauder.ubc.ca/data.html
"""
data = pd.read_csv('dollaryen.csv', index_col=0)
y = 100 * np.diff(np.log(data.values.ravel()))
n = y.size
series_date = pd.to_datetime(data.index[1:])
#%% SVモデルの設定
sv_model = pm.Model()
with sv_model:
    nu = pm.Exponential('nu', 0.2)
    sigma = pm.HalfCauchy('sigma', beta=1.0)
    rho = pm.Uniform('rho', lower=-1.0, upper=1.0)
    omega = pm.HalfCauchy('omega', beta=1.0)
    log_vol = pm.AR('log_vol', rho, sigma=omega, shape=n,
                    init=pm.Normal.dist(sigma=omega/pm.math.sqrt(1 - rho**2)))
    observation = pm.StudentT('y', nu, sigma=sigma*pm.math.exp(log_vol),
                              observed=y)
#%% 事後分布からのサンプリング
n_draws = 5000
n_chains = 4
n_tune = 2000
with sv_model:
    trace = pm.sample(draws=n_draws, chains=n_chains, tune=n_tune,
                      target_accept=0.95, max_treedepth=50, random_seed=123)
    param_names = ['nu', 'sigma', 'rho', 'omega']
    print(az.summary(trace, var_names=param_names))
#%% 事後分布のグラフの作成
labels = ['$\\nu$', '$\\sigma$', '$\\rho$', '$\\omega$']
k = len(labels)
x_minimum = [ 3.0, 0.15, 0.9, 0.02]
x_maximum = [17.0, 0.85, 1.0, 0.16]
fig1, ax1 = plt.subplots(k, 2, num=1, figsize=(8, 1.5*k), facecolor='w')
for index in range(k):
    mc_trace = trace[param_names[index]]
    x_min = x_minimum[index]
    x_max = x_maximum[index]
    x = np.linspace(x_min, x_max, 250)
    posterior = st.gaussian_kde(mc_trace).evaluate(x)
    ax1[index, 0].plot(mc_trace, 'k-', linewidth=0.1)
    ax1[index, 0].set_xlim(1, n_draws*n_chains)
    ax1[index, 0].set_ylabel(labels[index], fontproperties=jpfont)
    ax1[index, 1].plot(x, posterior, 'k-')
    ax1[index, 1].set_xlim(x_min, x_max)
    ax1[index, 1].set_ylim(0, 1.1*posterior.max())
    ax1[index, 1].set_ylabel('確率密度', fontproperties=jpfont)
ax1[k-1, 0].set_xlabel('乱数系列', fontproperties=jpfont)
ax1[k-1, 1].set_xlabel('周辺事後分布', fontproperties=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_sv_posterior.png', dpi=300)
plt.show()
#%% ボラティリティのプロット
vol = np.median(np.tile(trace['sigma'],
                (n, 1)).T * np.exp(trace['log_vol']), axis=0)
fig2 = plt.figure(num=2, facecolor='w')
plt.plot(series_date, y, 'k-', linewidth=0.5, label='ドル円為替レート')
plt.plot(series_date, 2.0 * vol, 'k:', linewidth=0.5, label='2シグマ区間')
plt.plot(series_date, -2.0 * vol, 'k:', linewidth=0.5)
plt.xlim(series_date[0], series_date[-1])
plt.xticks(['2014', '2015', '2016', '2017'])
plt.xlabel('営業日', fontproperties=jpfont)
plt.ylabel('日次変化率 (%)', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pybayes_fig_sv_volatility.png', dpi=300)
plt.show()
