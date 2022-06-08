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
#%% 使用電力量データの読み込み
"""
    電灯電力需要実績月報・用途別使用電力量・販売電力合計・10社計
    電気事業連合会ウェブサイト・電力統計情報より入手
    http://www.fepc.or.jp/library/data/tokei/index.html
"""
data = pd.read_csv('electricity.csv', index_col=0)
y0 = np.log(data.values.reshape((data.shape[0]//3, 3)).sum(axis=1))
y = 100 * (y0 - y0[0])
n = y.size
series_date = pd.date_range(start='1/1/1989', periods=n, freq='Q')
#%% 確率的トレンド+季節変動
trend_coef = np.array([2.0, -1.0])
seasonal_coef = np.array([-1.0, -1.0, -1.0])
timeseries_decomp = pm.Model()
with timeseries_decomp:
    sigma = pm.HalfCauchy('sigma', beta=1.0)
    tau = pm.HalfCauchy('tau', beta=1.0)
    omega = pm.HalfCauchy('omega', beta=1.0)
    trend = pm.AR('trend', trend_coef, sigma=tau, shape=n)
    seasonal = pm.AR('seasonal', seasonal_coef, sigma=omega, shape=n)
    observation = pm.Normal('y', mu=trend+seasonal, sigma=sigma, observed=y)
#%% 事後分布からのサンプリング
n_draws = 5000
n_chains = 4
n_tune = 2000
with timeseries_decomp:
    trace = pm.sample(draws=n_draws, chains=n_chains, tune=n_tune,
                      target_accept=0.95, random_seed=123)
    param_names = ['sigma', 'tau', 'omega']
    print(az.summary(trace, var_names=param_names))
#%% 事後分布のグラフの作成
series_name = ['原系列', '平滑値', 'トレンド', '季節変動', 'ノイズ']
labels = ['$\\sigma$', '$\\tau$', '$\\omega$']
k = len(labels)
fig1, ax1 = plt.subplots(k, 2, num=1, figsize=(8, 1.5*k), facecolor='w')
for index in range(k):
    mc_trace = trace[param_names[index]]
    x_min = mc_trace.min() - 0.2 * np.abs(mc_trace.min())
    x_max =  mc_trace.max() + 0.2 * np.abs(mc_trace.max())
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
plt.savefig('pybayes_fig_decomp_posterior.png', dpi=300)
plt.show()
#%% 時系列の分解
trend = trace['trend'].mean(axis=0)
seasonal = trace['seasonal'].mean(axis=0)
noise = y - trend - seasonal
series = np.vstack((y, trend + seasonal, trend, seasonal, noise)).T
results = pd.DataFrame(series, index=series_date, columns=series_name)
fig2, ax2 = plt.subplots(4, 1, sharex='col',
                         num=2, figsize=(8, 6), facecolor='w')
for index in range(4):
    ts_name = series_name[index+1]
    ax2[index].plot(results[ts_name], 'k-', label=ts_name)
    ax2[index].set_ylabel(ts_name, fontproperties=jpfont)
ax2[0].plot(results[series_name[0]], 'k:', label=series_name[0])
ax2[0].set_xlim(series_date[0], series_date[-1])
ax2[0].legend(loc='lower right', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_decomp_timeseries.png', dpi=300)
plt.show()
