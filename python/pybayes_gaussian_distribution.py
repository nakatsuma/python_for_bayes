# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
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
#%% 正規分布のグラフ
value_mu = np.array([0.0, 2.0, -2.0])
value_sigma = np.array([1.0, 0.5, 2.0])
styles = ['-', '--', '-.']
x = np.linspace(-6, 6, 250)
fig, ax = plt.subplots(1, 2, sharex='row',
                       num=1,  figsize=(8, 4), facecolor='w')
ax[0].set_xlim(-6, 6)
ax[0].set_ylabel('確率密度', fontproperties=jpfont)
#   平均による分布の形状の変化
for index in range(value_mu.size):
    mu_i = value_mu[index]
    plot_label = '$\\mu$ = {0:< 3.1f}'.format(mu_i)
    ax[0].plot(x, st.norm.pdf(x, loc=mu_i), color='k',
               linestyle=styles[index], label=plot_label)
ax[0].set_ylim(0, 0.55)
ax[0].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[0].legend(loc='upper right', frameon=False, prop=jpfont)
#   分散による分布の形状の変化
for index in range(value_mu.size):
    sigma_i = value_sigma[index]
    plot_label = '$\\sigma$ = {0:<3.1f}'.format(sigma_i)
    ax[1].plot(x, st.norm.pdf(x, scale=sigma_i), color='k',
               linestyle=styles[index], label=plot_label)
ax[1].set_ylim(0, 0.9)
ax[1].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[1].legend(loc='upper right', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_gaussian_distribution.png', dpi=300)
plt.show()
