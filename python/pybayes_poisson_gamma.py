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
#%%  ポアソン分布とガンマ分布のグラフの作成
value_l = np.array([1.0, 3.0, 6.0])
value_a = np.array([1.0, 2.0, 2.0, 6.0])
value_t = np.array([2.0, 1.0, 2.0, 1.0])
styles = ['-', '--', '-.', ':']
markers = ['o', '^', 's']
fig, ax = plt.subplots(1, 2, num=1, figsize=(8, 4), facecolor='w')
#   ポアソン分布のグラフの作成
x1 = np.linspace(0, 12, 13)
for index in range(value_l.size):
    l_i = value_l[index]
    plot_label = '$\\lambda$ = {0:3.1f}'.format(l_i)
    ax[0].plot(x1, st.poisson.pmf(x1, l_i),
               color='k', marker=markers[index],
               linestyle='-', linewidth=0.5, label=plot_label)
ax[0].set_xlim(-0.2, 12.2)
ax[0].set_xticks((0, 5, 10))
ax[0].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[0].set_ylabel('確率', fontproperties=jpfont)
ax[0].legend(loc='upper right', frameon=False, prop=jpfont)
#   ガンマ分布のグラフの作成
x2 = np.linspace(0, 13, 250)
for index in range(value_a.size):
    a_i = value_a[index]
    t_i = value_t[index]
    plot_label = '$\\alpha$ = {0:3.1f}, $\\theta$ = {1:3.1f}' \
                 .format(a_i, t_i)
    ax[1].plot(x2, st.gamma.pdf(x2, a_i, scale=t_i), color='k',
               linestyle=styles[index], label=plot_label)
ax[1].set_xlim(0, 13)
ax[1].set_ylim(0, 0.55)
ax[1].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[1].set_ylabel('確率密度', fontproperties=jpfont)
ax[1].legend(loc='upper right', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_poisson_gamma.png', dpi=300)
plt.show()
