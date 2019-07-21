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
#%%  逆ガンマ分布とt分布のグラフの作成
value_a = np.array([1.0, 3.0, 5.0, 5.0])
value_b = np.array([2.0, 2.0, 2.0, 1.0])
value_n = np.array([1.0, 2.0, 5.0])
styles = ['-', '--', '-.', ':']
fig, ax = plt.subplots(1, 2, num=1, figsize=(8, 4), facecolor='w')
#   逆ガンマ分布のグラフの作成
x1 = np.linspace(0, 2.3, 250)
for index in range(value_a.size):
    a_i = value_a[index]
    b_i = value_b[index]
    plot_label = '$\\alpha$ = {0:3.1f}, $\\beta$ = {1:3.1f}' \
                 .format(a_i, b_i)
    ax[0].plot(x1, st.invgamma.pdf(x1, a_i, scale=b_i), color='k',
               linestyle=styles[index], label=plot_label)
ax[0].set_xlim(0, 2.3)
ax[0].set_ylim(0, 5)
ax[0].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[0].set_ylabel('確率密度', fontproperties=jpfont)
ax[0].legend(loc='upper right', frameon=False, prop=jpfont)
#   t分布のグラフの作成
x2 = np.linspace(-6.5, 6.5, 250)
for index in range(value_n.size):
    n_i = value_n[index]
    plot_label = '$\\nu$ = {0:3.1f}'.format(n_i)
    ax[1].plot(x2, st.t.pdf(x2, n_i), color='k',
               linestyle=styles[index], label=plot_label)
ax[1].plot(x2, st.norm.pdf(x2), color='k',
           linestyle=styles[-1], label='$\\nu = \\infty$')
ax[1].set_xlim(-6.5, 6.5)
ax[1].set_ylim(0, 0.42)
ax[1].set_xlabel('確率変数の値', fontproperties=jpfont)
ax[1].set_ylabel('確率密度', fontproperties=jpfont)
ax[1].legend(loc='upper right', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_invgamma_t.png', dpi=300)
plt.show()
