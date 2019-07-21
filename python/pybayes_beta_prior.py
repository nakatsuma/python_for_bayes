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
#%% ベルヌーイ分布の成功確率qの事前分布
fig1 = plt.figure(num=1, facecolor='w')
q = np.linspace(0, 1, 250)
plt.plot(q, st.uniform.pdf(q), 'k-')
plt.plot(q, st.beta.pdf(q, 4, 6), 'k--')
plt.xlim(0, 1)
plt.ylim(0, 2.8)
plt.legend(['(A) 一様分布 ($\\alpha$ = 1, $\\beta$ = 1)',
            '(B) ベータ分布 ($\\alpha$ = 4, $\\beta$ = 6)'],
            loc='best', frameon=False, prop=jpfont)
plt.xlabel('成功確率 q', fontproperties=jpfont)
plt.ylabel('確率密度', fontproperties=jpfont)
plt.savefig('pybayes_fig_beta_prior.png', dpi=300)
plt.show()
