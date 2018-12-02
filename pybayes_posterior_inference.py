# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   SciPyのoptimizeモジュールの読み込み
import scipy.optimize as opt
#   Pandasの読み込み
import pandas as pd
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
#%%   ベータ分布のHPD区間の計算
def beta_hpdi(ci0, alpha, beta, prob):
    """
        入力
        ci0:    HPD区間の初期値
        alpha:  ベータ分布のパラメータ1
        beta:   ベータ分布のパラメータ2
        prob:   HPD区間の確率 (0 < prob < 1)
        出力
        HPD区間
    """
    def hpdi_conditions(v, a, b, p):
        """
            入力
            v:  HPD区間
            a:  ベータ分布のパラメータ1
            b:  ベータ分布のパラメータ2
            p:  HPD区間の確率 (0 < p < 1)
            出力
            HPD区間の条件式の値
        """
        eq1 = st.beta.cdf(v[1], a, b) - st.beta.cdf(v[0], a, b) - p
        eq2 = st.beta.pdf(v[1], a, b) - st.beta.pdf(v[0], a, b)
        return np.hstack((eq1, eq2))
    return opt.root(hpdi_conditions, ci0, args=(alpha, beta, prob)).x
#%% 損失関数のグラフ
q = np.linspace(0, 1, 250)
fig1 = plt.figure(num=1, facecolor='w')
plt.plot(q, (q - 0.5)**2, 'k-', label='2乗損失 $(q-\\delta)^2$')
plt.plot(q, np.abs(q - 0.5), 'k--', label='絶対損失 $|q-\\delta|$')
plt.axhline(y=1, color='k', linestyle='-.',
            label='0-1損失 $1_{q}(\\delta)$')
plt.plot([0.5, 0.5], [0, 1], 'k:', linewidth=0.5)
plt.plot(0.5, 0, marker='o', mec='k', mfc='k')
plt.plot(0.5, 1, marker='o', mec='k', mfc='w')
plt.xlim(0, 1)
plt.ylim(-0.05, 1.1)
plt.xlabel('点推定 $\\delta$', fontproperties=jpfont)
plt.ylabel('損失', fontproperties=jpfont)
plt.legend(loc=(0.65, 0.55), frameon=False, prop=jpfont)
plt.savefig('pybayes_fig_loss_function.png', dpi=300)
plt.show()
#%% 信用区間とHPD区間の比較
a = 2.0
b = 5.0
prob = 0.9
ci = st.beta.interval(prob, a, b)
hpdi = beta_hpdi(ci, a, b, prob)
q = np.linspace(0, 1, 250)
qq = [np.linspace(ci[0], ci[1], 250), np.linspace(hpdi[0], hpdi[1], 250)]
label1 = 'ベータ分布 ($\\alpha$ = {0:<3.1f}, $\\beta$ = {1:<3.1f})' \
         .format(a, b)
label2 = ['信用区間', 'HPD区間']
fig2, ax2 = plt.subplots(2, 1, sharex='all', sharey='all',
                         num=2, facecolor='w')
ax2[1].set_xlim(0, 1)
ax2[1].set_ylim(0, 2.8)
ax2[1].set_xlabel('成功確率 q', fontproperties=jpfont)
for index in range(2):
    plot_label = '{0:2.0f}%{1:s}'.format(100*prob, label2[index])
    ax2[index].plot(q, st.beta.pdf(q, a, b), 'k-', label=label1)
    ax2[index].fill_between(qq[index], st.beta.pdf(qq[index], a, b),
                            color='0.5', label=plot_label)
    ax2[index].axhline(y=st.beta.pdf(hpdi[0], a, b),
                       color='k', linestyle='-', linewidth=0.5)
    ax2[index].set_ylabel('確率密度', fontproperties=jpfont)
    ax2[index].legend(loc='upper right', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_ci_hpdi.png', dpi=300)
plt.show()
