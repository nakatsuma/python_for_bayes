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
#%% ベルヌーイ分布の成功確率に関するベイズ推論
#   ベータ分布のHPD区間の計算
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
#   ベルヌーイ分布の成功確率の事後統計量の計算
def bernoulli_stats(data, a0, b0, prob):
    """
        入力
        data:   データ（取りうる値は0か1）
        a0:     事前分布のパラメータ1
        b0:     事前分布のパラメータ2
        prob:   区間確率 (0 < prob < 1)
        出力
        results:事後統計量のデータフレーム
        a:      事後分布のパラメータ1
        b:      事後分布のパラメータ2
    """
    n = data.size
    sum_data = data.sum()
    a = sum_data + a0
    b = n - sum_data + b0
    mean_pi = st.beta.mean(a, b)
    median_pi = st.beta.median(a, b)
    mode_pi = (a - 1.0) / (a + b - 2.0)
    sd_pi = st.beta.std(a, b)
    ci_pi = st.beta.interval(prob, a, b)
    hpdi_pi = beta_hpdi(ci_pi, a, b, prob)
    stats = np.hstack((mean_pi, median_pi, mode_pi, sd_pi, ci_pi, hpdi_pi))
    stats = stats.reshape((1, 8))
    stats_string = ['平均', '中央値', '最頻値', '標準偏差', '信用区間（下限）',
                    '信用区間（上限）', 'HPD区間（下限）', 'HPD区間（上限）']
    param_string = ['成功確率 q']
    results = pd.DataFrame(stats, index=param_string, columns=stats_string)
    return results, a, b
#%% ベルヌーイ分布からのデータの生成
p = 0.25
n = 50
np.random.seed(99)
data = st.bernoulli.rvs(p, size=n)
#%% 事後統計量の計算
a0 = 1.0
b0 = 1.0
prob = 0.95
results, a, b = bernoulli_stats(data, a0, b0, prob)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
fig1 = plt.figure(num=1, facecolor='w')
q = np.linspace(0, 1, 250)
plt.plot(q, st.beta.pdf(q, a, b), 'k-', label='事後分布')
plt.plot(q, st.beta.pdf(q, a0, b0), 'k:', label='事前分布')
plt.xlim(0, 1)
plt.ylim(0, 7)
plt.xlabel('成功確率 q', fontproperties=jpfont)
plt.ylabel('確率密度', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pybayes_fig_bernoulli_posterior.png', dpi=300)
plt.show()
#%% 事前分布とデータの累積が事後分布の形状に与える影響の可視化
np.random.seed(99)
data = st.bernoulli.rvs(p, size=250)
value_size = np.array([10, 50, 250])
value_a0 = np.array([1.0, 6.0])
value_b0 = np.array([1.0, 4.0])
styles = [':', '-.', '--', '-']
fig2, ax2 = plt.subplots(1, 2, sharey='all', sharex='all',
                         num=2, figsize=(8, 4), facecolor='w')
ax2[0].set_xlim(0, 1)
ax2[0].set_ylim(0, 15.5)
ax2[0].set_ylabel('確率密度', fontproperties=jpfont)
for index in range(2):
    style_index = 0
    a0_i = value_a0[index]
    b0_i = value_b0[index]
    ax2[index].plot(q, st.beta.pdf(q, a0_i, b0_i), color='k',
                    linestyle=styles[style_index],
                    label='事前分布 Beta({0:<3.1f}, {1:<3.1f})' \
                    .format(a0_i, b0_i))
    for n_j in value_size:
        style_index += 1
        sum_data = np.sum(data[:n_j])
        a_j = sum_data + a0_i
        b_j = n_j - sum_data + b0_i
        ax2[index].plot(q, st.beta.pdf(q, a_j, b_j), color='k',
                        linestyle=styles[style_index],
                        label='事後分布 ( n = {0:<3d} )'.format(n_j))
    ax2[index].set_xlabel('成功確率 q', fontproperties=jpfont)
    ax2[index].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_bernoulli_posterior_convergence.png', dpi=300)
plt.show()
