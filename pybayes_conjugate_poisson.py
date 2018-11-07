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
#%% ポアソン分布に関するベイズ推論
#   ガンマ分布のHPD区間の計算
def gamma_hpdi(ci0, alpha, theta, prob):
    """
        入力
        ci0:    HPD区間の初期値
        alpha:  ガンマ分布の形状パラメータ
        theta:  ガンマ分布の尺度パラメータ
        prob:   HPD区間の確率 (0 < prob < 1)
        出力
        HPD区間
    """
    def hpdi_conditions(v, a, t, p):
        """
            入力
            v:  HPD区間
            a:  ガンマ分布の形状パラメータ
            t:  ガンマ分布の尺度パラメータ
            p:  HPD区間の確率 (0 < p < 1)
            出力
            HPD区間の条件式の値
        """
        eq1 = st.gamma.cdf(v[1], a, scale=t) \
              - st.gamma.cdf(v[0], a, scale=t) - p
        eq2 = st.gamma.pdf(v[1], a, scale=t) \
              - st.gamma.pdf(v[0], a, scale=t)
        return np.hstack((eq1, eq2))
    return opt.root(hpdi_conditions, ci0, args=(alpha, theta, prob)).x
#   ポアソン分布のパラメータの事後統計量の計算
def poisson_stats(data, a0, b0, prob):
    """
        入力
        data:   データ
        a0:     事前分布の形状パラメータ
        b0:     事前分布の尺度パラメータの逆数
        prob:   区間確率 (0 < prob < 1)
        出力
        results:    事後統計量のデータフレーム
        a_star:     事後分布の形状パラメータ
        b_star:     事後分布の尺度パラメータの逆数
    """
    n = data.size
    a_star = data.sum() + a0
    b_star = n + b0
    theta_star = 1.0 / b_star
    mean_lam = st.gamma.mean(a_star, scale=theta_star)
    median_lam = st.gamma.median(a_star, scale=theta_star)
    mode_lam = (a_star - 1.0) * theta_star
    sd_lam = st.gamma.std(a_star, scale=theta_star)
    ci_lam = st.gamma.interval(prob, a_star, scale=theta_star)
    hpdi_lam = gamma_hpdi(ci_lam, a_star, theta_star, prob)
    stats = np.hstack((mean_lam, median_lam, mode_lam,
                       sd_lam, ci_lam, hpdi_lam)).reshape((1, 8))
    stats_string = ['平均', '中央値', '最頻値', '標準偏差', '信用区間（下限）',
                    '信用区間（上限）', 'HPD区間（下限）', 'HPD区間（上限）']
    param_string = ['$\\lambda$']
    results = pd.DataFrame(stats, index=param_string, columns=stats_string)
    return results, a_star, b_star
#%% ポアソン分布からのデータの生成
lam = 3.0
n = 50
np.random.seed(99)
data = st.poisson.rvs(lam, size=n)
#%% 事後統計量の計算
a0 = 1.0
b0 = 1.0
prob = 0.95
results, a_star, b_star = poisson_stats(data, a0, b0, prob)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
fig = plt.figure(num=1, facecolor='w')
x = np.linspace(0, 6, 250)
plt.plot(x, st.gamma.pdf(x, a_star, scale=1.0/b_star), 'k-',
         label='事後分布')
plt.plot(x, st.gamma.pdf(x, a0, scale=1.0/b0), 'k:',
         label='事前分布')
plt.xlim(0, 6)
plt.ylim(0, 1.75)
plt.xlabel('$\\lambda$', fontproperties=jpfont)
plt.ylabel('確率密度', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pybayes_fig_poisson_posterior.png', dpi=300)
plt.show()
