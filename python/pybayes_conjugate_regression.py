# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   SciPyのoptimizeモジュールの読み込み
import scipy.optimize as opt
#   SciPyのLinalgモジュールの読み込み
import scipy.linalg as la
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
#%% 回帰モデルの係数と誤差項の分散に関するベイズ推論
#   逆ガンマ分布のHPD区間の計算
def invgamma_hpdi(hpdi0, alpha, beta, prob):
    """
        入力
        hpdi0:  HPD区間の初期値
        alpha:  逆ガンマ分布の形状パラメータ
        beta:   逆ガンマ分布の尺度パラメータ
        prob:   HPD区間の確率 (0 < prob < 1)
        出力
        HPD区間
    """
    def hpdi_conditions(v, a, b, p):
        """
            入力
            v:  HPD区間
            a:  逆ガンマ分布の形状パラメータ
            b:  逆ガンマ分布の尺度パラメータ
            p:  HPD区間の確率 (0 < p < 1)
            出力
            HPD区間の条件式の値
        """
        eq1 = st.invgamma.cdf(v[1], a, scale=b) \
              - st.invgamma.cdf(v[0], a, scale=b) - p
        eq2 = st.invgamma.pdf(v[1], a, scale=b) \
              - st.invgamma.pdf(v[0], a, scale=b)
        return np.hstack((eq1, eq2))
    return opt.root(hpdi_conditions, hpdi0, args=(alpha, beta, prob)).x
#   回帰モデルの係数と誤差項の分散の事後統計量の計算
def regression_stats(y, X, b0, A0, nu0, lam0, prob):
    """
        入力
        y:      被説明変数
        X:      説明変数
        b0:     回帰係数の条件付事前分布（多変量正規分布）の平均
        A0:     回帰係数の条件付事前分布（多変量正規分布）の精度行列
        nu0:    誤差項の分散の事前分布（逆ガンマ分布）の形状パラメータ
        lam0:   誤差項の分散の事前分布（逆ガンマ分布）の尺度パラメータ
        prob:   区間確率 (0 < prob < 1)
        出力
        results:    事後統計量のデータフレーム
        b_star:     回帰係数の条件付事後分布（多変量正規分布）の平均
        A_star:     回帰係数の条件付事後分布（多変量正規分布）の精度行列
        nu_star:    誤差項の分散の事後分布（逆ガンマ分布）の形状パラメータ
        lam_star:   誤差項の分散の事後分布（逆ガンマ分布）の尺度パラメータ
    """
    XX = X.T.dot(X)
    Xy = X.T.dot(y)
    b_ols = la.solve(XX, Xy)
    A_star = XX + A0
    b_star = la.solve(A_star, Xy + A0.dot(b0))
    C_star = la.inv(la.inv(XX) + la.inv(A0))
    nu_star = y.size + nu0
    lam_star =  np.square(y - X.dot(b_ols)).sum() \
                + (b0 - b_ols).T.dot(C_star).dot(b0 - b_ols) + lam0
    h_star = np.sqrt(lam_star / nu_star * np.diag(la.inv(A_star)))
    sd_b = st.t.std(nu_star, loc=b_star, scale=h_star)
    ci_b = np.vstack(st.t.interval(prob, nu_star, loc=b_star, scale=h_star))
    hpdi_b = ci_b
    stats_b = np.vstack((b_star, b_star, b_star, sd_b, ci_b, hpdi_b)).T
    mean_sigma2 = st.invgamma.mean(0.5*nu_star, scale=0.5*lam_star)
    median_sigma2 = st.invgamma.median(0.5*nu_star, scale=0.5*lam_star)
    mode_sigma2 = lam_star / (nu_star + 2.0)
    sd_sigma2 = st.invgamma.std(0.5*nu_star, scale=0.5*lam_star)
    ci_sigma2 = st.invgamma.interval(prob, 0.5*nu_star, scale=0.5*lam_star)
    hpdi_sigma2 = invgamma_hpdi(ci_sigma2, 0.5*nu_star, 0.5*lam_star, prob)
    stats_sigma2 = np.hstack((mean_sigma2, median_sigma2, mode_sigma2,
                              sd_sigma2, ci_sigma2, hpdi_sigma2))
    stats = np.vstack((stats_b, stats_sigma2))
    stats_string = ['平均', '中央値', '最頻値', '標準偏差', '信用区間（下限）',
                    '信用区間（上限）', 'HPD区間（下限）', 'HPD区間（上限）']
    param_string = ['切片 $\\alpha$', '傾き $\\beta$', '分散 $\\sigma^2$']
    results = pd.DataFrame(stats, index=param_string, columns=stats_string)
    return results, b_star, h_star, nu_star, lam_star
#%% 回帰モデルからのデータの生成
n = 50
np.random.seed(99)
u = st.norm.rvs(scale=0.7, size=n)
x = st.uniform.rvs(loc=-np.sqrt(3.0), scale=2.0*np.sqrt(3.0), size=n)
y = 1.0 + 2.0 * x + u
X = np.stack((np.ones(n), x), axis=1)
fig1 = plt.figure(num=1, facecolor='w')
plt.scatter(x, y, color='0.5', marker='+', label='観測値')
x_range = (-np.sqrt(3.0), np.sqrt(3.0))
y_range = (1.0 - 2.0*np.sqrt(3.0), 1.0 + 2.0*np.sqrt(3.0))
plt.plot(x_range, y_range, 'k-', label='回帰直線')
plt.xlim(x_range)
plt.ylim(y_range[0] - 2.0, y_range[1] + 2.0)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', frameon=False, prop=jpfont)
plt.savefig('pybayes_fig_regression_scatter.png', dpi=300)
plt.show()
#%% 事後統計量の計算
b0 = np.zeros(2)
A0 = 0.2 * np.eye(2)
nu0 = 5.0
lam0 = 7.0
h0 = np.sqrt(np.diag(lam0 / nu0 * la.inv(A0)))
prob = 0.95
results, b_star, h, nu_star, lam_star = regression_stats(y, X, b0, A0, nu0,
                                                         lam0, prob)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
labels = ['切片 $\\alpha$', '傾き $\\beta$', '分散 $\\sigma^2$']
fig2, ax2 = plt.subplots(1, 3, sharey='all', sharex='all',
                         num=2, figsize=(12, 4), facecolor='w')
x = np.linspace(0, 3.2, 250)
ax2[0].set_xlim(0, 3.2)
ax2[0].set_ylim(0, 4)
ax2[0].set_ylabel('確率密度', fontproperties=jpfont)
for index in range(3):
    if index < 2:
        posterior = st.t.pdf(x, nu_star, loc=b_star[index], scale=h[index])
        prior = st.t.pdf(x, nu0, loc=b0[index], scale=h0[index])
    else:
        posterior = st.invgamma.pdf(x, 0.5*nu_star, scale=0.5*lam_star)
        prior = st.invgamma.pdf(x, 0.5*nu0, scale=0.5*lam0)
    ax2[index].plot(x, posterior, 'k-', label='事後分布')
    ax2[index].plot(x, prior, 'k:', label='事前分布')
    ax2[index].set_xlabel(labels[index], fontproperties=jpfont)
    ax2[index].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_regression_posterior.png', dpi=300)
plt.show()
