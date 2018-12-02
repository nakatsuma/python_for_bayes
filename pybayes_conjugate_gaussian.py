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
#%% 正規分布の平均と分散に関するベイズ推論
#   逆ガンマ分布のHPD区間の計算
def invgamma_hpdi(ci0, alpha, beta, prob):
    """
        入力
        ci0:    HPD区間の初期値
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
    return opt.root(hpdi_conditions, ci0, args=(alpha, beta, prob)).x
#   正規分布の平均と分散の事後統計量の計算
def gaussian_stats(data, mu0, n0, nu0, lam0, prob):
    """
        入力
        data:   データ
        mu0:    平均の条件付事前分布（正規分布）の平均
        n0:     平均の条件付事前分布（正規分布）の精度パラメータ
        nu0:    分散の事前分布（逆ガンマ分布）の形状パラメータ
        lam0:   分散の事前分布（逆ガンマ分布）の尺度パラメータ
        prob:   区間確率 (0 < prob < 1)
        出力
        results:    事後統計量のデータフレーム
        mu_star:    平均の条件付事後分布（正規分布）の平均
        n_star:     平均の条件付事後分布（正規分布）の精度パラメータ
        nu_star:    分散の事後分布（逆ガンマ分布）の形状パラメータ
        lam_star:   分散の事後分布（逆ガンマ分布）の尺度パラメータ
    """
    n = data.size
    mean_data = data.mean()
    ssd_data = n * data.var()
    n_star = n + n0
    mu_star = (n * mean_data + n0 * mu0) / n_star
    nu_star = n + nu0
    lam_star = ssd_data + n * n0 / n_star * (mu0 - mean_data)**2 + lam0
    tau_star = np.sqrt(lam_star / nu_star / n_star)
    sd_mu = st.t.std(nu_star, loc=mu_star, scale=tau_star)
    ci_mu = st.t.interval(prob, nu_star, loc=mu_star, scale=tau_star)
    mean_sigma2 = st.invgamma.mean(0.5*nu_star, scale=0.5*lam_star)
    mode_sigma2 = lam_star / (nu_star + 2.0)
    median_sigma2 = st.invgamma.median(0.5*nu_star, scale=0.5*lam_star)
    sd_sigma2 = st.invgamma.std(0.5*nu_star, scale=0.5*lam_star)
    ci_sigma2 = st.invgamma.interval(prob, 0.5*nu_star, scale=0.5*lam_star)
    hpdi_sigma2 = invgamma_hpdi(ci_sigma2, 0.5*nu_star, 0.5*lam_star, prob)
    stats_mu = np.hstack((mu_star, mu_star, mu_star, sd_mu, ci_mu, ci_mu))
    stats_sigma2 = np.hstack((mean_sigma2, median_sigma2, mode_sigma2,
                              sd_sigma2, ci_sigma2, hpdi_sigma2))
    stats = np.vstack((stats_mu, stats_sigma2))
    stats_string = ['平均', '中央値', '最頻値', '標準偏差', '信用区間（下限）',
                    '信用区間（上限）', 'HPD区間（下限）', 'HPD区間（上限）']
    param_string = ['平均 $\\mu$', '分散 $\\sigma^2$']
    results = pd.DataFrame(stats, index=param_string, columns=stats_string)
    return results, mu_star, tau_star, nu_star, lam_star
#%% 正規分布からのデータの生成
mu = 1.0
sigma = 2.0
n = 50
np.random.seed(99)
data = st.norm.rvs(loc=mu, scale=sigma, size=n)
#%% 事後統計量の計算
mu0 = 0.0
n0 = 0.2
nu0 = 5.0
lam0 = 7.0
tau0 = np.sqrt(lam0 / nu0 / n0)
prob = 0.95
results, mu_star, tau_star, nu_star, lam_star \
    = gaussian_stats(data, mu0, n0, nu0, lam0, prob)
print(results.to_string(float_format='{:,.4f}'.format))
#%% 事後分布のグラフの作成
fig, ax = plt.subplots(1, 2, num=1, figsize=(8, 4), facecolor='w')
#   平均の周辺事後分布のグラフの作成
x1 = np.linspace(-6, 6, 250)
ax[0].plot(x1, st.t.pdf(x1, nu_star, loc=mu_star, scale=tau_star),
           'k-', label='事後分布')
ax[0].plot(x1, st.t.pdf(x1, nu0, loc=mu0, scale=tau0),
           'k:', label='事前分布')
ax[0].set_xlim(-6, 6)
ax[0].set_ylim(0, 1.55)
ax[0].set_xlabel('平均 $\\mu$', fontproperties=jpfont)
ax[0].set_ylabel('確率密度', fontproperties=jpfont)
ax[0].legend(loc='best', frameon=False, prop=jpfont)
#   分散の周辺事後分布のグラフの作成
x2 = np.linspace(0, 10, 250)
ax[1].plot(x2, st.invgamma.pdf(x2, 0.5*nu_star, scale=0.5*lam_star),
           'k-', label='事後分布')
ax[1].plot(x2, st.invgamma.pdf(x2, 0.5*nu0, scale=0.5*lam0),
           'k:', label='事前分布')
ax[1].set_xlim(0, 10)
ax[1].set_ylim(0, 0.65)
ax[1].set_xlabel('分散 $\\sigma^2$', fontproperties=jpfont)
ax[1].set_ylabel('確率密度', fontproperties=jpfont)
ax[1].legend(loc='best', frameon=False, prop=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_gaussian_posterior.png', dpi=300)
plt.show()
