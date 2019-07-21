# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   Pandasの読み込み
import pandas as pd
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
#%% 使用電力量データの読み込み
"""
    電灯電力需要実績月報・用途別使用電力量・販売電力合計・10社計
    電気事業連合会ウェブサイト・電力統計情報より入手
    http://www.fepc.or.jp/library/data/tokei/index.html
"""
data1 = pd.read_csv('electricity.csv', index_col=0)
y1 = np.log(data1.values.reshape((data1.shape[0]//3, 3)).sum(axis=1))
y1 = 100 * (y1 - y1[0])
series_date1 = pd.date_range(start='1/1/1989', periods=y1.size, freq='Q')
#%% ドル円為替レート日次データの読み込み
"""
    The Pacific Exchange Rate Serviceより入手
    http://fx.sauder.ubc.ca/data.html
"""
data2 = pd.read_csv('dollaryen.csv', index_col=0)
y2 = 100 * np.diff(np.log(data2.values.ravel()))
series_date2 = pd.to_datetime(data2.index[1:])
#%% 時系列プロット
fig, ax = plt.subplots(3, 1, num=1, facecolor='w')
ax[0].plot(series_date1, y1, 'k-')
ax[0].set_xlim(series_date1[0], series_date1[-1])
ax[0].set_title('使用電力量（1989年第1四半期=0）', fontproperties=jpfont)
ax[0].set_ylabel('変化率', fontproperties=jpfont)
ax[1].plot(series_date2, y2, 'k-', linewidth=0.8)
ax[1].set_xlim(series_date2[0], series_date2[-1])
ax[1].set_xticks(['2014', '2015', '2016', '2017'])
ax[1].set_title('ドル円為替レート日次変化率', fontproperties=jpfont)
ax[2].plot(series_date2, np.abs(y2), 'k-', linewidth=0.8)
ax[2].set_xlim(series_date2[0], series_date2[-1])
ax[2].set_xticks(['2014', '2015', '2016', '2017'])
ax[2].set_title('ドル円為替レート日次変化率（絶対値）', fontproperties=jpfont)
plt.tight_layout()
plt.savefig('pybayes_fig_timeseries_data.png', dpi=300)
plt.show()
