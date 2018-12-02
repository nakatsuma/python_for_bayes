# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   MatplotlibのPyplotモジュールの読み込み
import matplotlib.pyplot as plt
#%% ベータ分布の確率密度関数
q = np.linspace(0, 1, 250)
value_a = np.array([0.5, 1.0, 2.0, 4.0])
value_b = np.array([0.5, 1.0, 2.0, 4.0])
rows = value_a.shape[0]
cols = value_b.shape[0]
fig, ax = plt.subplots(rows, cols, sharex='all', sharey='all',
                       num=1, facecolor='w')
ax[0, 0].set_xlim(0.0, 1.0)
ax[0, 0].set_ylim(0.0, 4.5)
for row_index in range(rows):
    a = value_a[row_index]
    ax[row_index, 0].set_ylabel('$\\alpha$ = {0:3.1f}'.format(a),
                                fontsize=12)
    for column_index in range(cols):
        b = value_b[column_index]
        ax[row_index, column_index].plot(q, st.beta.pdf(q, a, b), 'k-')
        if row_index == 0:
            ax[0, column_index].set_title('$\\beta$ = {0:3.1f}'.format(b),
                                          fontsize=12)
plt.tight_layout()
plt.savefig('pybayes_fig_beta_distribution.png', dpi=300)
plt.show()
