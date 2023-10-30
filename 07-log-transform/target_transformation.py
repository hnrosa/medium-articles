# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:45:33 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, kurtosis

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

from yellowbrick.regressor import ResidualsPlot

# %%

df = pd.read_csv('yacht_hydro.csv')

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi = 300)
sns.histplot(df, x = 'Rr', ax = ax)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('k')
ax.tick_params(labelsize = 15)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.grid(False)

# %%

total = np.sum(df['Rr'] <= 0)

print(f'Número de Valores Negativos e Zeros: {total}')

# %%

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi = 300)
sns.histplot(df, x = 'Rr', ax = ax, log_scale=True)
ax.set_xlabel('log(Rr)')
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('k')
ax.tick_params(labelsize = 15)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.grid(False)

# %%


print('Rr Skewness: {:.2f}'.format(skew(df['Rr'])))
print('Rr Kurtosis: {:.2f}'.format(kurtosis(df['Rr'])))
print('log(Rr) Skewness: {:.2f}'.format(skew(np.log(df['Rr']))))
print('log(Rr) Kurtosis: {:.2f}'.format(kurtosis(np.log(df['Rr']))))


# %% 

X = df.drop(['Rr'], axis=1)

y = df['Rr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

linear_model = LinearRegression()

transf_model = TransformedTargetRegressor(
    regressor = LinearRegression(),
    func = np.log,
    inverse_func= np.exp
    )

# %% 

fig, ax = plt.subplots(1, 2, figsize = (18, 6), dpi = 300);
    
viz = ResidualsPlot(linear_model, ax = ax[0], train_color = 'b', test_color = 'r')
viz.fit(X_train, y_train)
viz.score(X_test, y_test)

viz = ResidualsPlot(transf_model, ax = ax[1], train_color = 'b', test_color = 'r')
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();




















