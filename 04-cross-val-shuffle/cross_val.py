# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:00:19 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""
# %%

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# %%

rng = np.random.default_rng(seed = 101)
res  = rng.normal(loc = 0., scale = 25., size = 300 * 5)

x = np.linspace(0, 10, 300)

X = np.hstack([x, x, x, x, x])
X = np.sort(X)


y = -0.3*X**4 + 3 * X**3 - 2*X**2 + 10 * X
y = y - res

fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi = 300)

ax.scatter(X, y, alpha = 0.3, color = 'b')
ax.set_xlabel('Dependent Variable')
ax.set_ylabel('Target')
ax.set_title('Complete Dataset')

# %%

kf_no = KFold(4)
kf = KFold(4, shuffle = True, random_state = 101)

fig, ax = plt.subplots(2, 4, figsize = (20, 9), dpi = 300)

for i, (train_index, test_index) in enumerate(kf_no.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    ax[0, i].scatter(X_train, y_train, color = 'b', alpha = 0.3, label = 'Training Data')
    ax[0, i].scatter(X_test, y_test, color = 'y', alpha = 0.3, label = 'Test Data')
    ax[0, i].set_xlabel('Dependent Variable', fontsize = 10)
    ax[0, i].set_ylabel('Target', fontsize = 10)
    ax[0, i].set_title(f'Fold {i+1} w/o Shuffle', fontsize = 15)
    ax[0, i].legend()
    
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    ax[1, i].scatter(X_train, y_train, color = 'b', alpha = 0.3, label = 'Training Data')
    ax[1, i].scatter(X_test, y_test, color = 'y', alpha = 0.3, label = 'Test Data')
    ax[1, i].set_xlabel('Dependent Variable', fontsize = 10)
    ax[1, i].set_ylabel('Target', fontsize = 10)
    ax[1, i].set_title(f'Fold {i+1} with Shuffle', fontsize = 15)
    ax[1, i].legend()
    
fig.tight_layout()

# %%

model = RandomForestRegressor()

fig, ax = plt.subplots(2, 4, figsize = (20, 9), dpi = 300)

for i, (train_index, test_index) in enumerate(kf_no.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    model.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model.predict(X.reshape(-1, 1))
    
    y_test_pred = model.predict(X_test.reshape(-1, 1))
    rmse = mean_squared_error(y_test, y_test_pred, squared = True)
    
    ax[0, i].scatter(X_train, y_train, color = 'b', alpha = 0.3, label = 'Training Data')
    ax[0, i].scatter(X_test, y_test, color = 'y', alpha = 0.3, label = 'Test Data')
    ax[0, i].plot(X, y_pred, color = 'r', label = 'Model Pr ediction')
    ax[0, i].set_xlabel('Dependent Variable', fontsize = 10)
    ax[0, i].set_ylabel('Target', fontsize = 10)
    ax[0, i].set_title(f'Fold {i+1} w/o Shuffle, Test RMSE: {int(rmse):4d}', fontsize = 15)
    ax[0, i].legend(loc = 2)
    
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    model.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model.predict(X.reshape(-1, 1))
    
    y_test_pred = model.predict(X_test.reshape(-1, 1))
    rmse = mean_squared_error(y_test, y_test_pred, squared = True)
    
    ax[1, i].scatter(X_train, y_train, color = 'b', alpha = 0.3, label = 'Training Data')
    ax[1, i].scatter(X_test, y_test, color = 'y', alpha = 0.3, label = 'Test Data')
    ax[1, i].plot(X, y_pred, color = 'r', label = 'Model Prediction')
    ax[1, i].set_xlabel('Dependent Variable', fontsize = 10)
    ax[1, i].set_ylabel('Target', fontsize = 10)
    ax[1, i].set_title(f'Fold {i+1} with Shuffle, Test RMSE: {int(rmse):4d}', fontsize = 15)
    ax[1, i].legend(loc = 2)
    
fig.tight_layout()






