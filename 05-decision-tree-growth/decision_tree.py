# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:43:43 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
from mlxtend.plotting import plot_decision_regions
#

# %%

df = load_penguins().dropna()

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi = 300)
sns.scatterplot(data = df, x = 'bill_length_mm', y = 'body_mass_g', 
                palette = ['b', 'r', 'g'], hue = 'species',
                style = 'species',  markers = ['s', '^', 'o'])

X = df.loc[:, ['bill_length_mm', 'body_mass_g']]

y = df['species']


# %%


clf1 = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 121, stratify = y)

le = LabelEncoder().fit(y_train)

clf1.fit(X_train, y_train)

y_pred = clf1.predict(X_test)

print(classification_report(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
dot_data = tree.export_graphviz(clf1, 
                   feature_names= list(X.columns),  
                   class_names=  list(le.classes_),
                   filled=True)

graphviz.Source(dot_data, format="png")

# %%

kf = StratifiedKFold(4, shuffle = True, random_state = 121)

param_dist = {
    'min_impurity_decrease': np.linspace(0.01, 0.1, 20)
    }

grid_search = GridSearchCV(clf1, param_dist, cv = kf, verbose = 1)

grid_search.fit(X_train, y_train)

# %% 

clf2 = DecisionTreeClassifier(**grid_search.best_params_)

clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)


print(f'Best min_impurity_decrease: {grid_search.best_params_["min_impurity_decrease"]:.3f}')
print(classification_report(y_test, y_pred))


fig = plt.figure(figsize=(25,20))
dot_data = tree.export_graphviz(clf2, 
                   feature_names= list(X.columns),  
                   class_names=  list(le.classes_),
                   filled=True)

graphviz.Source(dot_data, format="png")

# %%

y_train_ = le.transform(y_train)
X_train_ = X_train.to_numpy()

clf1.fit(X_train_, y_train_)
clf2.fit(X_train_, y_train_)

graphviz.Source(dot_data, format="png")

fig, ax = plt.subplots(1, 2, figsize = (15, 6), dpi = 300)

plot_decision_regions(X_train_, y_train_, clf = clf1, ax = ax[0], colors = 'b,r,g')
plot_decision_regions(X_train_, y_train_, clf = clf2, ax = ax[1], colors = 'b,r,g')

handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, list(le.classes_))

handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles, list(le.classes_))

ax[0].set_xlabel('bill_depth_mm')
ax[1].set_xlabel('bill_depth_mm')

ax[0].set_ylabel('body_mass_g')
ax[1].set_ylabel('body_mass_g')

ax[0].set_title('Decision Tree Classification Regions w/o Tuning')
ax[1].set_title('Decision Tree Classification Regions with Tuning')


