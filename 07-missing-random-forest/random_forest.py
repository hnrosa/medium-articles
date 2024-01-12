# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:37:02 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""

# %%
from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from category_encoders import JamesSteinEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as acc_score

#%%

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

y = y['income'].str.contains('>50K')
  
# metadata 
print(X.info())


# %% IT WILL NOT WORK!

random_clf = Pipeline([
    ('encoder', JamesSteinEncoder(handle_unknown='return_nan',
                                  handle_missing='return_nan')),
    ('classifier', RandomForestClassifier())
    ])

random_clf.fit(X, y) 

y_pred = random_clf.predict(X)

print(f'Training Sucessfully Done. Train Acc. Score: {acc_score(y, y_pred)}')

# %%

random_clf = Pipeline([
    ('encoder', JamesSteinEncoder(handle_unknown='return_nan',
                                  handle_missing='return_nan')),
    ('classifier', BaggingClassifier(
        estimator = DecisionTreeClassifier(
        max_features = 'sqrt'
            ),
        n_estimators = 100
        ))
    ])

random_clf.fit(X, y) 

y_pred = random_clf.predict(X)

print(f'Training Sucessfully Done. Train Acc. Score: {acc_score(y, y_pred):.3f}')