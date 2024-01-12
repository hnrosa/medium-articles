# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:34:22 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""

from sklearn.preprocessing import TargetEncoder as skTargetEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder as ceTargetEncoder
from category_encoders import JamesSteinEncoder
from feature_engine.encoding import MeanEncoder 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np

df_orig = pd.read_csv('mushroom-dataset.csv')

df = df_orig.copy()

sample_ind = df.sample(15, random_state = 99).index

feat = 'habitat'

print(df[feat].value_counts())

print(df.loc[sample_ind, [feat, 'target']])

# %%

encode = df.groupby(feat).apply(lambda df: df['target'].mean())

df['encode_naive'] = df[feat].replace(encode.to_dict())

print(df.loc[sample_ind, [feat, 'encode_naive', 'target']])

# %%

def smooth_encoding(df_, k = 20, f = 100):
    
    ni = df_.shape[0]
    
    alpha = 1 / (1 + np.exp(-(ni - k)/f))
    
    encode = alpha * df_['target'].mean() + (1-alpha) * df['target'].mean()
    
    return encode

encode = df.groupby(feat).apply(smooth_encoding)

df['encode_smooth'] = df[feat].replace(encode.to_dict())

print(df.loc[sample_ind, [feat, 'encode_naive', 'encode_smooth', 'target']])

# %%

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 99)

pipe_naive = Pipeline([
    ('encoding', MeanEncoder()),
    ('predictor', LogisticRegression(max_iter = 10_000))
    ])

pipe_feat_engine = Pipeline([
    ('encoding', MeanEncoder(smoothing = 'auto')),
    ('predictor', LogisticRegression(max_iter = 10_000))
    ])

pipe_ce = Pipeline([
    ('encoding', ceTargetEncoder()),
    ('predictor', LogisticRegression(max_iter = 10_000))
    ])

pipe_js = Pipeline([
    ('encoding', JamesSteinEncoder()),
    ('predictor', LogisticRegression(max_iter = 10_000))
    ])

pipe_sk = Pipeline([
    ('encoding', skTargetEncoder(target_type = 'binary', cv = 10)),
    ('predictor', LogisticRegression(max_iter = 10_000))
    ])

pipes = [pipe_naive, pipe_feat_engine, pipe_ce, pipe_js, pipe_sk]
names = ['NaiveEncoder', 'FeatEngineEncoder', 'CatEncoder', 'JamesSteinEncoder', 'SkLearnEncoder']

df = df_orig.drop('stalk-root', axis = 1)

X = df.iloc[:, 1:5]
y = df.iloc[:, 0]

cross_val_results = []

print('\n--------Accuracies-------')
for name, pipe in zip(names, pipes):
    cv_scores = cross_val_score(pipe, X, y, cv = kf, scoring = 'accuracy', verbose = True)
    print(f'{name:18s}: {cv_scores.mean():.3f}')





