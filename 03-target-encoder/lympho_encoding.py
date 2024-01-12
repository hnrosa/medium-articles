# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:09:18 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa
"""

import pandas as pd
import numpy as np


df = pd.read_csv('lymphography-dataset.csv', header = None)

y = df.iloc[:, 0]
X = df.iloc[:, 1]

X = X.replace({1: 'normal', 2: 'arched', 3: 'deformed', 4: 'displaced'})
y = y.replace({1: 'normal', 2: 'metastases', 3: 'malign', 4: 'fibrosis'})

df = pd.DataFrame(data = np.array([X, y]).T, columns = ['lymphatics', 'target'])

sample_ind = df.sample(15, random_state = 102).index

print(df.loc[sample_ind, :], end = '\n\n')

# %%

def smooth_encoding(df_, target, k = 1, f = 5):
    
    ni = df_.shape[0]
    
    alpha = 1 / (1 + np.exp(-(ni - k)/f))
    
    encode = alpha * df_[target].mean() + (1-alpha) * df[target].mean()
    
    return encode

y_lst = y.unique()

for y_ in y_lst:
    df[f'encode_{y_}'] = df['target'] == y_
    encode = df.groupby('lymphatics').apply(lambda df: smooth_encoding(df, target = f'encode_{y_}'))
    df[f'encode_{y_}'] = df['lymphatics'].replace(encode.to_dict())

print(df.loc[sample_ind, :].to_string())

