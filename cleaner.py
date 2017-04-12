# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:02:23 2017

@author: Yiwei Xia
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("raw.csv")

for column in ['sales', 'salary']:
    dummies = pd.get_dummies(df[column])
    df[dummies.columns] = dummies

del df['salary']

df.to_csv("processed.csv")

def split(df):
    train, test = train_test_split(df, test_size=.2, random_state=42)
    train.to_csv("train_80.csv")
    test.to_csv("test_20.csv")
