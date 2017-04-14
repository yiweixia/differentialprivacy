# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:02:23 2017

@author: Yiwei Xia
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

df = pd.read_csv("raw.csv")

df = df.rename(columns = {'sales': 'job'})

for column in ['job', 'salary']:
    dummies = pd.get_dummies(df[column])
    df[column + "_" + dummies.columns] = dummies

del df['salary']
del df['job']

df.to_csv("processed.csv")

def split_train_test(df):
    train, test = train_test_split(df, test_size=.2, random_state=42)
    return{"train":train, "test":test}

# categorizes boolean 1/0 values into 1,2,3
# e.g. categorize(df, "salary") will return a new dataframe with one category salary_categorized = {1,2,3}
def categorize(df, starting_string):
    i = 0
    new_name = starting_string + "_categorized"
    df[new_name] = 0;
    for column in list(df):
        if column.startswith(starting_string) and column != new_name:
            df[column] = df[column]*i
            df[new_name] = df[new_name] + df[column]
            del df[column]
            i += 1
    
# splits dataframe columns based on starting string
# e.g. split_xy(df, "salary") will have salary_high/medium/low as Y, and everything else as X
def split_xy(df, starting_string):
    y = df.filter(regex='^' + starting_string, axis=1)
    for column in list(df):
        if column.startswith(starting_string):
            del df[column]
    x = df
    return {"x":x, "y":y}

def normalize(df, col):
    #(xi - min(x))/ (max(x) - min(x))
    df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())

def super_split(df, starting_string, laplace):
    categorize(df, starting_string)
    categorize(df, 'salary')
    
    if not os.path.exists("data/"):
        os.makedirs("data/")
    path = "data/" + starting_string + "_categorical/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if laplace:
        path = path + "laplace/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        for col in list(df):
            normalize(df, col)
            l = [None] * len(df)
            for i, val in df[col].iteritems():
                l[i] = val + np.random.laplace(scale=0.5)
            df[col] = pd.Series(l, index = df.index)
    
    tt_split = split_train_test(df)
    train = tt_split['train']
    test = tt_split['test']
    train_xy = split_xy(train, starting_string)
    test_xy = split_xy(test, starting_string)

    train_xy['x'].to_csv(path + "/train_x.csv")
    train_xy['y'].to_csv(path + "/train_y.csv")
    test_xy['x'].to_csv(path + "/test_x.csv")
    test_xy['y'].to_csv(path + "/test_y.csv")
        
def super_split_salary():
    super_split(df, "salary", False)
    
def super_split_job():
    super_split(df, "job", False)
    
def super_split_job_laplace():
    super_split(df, "job", True)