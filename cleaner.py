# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:15:48 2017

@author: dkim63
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:02:23 2017

@author: Yiwei Xia
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import math

df = pd.read_csv("raw.csv")

df = df.rename(columns = {'sales': 'job'})

for column in ['job', 'salary']:
    dummies = pd.get_dummies(df[column])
    df[column + "_" + dummies.columns] = dummies

del df['salary']
del df['job']

df.to_csv("processed.csv")

# splits data frame into training and test sets
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
        if not isinstance(column, str):
            continue
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

# checks if vals is a boolean variable
def is_boolean_variable(vals):
    if vals.size == 2:
        if vals[0] == 0 and vals[1] == 1:
            return True
            
    return False

# samples from the original distribution
def add_noise_categorical(x, original, cat_chance):
    if random.random() < cat_chance:
        return random.choice(original)
    else:
        return x
    
# takes processed data, applies noise according to laplace if continuous variable, cat_chance if categorical or boolean
# cat_chance is some value between 0 and 1
def apply_noise(df, cat_chance):
    
    categorize(df, 'salary')
    categorize(df, 'job')
    
    for column in list(df):       

        vals = df[column].astype('category').values.categories

        original = df[column].copy()
        
        # if it's a boolean or categorical
        if is_boolean_variable(vals) or "categorized" in column:
            column = [add_noise_categorical(x, original, cat_chance) for x in column]

        else:
            l = [None] * len(df)
            var = df[column].var()
            b = math.sqrt(var/2)
            for i, val in df[column].iteritems():
                l[i] = val + np.random.laplace(scale=b)
            df[column] = pd.Series(l, index = df.index)
            normalize(df, column)
        
def super_split(df, starting_string, laplace):
    categorize(df, starting_string)
    categorize(df, 'salary')
    
    if not os.path.exists("data/"):
        os.makedirs("data/")
    path = "data/" + starting_string + "_categorical/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    for column in list(df):
        normalize(df, column)
    
    if laplace:
        path = path + "laplace/"
        if not os.path.exists(path):
            os.makedirs(path)
        apply_noise(df, 0.25)
    
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