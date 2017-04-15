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
from sklearn.preprocessing import normalize
import os
import random
import numpy as np
import math

def split_category(df):
    
    del_queue = []
    for column in list(df):
        if column.startswith('job'):
            df[column] = df[column].apply(str)
            dummies = pd.get_dummies(df[column])
            df["job_" + dummies.columns] = dummies
            del_queue.append(column)
        if column.startswith('salary'):
            df[column] = df[column].apply(str)
            dummies = pd.get_dummies(df[column])
            df["salary_" + dummies.columns] = dummies
            del_queue.append(column)
            
    for d in del_queue:
        del df[d]

    return df

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
    
    return df
    
# splits dataframe columns based on starting string
# e.g. split_xy(df, "salary") will have salary_high/medium/low as Y, and everything else as X
def split_xy(df, starting_string):
    y = df.filter(regex='^' + starting_string, axis=1)
    for column in list(df):
        if column.startswith(starting_string):
            del df[column]
    x = df
    return {"x":x, "y":y}

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
    
    cols = list(df)
    
    df = pd.DataFrame(normalize(df, axis=0, norm='max'), columns = cols)
    
    for column in list(df):
        
        vals = df[column].astype('category').values.categories

        original = df[column].copy()
        
        # if it's a boolean or categorical
        if is_boolean_variable(vals) or "categorized" in column:
            df[column] = [add_noise_categorical(x, original, cat_chance) for x in df[column]]

        else:
            normalize(df[column].values.reshape(-1, 1), norm='max')
            
            l = [None] * len(df)
            var = df[column].var()
            b = math.sqrt(var/2)
            for i, val in df[column].iteritems():
                l[i] = val + np.random.laplace(scale=b)
            df[column] = pd.Series(l, index = df.index)

    return split_category(df)
        
def g5(x):
    if x > .5:
        return 1
    else:
        return 0
            
# turns satisfaction into a boolean value either > .5 or not
def satisfaction_mask_boolean(df):
    if 'satisfaction_level' in list(df):
        return [g5(x) for x in df['satisfaction_level']]
    else:
        return 0

def super_split(df, starting_string, needs_categorizing, laplace):
    
    if needs_categorizing:
        categorize(df, starting_string)
        
    if not os.path.exists("data/"):
        os.makedirs("data/")
    path = "data/" + starting_string + "_categorical"
    if laplace:
        path += "_laplace"
    path += "/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    tt_split = split_train_test(df)
    train = tt_split['train']
    test = tt_split['test']
    train_xy = split_xy(train, starting_string)
    test_xy = split_xy(test, starting_string)

    train_xy['x'].to_csv(path + "/train_x.csv")
    train_xy['y'].to_csv(path + "/train_y.csv")
    test_xy['x'].to_csv(path + "/test_x.csv")
    test_xy['y'].to_csv(path + "/test_y.csv")

def satisfaction(df, cat_chance):
    df['satisfaction_level'] = satisfaction_mask_boolean(df)
    super_split_satisfaction(df)
    super_split_noisy_satisfaction(df, cat_chance)
    
def super_split_satisfaction(df):
    super_split(df, "satisfaction", False, False)
    
def super_split_noisy_satisfaction(df, cat_chance):
    df = apply_noise(df, cat_chance)
    super_split(df, "satisfaction", False, True)
    
def super_split_job():
    super_split(df, "job", False)
    
def refresh():
    df = pd.read_csv("raw.csv")
    df = df.rename(columns = {'sales': 'job'})
    df = split_category(df)
    df.to_csv("processed.csv")
    return df
    
def nosat(cat):
    print("generating for " + str(cat))
    df = refresh()
    satisfaction(df, cat)
    

