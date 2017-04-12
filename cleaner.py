# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:02:23 2017

@author: Yiwei Xia
"""

import numpy as np
import pandas as pd

df = pd.read_csv("raw.csv")

for column in ['sales', 'salary']:
    dummies = pd.get_dummies(df[column])
    df[dummies.columns] = dummies

del df['salary']

df.to_csv("processed.csv")
