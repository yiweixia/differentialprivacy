# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:21:29 2017

@author: Yiwei Xia
"""

from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets



# import some data to play with
X = pd.read_csv("data/salary_categorical/train_x.csv")
Y = pd.read_csv("data/salary_categorical/train_y.csv")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)