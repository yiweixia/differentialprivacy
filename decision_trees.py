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

aX = pd.read_csv("data/salary_categorical/train_x.csv")
aY = pd.read_csv("data/salary_categorical/train_y.csv")
aX_p = pd.read_csv("data/salary_categorical/test_x.csv")
aY_t = pd.read_csv("data/salary_categorical/test_y.csv")

# X: training X
# Y: training Y
# X_p: validation X
# Y_t: validation Y
# predicted: predicted values for Y
# difference: Y_t = predicted -> 1, Y_t != predicted -> 0 
def run_decision_tree(X, Y, X_p, Y_t):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    Y_p = clf.predict(X_p)
    
    Y_p = pd.DataFrame(Y_p)
    
    diff = Y_p[1] - Y_t['salary_categorized']
    
    diff.to_csv("data/results/decision_trees.csv")
    
    def one_zero(x):
        if x != 0:
            return 0
        else:
            return 1
        
    diff = diff.apply(one_zero)
    
    return{"predicted":Y_p, "difference":diff}

def testo():
    return run_decision_tree(aX, aY, aX_p, aY_t)