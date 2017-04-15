# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:21:29 2017

@author: Yiwei Xia
"""

from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score

def get_files(var, laplace):
    folder = "data/" + var + "_categorical"
    if laplace:
        folder += "_laplace"
    folder += "/"
    X = pd.read_csv(folder + "train_x.csv")
    Y = pd.read_csv(folder + "train_y.csv")
    X_test = pd.read_csv(folder + "test_x.csv")
    Y_test = pd.read_csv(folder + "test_y.csv")
    return X, Y, X_test, Y_test

def testo():
    X, Y, X_t, Y_t = get_files('satisfaction', False)
    Xl, Yl, Xl_t, Yl_t = get_files('satisfaction', True)
    no_l = decision_tree(X, Y)
    l = decision_tree(Xl, Yl)
    return no_l, l
    
def decision_tree(X_train, Y_train):

    clf = tree.DecisionTreeClassifier()
    scores_decision_tree = cross_val_score(estimator=clf, X=X_train, y=Y_train['satisfaction_level'],cv= 5)
    return scores_decision_tree