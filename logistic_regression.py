# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:29:18 2017

@author: dkim63
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score
from ggplot import *

k = 10

def load_data(prefix):
    path = 'data\\job_categorical\\laplace\\' + prefix
    data = pd.read_csv(path + '_x.csv')
    X = np.array(data)[:, 1:]

    data = pd.read_csv(path + '_y.csv')
    tmpY = np.array(data)[:, 1:]
    Y  = [val for sublist in tmpY for val in sublist]
    return X, np.array(Y)

def kfold_cross_validation(trainX, trainY):
    logreg = linear_model.LogisticRegression()
    scores = cross_val_score(logreg, trainX, trainY, cv=k)
    print scores
    return logreg
#==============================================================================
#     section_size = int(round(len(trainX) * k))
#     for i in range(int(k*100)):
#         X = np.concatenate([trainX[:i*section_size], trainX[(i+1)*section_size:]])
#         Y = np.concatenate([trainY[:i*section_size], trainY[(i+1)*section_size:]])
#         validateX = trainX[i*section_size:(i+1)*section_size]
#         validateY = trainY[i*section_size:(i+1)*section_size]
#         logreg.fit(X, Y)
#         print i
#         print logreg.score(X, Y)
#==============================================================================

def plot(logreg, testX, testY):
    preds = logreg.predict_proba(testX)[:,1]
    fpr, tpr, _ = metrics.roc_curve(testY, preds)

    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    ggplot(df, aes(x='FPR', y='TPR')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

def main():
    trainX, trainY = load_data('train')
    testX, testY = load_data('test')
    logreg = kfold_cross_validation(trainX, trainY)
    #plot(logreg, testX, testY)

if __name__ == "__main__":
    main()