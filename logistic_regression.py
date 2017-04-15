# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:29:18 2017

@author: dkim63
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict

def load_data(path):
    data = pd.read_csv(path + '_x.csv')
    X = np.array(data)[:, 1:]

    data = pd.read_csv(path + '_y.csv')
    tmpY = np.array(data)[:, 1:]
    Y  = [val for sublist in tmpY for val in sublist]
    
    return X, np.array(Y)

def accuracy(pred, Y):
    return float(np.sum(pred == Y))/len(pred) * 100

def kfold_cross_validation(trainX, trainY, k):
    logreg = linear_model.LogisticRegression()
    pred = cross_val_predict(logreg, X = trainX, y = trainY, cv=k)
    logreg.fit(trainX, trainY)
    print "On anonymized training set:" + str(accuracy(pred, trainY))
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

def test_on_non_noisy(logreg, testX, testY):
    pred = logreg.predict(testX)
    print "On non-anomymized test set: " + str(accuracy(pred, testY))

def plot(logreg, testX, testY):
    preds = logreg.predict_proba(testX)[:,1]
    fpr, tpr, _ = metrics.roc_curve(testY, preds)

    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    df.plot(x='fpr', y='tpr', figsize=(12,10), title="With Laplace Mechanism and Category Chance = 0")

def main():
    trainX, trainY = load_data('data\\satisfaction_categorical\\train')
    testX, testY = load_data('data\\satisfaction_categorical_laplace\\train')
    logreg = kfold_cross_validation(trainX, trainY, 5)
    test_on_non_noisy(logreg, testX, testY)
    plot(logreg, testX, testY)

if __name__ == "__main__":
    main()