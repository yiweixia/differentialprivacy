'''
Created on Apr 11, 2017

@author: zjwar_000
'''
import ParsingCvFiles as pcsv
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from _random import Random



##tested using old data filtering


def Bayes(input_file):
    all_data = pcsv.loadData(input_file)
    train, test = pcsv.createMatrix(all_data)
    
    np.random.shuffle(train)
    
    #Separate into X and y
    X_train = train[:,2:np.size(train,axis = 1)]
    y_train = train[:,1] 
    
    #Normalize input data, 
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    #mask y to create 1 and 0 classes based on threshold
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0
    
    
    gnb = GaussianNB()
    scoresG = cross_val_score(estimator=gnb, X=X_train, y=y_train,cv= 5)
    
    print "Gaussian Scores"
    print scoresG
    
    mnb = MultinomialNB()
    scoresM = cross_val_score(estimator=mnb, X = X_train, y = y_train, cv = 5)
    
    print "Multinomial Scores"
    print scoresM
    
    bnb = BernoulliNB()
    scoresB = cross_val_score(estimator=bnb, X = X_train, y = y_train, cv = 5)
    
    print "Bernoulli Scores"
    print scoresB
    

Bayes("HR_comma_sep.csv")
