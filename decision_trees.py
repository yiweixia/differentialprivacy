# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:21:29 2017

@author: Yiwei Xia
"""

from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import confusion_matrix,\
    precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve

def get_files(var, laplace):
    
    folder = "data/" + var + "_categorical"
    if laplace:
        folder += "_laplace"
    folder += "/"
    
    X = pd.read_csv(folder + "train_x.csv")
    X = X.drop(X.columns[0], axis=1)
    
    Y = pd.read_csv(folder + "train_y.csv")
    Y = Y['satisfaction_level']

    X_test = pd.read_csv(folder + "test_x.csv")
    X_test = X_test.drop(X_test.columns[0], axis=1)
    
    Y_test = pd.read_csv(folder + "test_y.csv")
    Y_test = Y_test['satisfaction_level']
    
    return X, Y, X_test, Y_test
    
def testo():
    X, Y, X_t, Y_t = get_files('satisfaction', False)
    Xl, Yl, Xl_t, Yl_t = get_files('satisfaction', True)
    
    get_info(X, Y, X_t, Y_t, False)
    get_info(Xl, Yl, Xl_t, Yl_t, True)
    
def get_info(X, Y, X_t, Y_t, l):
    
    if l:
        title = "privatized"
    else:
        title = "not privatized"
        
    print(title)
    
    clf = tree.DecisionTreeClassifier()
    
    cv = 20
    learningCurve(clf, X, Y, cv, 1, title)
    
    clf.fit(X, Y)
    Y_p = clf.predict(X_t)
    
    print("k-fold average")
    scores_decision_tree = cross_val_score(estimator=clf, X=X, y=Y,cv= 5)
    print(str(np.average(scores_decision_tree)))

    print("test accuracy:")
    accuracy = accuracy_score(Y_t, Y_p)
    print(str(accuracy))
    
    cm = confusion_matrix(Y_t, Y_p)
    plotConfusionMatrix(cm)
    
    metrix = precision_recall_fscore_support(Y_t, Y_p, labels=[0,1])
    print(np.around(np.matrix(metrix)), decimals = 2)
    
def epsilon_test(laplace):
    
    if (laplace):
        X, Y, X_t, Y_t = get_files('satisfaction', True)
    else:
        X, Y, X_t, Y_t = get_files('satisfaction', False)
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)
    
    print("k-fold average")
    scores_decision_tree = cross_val_score(estimator=clf, X=X, y=Y,cv= 5)
    average = np.average(scores_decision_tree)
    print(str(average))
    
    return average

    
def learningCurve(estimator, X, Y, cv, n_jobs,title):
    #change linespace to modify points in graph
    train_sizes=np.linspace(.05, 1.0, 20)
    
    #initialize graph
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    #calculate learning curve points for 20 portions of the data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    
    #calculate means and std to put into graph
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    
    
    #add mean and std to the graph with transparent filling
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,
                     train_scores_mean+train_scores_std, alpha =0.1,
                     color = 'r')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,
                     test_scores_mean+test_scores_std, alpha =0.1,
                     color = 'r')
    
    #plot values of curve for both lines
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean,'o-',color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    
    #plot the graph
    plt.show()
    return

    
def plotConfusionMatrix(cm, classes=2):
    
    #plot confusion matrix and set labels
    plt.matshow(cm)
    plt.title("Confusion Matrix")
    #add color bar for reference
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel('Predicted label')
    return plt
    