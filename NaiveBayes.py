'''
Created on Apr 11, 2017

@author: zjwar_000
'''
import ParsingCsv as pcsv
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt
from _random import Random


def BayesTests():
    train, test = pcsv.createArray()
    
    
    tr1 = train[:,0:9]
    tr2 = train[:,10:np.size(train,axis=1)]
    train = np.hstack((tr1,tr2))
    train = train.astype(float)
    
    tr1 = test[:,0:9]
    tr2 = test[:,10:np.size(test,axis=1)]
    test = np.hstack((tr1,tr2))
    test = test.astype(float)
    
    np.random.shuffle(train)
        
    #Separate into X and y
    X_train = train[:,2:np.size(train,axis = 1)]
    y_train = train[:,1] 
    
    
    #Normalize input data, 
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    X_train_laplace = pcsv.addLaplaceNoise(X_train)
    
    
    
    #mask y to create 1 and 0 classes based on threshold
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0
    
    
    gnb = GaussianNB()
    scoresG = cross_val_score(estimator=gnb, X=X_train_laplace, y=y_train,cv= 5)
    
    print "Gaussian Scores"
    print scoresG
    
    mnb = MultinomialNB()
    scoresM = cross_val_score(estimator=mnb, X = X_train_laplace, y = y_train, cv = 5)
    
    print "Multinomial Scores"
    print scoresM
    
    bnb = BernoulliNB()
    scoresB = cross_val_score(estimator=bnb, X = X_train_laplace, y = y_train, cv = 5)
    
    print "Bernoulli Scores"
    print scoresB
  
def BayesLearningCurves(useLaplace = False):
    train, test = pcsv.createArray()
    
    #remove column with categorical strings
    tr1 = train[:,0:9]
    tr2 = train[:,10:np.size(train,axis=1)]
    train = np.hstack((tr1,tr2))
    train = train.astype(float)
    
    tr1 = test[:,0:9]
    tr2 = test[:,10:np.size(test,axis=1)]
    test = np.hstack((tr1,tr2))
    test = test.astype(float)
    
    #re-randomize, just because
    np.random.shuffle(train)
        
    #Separate into X and y
    X_train = train[:,2:np.size(train,axis = 1)]
    y_train = train[:,1] 
    
    
    #Normalize input data, 
    #xi/z where z = max(xi)
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    #using Laplace transforms or not
    if useLaplace:
        X_train = pcsv.addLaplaceNoise(X_train)
        title = "Naive Bayes Learning Curve with Laplace noise"
    else:
        title = "Naive Bayes Learning Curve"
    
    
    
    #mask y to create 1 and 0 classes based on threshold
    #threshold chosen to be 0.5, others fair more poorly
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0  
    
    
    #initialize classifier
    gnb = GaussianNB()
    
    #cross validation score, 5-fold is standard
    scoresG = cross_val_score(estimator=gnb, X=X_train, y=y_train,cv= 5)
    
    #accuracy with 5-fold cv
    predicted = cross_val_predict(gnb, X_train, y_train,cv=5)
    
    
    cv = 20
    learningCurve(gnb, X_train, y_train, cv, 1, title)
    
    
    print "K-fold accuracy"
    print np.mean(scoresG)
    
    print "\nPredicted accuracies"
    print accuracy_score(y_train, predicted)
    
    print "\nConfusion Matrix"
    cm = confusion_matrix(y_train,predicted)
    print cm
    
    print "\nPrecision, Recall, and F1-Scores"
    metrixs = np.reshape(precision_recall_fscore_support(y_train,predicted),(4,2))
    print metrixs[0:3,:]
    
    plotConfusionMatrix(cm, 2)
    
    plt.show()
    return 
    

def learningCurve(estimator, X, y, cv, n_jobs,title):
    #change linespace to modify points in graph
    train_sizes=np.linspace(.05, 1.0, 20)
    
    #initialize graph
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    #calculate learning curve points for 20 portions of the data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    
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
    
    return plt

def plotConfusionMatrix(cm, classes=2):
    
    #plot confusion matrix and set labels
    plt.matshow(cm)
    plt.title("Confusion Matrix")
    #add color bar for reference
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel('Predicted label')
    return plt


BayesLearningCurves(False)
BayesLearningCurves(True)
