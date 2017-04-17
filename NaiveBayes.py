'''
Created on Apr 11, 2017

@author: zjwar_000
'''
import ParsingCsv as pcsv
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, learning_curve, cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


def BayesTests():
    train, test = pcsv.createArray("train_80.csv","test_20.csv")
    
    
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
  
def Bayes(use_laplace = False, epsilon =1.0, chance=0.1, plot_learning_curve=False, is_test=False, cv =20):
    #extract data from csv
    path1 = "satisfaction_categorical\\"
    test_x_raw = pd.read_csv(path1+"test_x.csv", sep=',').values
    test_y_raw = pd.read_csv(path1+"test_y.csv", sep=',').values
    
    train_x_raw = pd.read_csv(path1+"train_x.csv", sep=',').values
    train_y_raw = pd.read_csv(path1+"train_y.csv", sep=',').values


    #Modify data into usable forms
    train_x = train_x_raw[:,1:-1]    
    train_y_raw = train_y_raw[:,1]
    train_y = np.reshape(train_y_raw, (np.size(train_y_raw),))
    
    test_x = test_x_raw[:,1:-1]
    test_y_raw = test_y_raw[:,1]
    test_y = np.reshape(test_y_raw, (np.size(test_y_raw),))
    
    #calculate delta for laplace mechanism
    deltaf = np.max(np.sum(np.abs(train_x), axis=1))
    lambd = deltaf/epsilon
    
    if use_laplace:
        train_x = pcsv.addLaplaceNoise(train_x, lambd, chance)

    if is_test:
        #fit with train data, use this to predict test data
        gnb = GaussianNB()
        gnb.fit(train_x, train_y)
        predicted = gnb.predict(test_x)
        
        print "\nPredicted accuracy"
        print accuracy_score(test_y, predicted)
        
        print "\nConfusion Matrix"
        cm =  confusion_matrix(test_y, predicted)
        print cm
        
        print "\nPrecision, Recall, and F1-Scores"
        metrixs = np.reshape(precision_recall_fscore_support(test_y,predicted),(4,2))
        print metrixs[0:3,:]
    
        plotConfusionMatrix(cm)
        plt.show()
        
        return 
    else:
    
        #initialize classifier
        gnb = GaussianNB()
        
        #cross validation score, 5-fold is standard
        scoresG = cross_val_score(estimator=gnb, X=train_x, y=train_y,cv= 5)
        
        #accuracy with 5-fold cv
        predicted = cross_val_predict(gnb, train_x, train_y,cv=5)
        
        if plot_learning_curve:
            if use_laplace:
                title = "Naive Bayes Learning Curve with Laplace noise"
            else:
                title = "Naive Bayes Learning Curve"
            learningCurve(gnb, train_x, train_y, cv, 1, title)
        
        
        print "K-fold accuracy"
        print np.mean(scoresG)
        
        print "\nPredicted accuracies"
        print accuracy_score(train_y, predicted)
        
        print "\nConfusion Matrix"
        cm = confusion_matrix(train_y,predicted)
        print cm
        
        print "\nPrecision, Recall, and F1-Scores"
        metrixs = np.reshape(precision_recall_fscore_support(train_y,predicted),(4,2))
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



is_test = False
learn_curve = True
eps = 3.0
#leave chance at 0.1 and cv at 20

print "Epsilon = " + str(eps)
print "Without Laplace"
Bayes(use_laplace=False, epsilon = eps, chance= 0.1, plot_learning_curve=learn_curve, is_test=is_test, cv =20)

print "With Laplace"
Bayes(use_laplace=True, epsilon = eps, chance= 0.1, plot_learning_curve=learn_curve, is_test=is_test, cv =20)

plt.plot()