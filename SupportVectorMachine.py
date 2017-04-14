'''
Created on Apr 12, 2017

@author: zjwar_000
'''
import numpy as np
import ParsingCsv as pcsv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import confusion_matrix,\
    precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import itertools

def svmTests():
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
    
    #Normalize input data, norms are 'l1', 'l2', or 'max'
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    
    
    X_train_laplace = pcsv.addLaplaceNoise(X_train)
    
    
    #mask y to create 1 and 0 classes based on threshold
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0
    
    
    #set up and train classifier using various kernels
    clf = SVC(kernel= 'rbf', gamma='auto')
    scoresGauss = cross_val_score(estimator=clf, X = X_train_laplace, y = y_train, cv = 5)

    clf = SVC(kernel= 'sigmoid', gamma='auto')
    scoresSigmoid = cross_val_score(estimator=clf, X = X_train_laplace, y = y_train, cv = 5)    

    clf = SVC(kernel= 'linear', gamma='auto', max_iter=5000)
    scoresLinear = cross_val_score(estimator=clf, X = X_train_laplace, y = y_train, cv = 5)
    
    clf = SVC(kernel= 'poly',degree=2, gamma='auto')
    scoresQuad = cross_val_score(estimator=clf, X = X_train_laplace, y = y_train, cv = 5)
    
    clf = SVC(kernel= 'poly',degree=3, gamma='auto')
    scoresPoly = cross_val_score(estimator=clf, X = X_train_laplace, y = y_train, cv = 5)

   
    #accuracies = {"Gauss":np.mean(scoresGauss), "Sigmoid":np.mean(scoresSigmoid), "Linear":np.mean(scoresLinear), "Quadratic":np.mean(scoresQuad),"Cubic":np.mean(scoresPoly)}
    print "Gauss, Sigmoid, Linear, Quad, Cubic"
    print np.mean(scoresGauss)
    print np.mean(scoresSigmoid)
    print np.mean(scoresLinear)
    print np.mean(scoresQuad) 
    print np.mean(scoresPoly)
    
        
def svm(useLaplace = False):
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
    
    #Normalize input data, norms are 'l1', 'l2', or 'max'
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    
    if useLaplace:
        X_train = pcsv.addLaplaceNoise(X_train)
        title = "SVM Learning Curve with Laplace noise"
    else:
        title = "SVM Learning Curve"
    
    
    #mask y to create 1 and 0 classes based on threshold
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0
    
    
    #set up and train classifier using various kernels
    clf = SVC(kernel= 'rbf', gamma='auto')
    scoresGauss = cross_val_score(estimator=clf, X = X_train, y = y_train, cv = 5)
    predicted = cross_val_predict(clf, X_train, y_train,cv=5)
    
    cv =20
    learningCurve(clf, X_train, y_train,cv, 1, title)
    
    print "\nK-fold accuracy"
    print np.mean(scoresGauss)
    
    print "\nPredicted accuracy"
    print accuracy_score(y_train, predicted)
    
    print "\nConfusion Matrix"
    cm =  confusion_matrix(y_train,predicted)
    print cm
    
    print "\nPrecision, Recall, and F1-Scores"
    metrixs = np.reshape(precision_recall_fscore_support(y_train,predicted),(4,2))
    print metrixs[0:3,:]

    plotConfusionMatrix(cm)

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
    

svm(False)
