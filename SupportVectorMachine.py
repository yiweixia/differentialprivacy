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
import pandas as pd
from numpy.linalg.linalg import norm


#compare carious svm implementations, with OLD csv parsing technique
def svmTests():
    train, test = pcsv.createArray("train_80.csv","test_20.csv")
    
    #adjust arrays
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

    #accuracies 
    print "Gauss, Sigmoid, Linear, Quad, Cubic"
    print np.mean(scoresGauss)
    print np.mean(scoresSigmoid)
    print np.mean(scoresLinear)
    print np.mean(scoresQuad) 
    print np.mean(scoresPoly)
    
        
def svm(use_laplace = False, epsilon = 1.0, chance = 0.1, plot_learning_curve = False, is_test = False, cv = 5):
    
    #read csv files into numpy arrays
    path1 = "satisfaction_categorical\\"
    
    test_x_raw = pd.read_csv(path1+"test_x.csv", sep=',').values
    test_y_raw = pd.read_csv(path1+"test_y.csv", sep=',').values

    train_x_raw = pd.read_csv(path1+"train_x.csv", sep=',').values
    train_y_raw = pd.read_csv(path1+"train_y.csv", sep=',').values
    
    #modifiy arrays into workable forms
    train_x = train_x_raw[:,1:-1]    
    train_y_raw = train_y_raw[:,1]
    train_y = np.reshape(train_y_raw, (np.size(train_y_raw),))
    
    test_x = test_x_raw[:,1:-1]
    test_y_raw = test_y_raw[:,1]
    test_y = np.reshape(test_y_raw, (np.size(test_y_raw),))
     
    #calculate for laplace mechanism 
    deltaf = np.max(np.sum(np.abs(train_x), axis=1))
    lambd = deltaf/epsilon
    
    #add noise or not
    if use_laplace:
        train_x = pcsv.addLaplaceNoise(train_x, lambd, chance )
    
    #run training or testing
    if is_test:
        clf = SVC(kernel= 'rbf', gamma='auto')
        clf.fit(train_x,train_y)
        predicted = clf.predict(test_x)
        
        print "\nPredicted accuracy"
        print accuracy_score(test_y, predicted)
        
        print "\nConfusion Matrix"
        cm =  confusion_matrix(test_y,predicted)
        print cm
        
        print "\nPrecision, Recall, and F1-Scores"
        metrixs = np.reshape(precision_recall_fscore_support(test_y,predicted),(4,2))
        print metrixs[0:3,:]
    
        plotConfusionMatrix(cm)
        plt.show()
        
    else:    
        clf = SVC(kernel= 'rbf', gamma='auto')
        scoresGauss = cross_val_score(estimator=clf, X = train_x, y = train_y, cv = 5)
        predicted = cross_val_predict(clf, train_x, train_y,cv=5)
    
        if plot_learning_curve:
            if use_laplace:
                title = "SVM Learning Curve with Laplace noise"
            else:
                title = "SVM Learning Curve"
            learningCurve(clf, train_x, train_y,cv, 1, title)
        
        print "\nK-fold accuracy"
        print np.mean(scoresGauss)
        
        print "\nPredicted accuracy"
        print accuracy_score(train_y, predicted)
        
        print "\nConfusion Matrix"
        cm =  confusion_matrix(train_y,predicted)
        print cm
        
        print "\nPrecision, Recall, and F1-Scores"
        metrixs = np.reshape(precision_recall_fscore_support(train_y,predicted),(4,2))
        print metrixs[0:3,:]
    
        #plotConfusionMatrix(cm)
        #plt.show()
        return plt
    
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
    

is_test = True
learn_curve = False
eps = 2.0
#keep chance at 0.1, cv at 20

print "Without Laplace"
svm(use_laplace=False, epsilon = eps, chance= 0.1, plot_learning_curve=learn_curve, is_test=is_test, cv =20)


print "With Laplace, epsilon = "+ str(eps)
svm(use_laplace =True,epsilon=eps, chance= 0.1, plot_learning_curve=learn_curve, is_test=is_test, cv =20)

plt.show()