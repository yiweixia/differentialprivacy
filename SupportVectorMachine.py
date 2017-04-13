'''
Created on Apr 12, 2017

@author: zjwar_000
'''
import numpy as np
import ParsingCvFiles as pcsv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import normalize


##tested using old data filtering technique


def SVM(X, y, C, degree, kernel):
    clf = SVC(C=C,degree=degree,gamma='auto',kernel=kernel)
    clf.fit(X,y)
    #clf.predict([[array]])
    
def svm(input_file):
    all_data = pcsv.loadData(input_file)
    train, test = pcsv.createMatrix(all_data)
    
    np.random.shuffle(train)
     
    #Separate into X and y
    X_train = train[:,2:np.size(train,axis = 1)]
    y_train = train[:,1] 
    
    #Normalize input data, norms are 'l1', 'l2', or 'max'
    normalize(X_train, norm='max', axis= 0, copy = False)
    
    #mask y to create 1 and 0 classes based on threshold
    y_train[train[:,1]>0.5] = 1
    y_train[~(train[:,1]>0.5)] = 0
    
   
    bestGauss = np.zeros((np.size(X_train, axis=1),5))
    bestSig = np.zeros((np.size(X_train, axis=1),5))
    bestLinear = np.zeros((np.size(X_train, axis=1),5))
    bestQuad = np.zeros((np.size(X_train, axis=1),5))
    bestCubic = np.zeros((np.size(X_train, axis=1),5))
    
    for j in range(0,np.size(X_train,axis=1)):
        for i in range(1,6):
            #modify feature j by power i, powers ranging 1 to 5
            X_train_modified = X_train
            X_train_modified[:,j] = np.power(X_train_modified[:,j],i)
    
            #set up and train classifier using various kernels
            clf = SVC(kernel= 'rbf', gamma='auto')
            scoresGauss = cross_val_score(estimator=clf, X = X_train_modified, y = y_train, cv = 5)
    
            clf = SVC(kernel= 'sigmoid', gamma='auto')
            scoresSigmoid = cross_val_score(estimator=clf, X = X_train_modified, y = y_train, cv = 5)    
    
            clf = SVC(kernel= 'linear', gamma='auto', max_iter=5000)
            scoresLinear = cross_val_score(estimator=clf, X = X_train_modified, y = y_train, cv = 5)
            
            clf = SVC(kernel= 'poly',degree=2, gamma='auto')
            scoresQuad = cross_val_score(estimator=clf, X = X_train_modified, y = y_train, cv = 5)
            
            clf = SVC(kernel= 'poly',degree=3, gamma='auto')
            scoresPoly = cross_val_score(estimator=clf, X = X_train_modified, y = y_train, cv = 5)
    
            bestGauss[j,i-1] = np.mean(scoresGauss) 
            bestSig[j,i-1] = np.mean(scoresSigmoid) 
            bestLinear[j,i-1] = np.mean(scoresLinear) 
            bestQuad[j,i-1] = np.mean(scoresQuad) 
            bestCubic[j,i-1] = np.mean(scoresPoly)
             
 
    #accuracies = {"Gauss":np.mean(scoresGauss), "Sigmoid":np.mean(scoresSigmoid), "Linear":np.mean(scoresLinear), "Quadratic":np.mean(scoresQuad),"Cubic":np.mean(scoresPoly)}
    #print "Gauss, Sigmoid, Linear, Quad, Cubic"
    #print np.mean(scoresGauss) + np.mean(scoresSigmoid) + np.mean(scoresLinear) + np.mean(scoresQuad) + np.mean(scoresPoly)
    
    #best combination for gaussian
    print "Gaussian"
    Gmax = np.max(bestGauss,axis=1) #this should be the max of each row, representing the best power to choose
    print bestGauss, Gmax

    print "\n\nSigmoid"
    Smax = np.max(bestSig,axis=1)
    print bestSig, Smax
    
    print "\n\nLinear"
    Lmax = np.max(bestLinear, axis=1)
    print bestLinear, Lmax
    
    print "\n\nQuadratic"
    Qmax = np.max(bestQuad, axis=1)
    print bestQuad, Qmax
    
    print "\n\nCubic"
    Cmax = np.max(bestCubic, axis=1)
    print bestCubic, Cmax
    
      
     
svm("HR_comma_sep.csv")
