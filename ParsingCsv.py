'''
Created on Apr 13, 2017

@author: zjwar_000
'''
'''
Created on Apr 11, 2017

@author: zjwar_000
'''
import numpy as np
import pandas as pd
import random

#code to filter data
def processData(input_file):
    df = pd.read_csv(input_file)
    for column in ['sales', 'salary']:
        dummies = pd.get_dummies(df[column])
        df[dummies.columns] = dummies

    del df['salary']
    
    df.to_csv("Filtered_Data.csv")
    return "Filtered_Data.csv"

#loading processed data into numpy array
def loadData(input_file):
    processed_data = processData(input_file)
    data_file = pd.read_csv(processed_data,sep=',')
    return data_file.values

#separate data into train and test
#cv splits are done 
def createMatrix(data_file):
    #80, 20 split
    train, test = np.split(data_file, [int(.8*len(data_file))])
    return train, test 

def createArray(file1, file2):
    data_file_train = pd.read_csv(file1, sep=',')
    data_file_test = pd.read_csv(file2, sep=',')

    return data_file_train.values,data_file_test.values

#should be down after normalization
def addLaplaceNoise(data, scale, chance):
    #for i in range(np.size(data, axis=0)):
    for i in range(0,4):
        for j in range(np.size(data,axis=1)):
            val = np.random.laplace(scale=scale)
            if data[i][j] + val < 0:
                data[i][j]-= val
            else:
                data[i][j] += val
    for j in range(4,np.size(data,axis=0)):
        for j in range(np.size(data,axis=1)):
            if random.random() < chance:
                data[i][j] = random.choice(data[j])
    return data   
