'''
Created on Apr 11, 2017

@author: zjwar_000
'''
import numpy as np
import pandas as pd

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

