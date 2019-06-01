# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:05:51 2019

@author: Ishan Yash
"""
from tslearn.metrics import dtw

import numpy as np

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "CBF" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"


train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]
#print(type(y_test[5]))




for i in range(y_train.shape[0]):
    train_DTW=[]
    x_train = X_train[i]
    for j in range(y_train.shape[0]):
        x_train1 = X_train[j]
        d1 = dtw(x_train,x_train1)
        train_DTW.append(d1)
     
for i in range(y_train.shape[0]):
    test_DTW =[]
    x_train = X_train[i]
    for j in range(y_test.shape[0]):
        x_test = X_test[j]
        d= dtw(x_train, x_test)
        #print(d)
        test_DTW.append(d)


print(len(feature_train_DTW))
print(len(feature_train_DTW))