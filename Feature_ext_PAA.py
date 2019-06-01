# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:05:51 2019

@author: Ishan Yash
"""
from tslearn.metrics import soft_dtw
from tslearn.metrics import cdist_soft_dtw
from tslearn.metrics import cdist_gak
from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw
from tslearn.svm import TimeSeriesSVC
from tslearn.piecewise import PiecewiseAggregateApproximation 

import numpy as np
import sklearn
from sklearn.svm import SVC

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "ECG200" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"


train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]
#print(type(y_test[5]))

feature_test_DTW =[]
feature_train_DTW=[]

model = PiecewiseAggregateApproximation(n_segments = 3)
X_train = model.fit_transform(X_train)
X_test = model.fit_transform(X_test)

for i in range(y_train.shape[0]):
    x_train1 = X_train[i]
    for j in range(y_train.shape[0]):
        x_train2 = X_train[j]
        d1 = model.distance(x_train1,x_train2)
        feature_train_DTW.append(d1)
     
for i in range(y_train.shape[0]):
    x_train = X_train[i]
    for j in range(y_test.shape[0]):
        x_test = X_test[j]
        d= model.distance(x_test,x_train)
        #print(d)
        feature_test_DTW.append(d)

feature_test_DTW = np.asmatrix(feature_test_DTW)
feature_train_DTW = np.asmatrix(feature_train_DTW)

feature_test_DTW.resize(X_test.shape[0],int(feature_test_DTW.shape[1]/X_test.shape[0]))
feature_train_DTW.resize(X_train.shape[0],int(feature_train_DTW.shape[1]/X_train.shape[0]))
        
clf = SVC(gamma='auto')
clf.fit(feature_train_DTW, y_test)
print(clf.score(feature_test_DTW, y_train))


        

       




