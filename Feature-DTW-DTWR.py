# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:17:10 2019

@author: Ishan Yash 
"""

import numpy as np
from sklearn import svm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from pyts.metrics import dtw
from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                              _return_path, _multiscale_region)

print('Feature-DTW_DTWR')



PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
ds = input('Enter the Time series Dataset: ')
dataset = str(ds) 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
X_test, y_test = fulltest[:, 1:], fulltest[:, 0]

window_size = int(X_test.shape[1]-1)
#Feature extraction DTW
# Dynamic Time Warping: classic
DTW_Classic_test = []
DTW_Classic_train = []
for i in range(len(X_test)):
    for j in range(len(X_train)):
        dtw_classic, path_classic = dtw(X_test[i], X_train[j], dist='square',
                                method='classic', return_path=True)
        DTW_Classic_test.append(dtw_classic)

for i in range(len(X_train)):
    for j in range(len(X_train)):
        dtw_classic, path_classic = dtw(X_train[i], X_train[j], dist='square',
                                method='classic', return_path=True)
        DTW_Classic_train.append(dtw_classic)

DTW_Classic_train = np.array(DTW_Classic_train)
DTW_Classic_train.resize(y_train.shape[0],int(len(DTW_Classic_train)/y_train.shape[0]))
DTW_Classic_test = np.array(DTW_Classic_test)
DTW_Classic_test.resize(y_test.shape[0],int(len(DTW_Classic_test)/y_test.shape[0]))

#DTW sakoechiba
DTW_sakoe_test = []
DTW_sakoe_train = []
for i in range(len(X_test)):
    for j in range(len(X_train)):
        dtw_sakoechiba, path_sakoechiba = dtw(X_test[i], X_train[j], dist='square', method='sakoechiba',options={'window_size': window_size}, return_path=True)
        DTW_sakoe_test.append(dtw_sakoechiba)

for i in range(len(X_train)):
    for j in range(len(X_train)):
        dtw_sakoechiba, path_sakoechiba = dtw(X_train[i], X_train[j], dist='square', method='sakoechiba',options={'window_size': window_size}, return_path=True)
        DTW_sakoe_train.append(dtw_sakoechiba)

DTW_sakoe_train = np.array(DTW_sakoe_train)
DTW_sakoe_train.resize(y_train.shape[0],int(len(DTW_sakoe_train)/y_train.shape[0]))
DTW_sakoe_test = np.array(DTW_sakoe_test)
DTW_sakoe_test.resize(y_test.shape[0],int(len(DTW_sakoe_test)/y_test.shape[0]))


test_DTW_DTWR = np.concatenate((DTW_sakoe_test,DTW_Classic_test),axis=1)

train_DTW_DTWR = np.concatenate((DTW_Classic_train,DTW_sakoe_train),axis=1)


train_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(train_DTW_DTWR)
test_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(test_DTW_DTWR)

train_concat_mv.resize(train_DTW_DTWR.shape[0],train_DTW_DTWR.shape[1])
test_concat_mv.resize(test_DTW_DTWR.shape[0],test_DTW_DTWR.shape[1])

#SVM

clf = svm.SVC(gamma='scale')
clf.fit(train_concat_mv, y_train)

print(dataset)
print('Accuracy: ',clf.score(test_concat_mv,y_test))
print('Error rate:', 1-clf.score(test_concat_mv,y_test))
