# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:05:51 2019

@author: Ishan Yash
"""

from tslearn.metrics import cdist_dtw
import numpy as np
from sklearn import svm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

print('Feature-ED-DTW-SAX-PAA-SVM')
nps = input("Enter the number of segments for PAA: ")
n_paa_segments = int(nps) #number of segments for PAA
nss = input('Enter the number if segments for SAX: ')
n_sax_symbols = int(nss)

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
ds = input('Enter the Time series Dataset: ')
dataset = str(ds) 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
X_test, y_test = fulltest[:, 1:], fulltest[:, 0]

#Feature extraction DTW

DTW_train= cdist_dtw(X_train,X_train)
DTW_test = cdist_dtw(X_test, X_train)  

#Feature extraction ED

EDist_test = []
for i in range(len(y_test)):
    for j in range(len(y_train)):
        dist = np.sqrt(np.sum((np.array(X_test[i,:])-np.array(X_train[j,:]))**2))
        EDist_test.append(dist)
        
EDist_train = []
for i in range(len(y_train)):
    for j in range(len(y_train)):
        dist1 = np.sqrt(np.sum((np.array(X_train[i,:])-np.array(X_train[j,:]))**2))
        EDist_train.append(dist1)
        
EDist_train = np.array(EDist_train)
EDist_train.resize(y_train.shape[0],int(len(EDist_train)/y_train.shape[0]))
EDist_test = np.array(EDist_test)
EDist_test.resize(y_test.shape[0],int(len(EDist_test)/y_test.shape[0]))

#PAA transform + PAA feature extraction

paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
Xtrain_paa = paa.inverse_transform(paa.fit_transform(X_train))
Xtest_paa = paa.inverse_transform(paa.fit_transform(X_test))

PAA_test =Xtest_paa[:,:,0]
PAA_train = Xtrain_paa[:,:,0]

'''
#PAA distance calculation

PAADist_train = []
PAADist_test = []

for i in range(len(y_train)):
    for j in range(len(y_train)):
        dist3 = paa.distance(Xtrain_paa[i,:],Xtest_paa[j,:])
        PAADist_train.append(dist3)

for i in range(len(y_test)):
    for j in range(len(y_train)):
        dist4 = paa.distance(Xtest_paa[i,:],Xtrain_paa[j,:])
        PAADist_test.append(dist4)   

PAADist_train = np.array(PAADist_train)
PAADist_train.resize(y_train.shape[0],int(len(PAADist_train)/y_train.shape[0]))
PAADist_test = np.array(PAADist_test)
PAADist_test.resize(y_test.shape[0],int(len(PAADist_test)/y_test.shape[0]))
'''
#SAX Transform + SAX feature extraction

sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
Xtrain_sax = sax.inverse_transform(sax.fit_transform(X_train))
Xtest_sax = sax.inverse_transform(sax.fit_transform(X_test))

SAX_test = Xtest_sax[:,:,0]
SAX_train = Xtrain_sax[:,:,0]
'''
#SAX distance calculation
SAXDist_train = []
SAXDist_test = []

for i in range(len(y_train)):
    for j in range(len(y_train)):
        dist3 = sax.distance(Xtrain_sax[i,:],Xtest_sax[j,:])
        SAXDist_train.append(dist3)

for i in range(len(y_test)):
    for j in range(len(y_train)):
        dist4 = sax.distance(Xtest_sax[i,:],Xtrain_sax[j,:])
        SAXDist_test.append(dist4)

SAXDist_train = np.array(SAXDist_train)
SAXDist_train.resize(y_train.shape[0],int(len(SAXDist_train)/y_train.shape[0]))
SAXDist_test = np.array(SAXDist_test)
SAXDist_test.resize(y_test.shape[0],int(len(SAXDist_test)/y_test.shape[0]))   
'''

#Feature concatenation

a = np.concatenate((PAA_test,SAX_test),axis=1)
b = np.concatenate((EDist_test,DTW_test),axis=1)
test = np.concatenate((a,b),axis=1)

c = np.concatenate((PAA_train,SAX_train),axis=1)
d = np.concatenate((EDist_train,DTW_train),axis=1)
train =  np.concatenate((c,d),axis=1)

train_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(train)
test_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(test)

train_concat_mv.resize(train.shape[0],train.shape[1])
test_concat_mv.resize(test.shape[0],test.shape[1])

#SVM

clf = svm.SVC(gamma='scale')
clf.fit(train_concat_mv, y_train)

print(dataset)
print('Accuracy: ',clf.score(test_concat_mv,y_test))




