# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:18:33 2019

@author: Ishan Yash
"""

import numpy as np
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from tslearn.piecewise import PiecewiseAggregateApproximation

print('Feature-PAA')
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

#PAA transform + PAA feature extraction

paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
Xtrain_paa = paa.inverse_transform(paa.fit_transform(X_train))
Xtest_paa = paa.inverse_transform(paa.fit_transform(X_test))



'''
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