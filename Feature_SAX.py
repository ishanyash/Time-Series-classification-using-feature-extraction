# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:39:03 2019

@author: Ishan Yash
"""
import numpy as np
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from tslearn.piecewise import PiecewiseAggregateApproximation
from sklearn import svm
from tqdm import tqdm

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

sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
Xtrain_sax = sax.inverse_transform(sax.fit_transform(X_train))
Xtest_sax = sax.inverse_transform(sax.fit_transform(X_test))

test =Xtest_sax[:,:,0]
train = Xtrain_sax[:,:,0]

clf = svm.SVC(gamma='scale')
tqdm(clf.fit(train, y_train))

print(dataset)
print('Accuracy: ',clf.score(test,y_test))

#train = Xtrain_sax.resize(y_train.shape[0],int(len(Xtrain_sax)/y_train.shape[0]))
#test = Xtest_sax.resize(y_test.shape[0],int(len(Xtest_sax)/y_test.shape[0]))


