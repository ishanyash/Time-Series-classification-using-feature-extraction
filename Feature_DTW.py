# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:03:58 2019

@author: Ishan Yash
"""

from dtw import dtw
import numpy as np
import sklearn
from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "CBF" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"


train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]

X_train = X_train.copy(order='C')
y_train = y_train.copy(order='C')
X_test = X_test.copy(order='C')
y_test = y_test.copy(order='C')

train_features_dtw = cdist_dtw(X_train,X_train)
train_features_dtw = train_features_dtw.resize(900,30)
test_features_dtw = cdist_dtw(X_train,X_test)


clf = SVC(gamma='auto',probability=True)
clf.fit(train_features_dtw, y_train)
#probab = clf.predict_proba(X_test)
score = clf.score(test_features_dtw,y_test)
print(score)

