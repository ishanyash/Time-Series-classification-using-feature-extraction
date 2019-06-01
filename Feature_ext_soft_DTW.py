# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:05:51 2019

@author: Ishan Yash
"""


from tslearn.metrics import cdist_soft_dtw_normalized
from tslearn.metrics import cdist_dtw
import numpy as np
from sklearn.svm import SVC

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "CBF" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"


train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]

#Feature extraction 
train_softdtw= cdist_soft_dtw_normalized(X_train,X_train)
test_softdtw = cdist_soft_dtw_normalized(X_train, X_test)

#train_dtw= cdist_dtw(X_train,X_train)
#
#train_dtw = np.transpose(train)
#test_dtw = np.transpose(test)

#train_softdtw = np.transpose(train)
#test_softdtw = np.transpose(test)

#test = np.vstack((test_softdtw,test_dtw))
#train = np.vstack((train_dtw,train_softdtw))

train = np.transpose(train_softdtw)
test = np.transpose(test_softdtw)

clf = SVC(gamma='auto')
clf.fit(train, y_train)
print(clf.score(test, y_test))



        

       




