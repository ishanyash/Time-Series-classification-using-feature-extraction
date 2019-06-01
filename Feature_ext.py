# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:05:51 2019

@author: Ishan Yash
"""

from tslearn.metrics import cdist_dtw
import numpy as np
from sklearn.svm import SVC

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "CBF" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"


fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
X_test, y_test = fulltest[:, 1:], fulltest[:, 0]


#Feature extraction DTW
DTW_train= cdist_dtw(X_train,X_train)
DTW_test = cdist_dtw(X_train, X_test)  

DTW_train = np.transpose(DTW_train)
DTW_test = np.transpose(DTW_test)

#Feature extraction ED

EDist_test = []
for i in range(len(y_train)):
    for j in range(len(y_test)):
        dist = np.sqrt(np.sum((np.array(X_train[i,:])-np.array(X_test[j,:]))**2))
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

#Feature concatenation

train_concat = np.hstack((EDist_train,DTW_train))
test_concat = np.hstack((EDist_test,DTW_test))

'''
correct = 0

for k in range(1,len(label_test)):
    classification_obj_1 = test_dataset[k,:]
    actual_class = label_test[k]
    predicted_class = class_algo(train_dataset,label_train,classification_obj_1)
    if predicted_class == actual_class:
        correct =correct+1
print("Predicted class: {0} ".format(correct))
'''

clf = SVC(gamma='auto')
clf.fit(train_concat, y_train)

print(dataset)
print('Accuracy: ', clf.score(test_concat, y_test))


