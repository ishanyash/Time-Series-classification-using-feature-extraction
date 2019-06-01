# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:14:41 2019

@author: Ishan Yash
"""

# this is the dtw distance implemented directly
import numpy as np

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "FacesUCR" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]

def dtw_dist(p,q):
    ep = np.sqrt(np.sum(np.square(p),axis=1));
    eq = np.sqrt(np.sum(np.square(q),axis=1));
    D = 1 - np.dot(p,q.T)/np.outer(ep,eq) # work out D all at once
    S = np.zeros_like(D)
    Lp = np.shape(p)[0]
    Lq = np.shape(q)[0]
    N = np.shape(p)[1]
    for i in range(Lp):
        for j in range(Lq):          
            if i==0 and j==0:  S[i,j] = D[i,j]
            elif i==0: S[i,j] = S[i,j-1] + D[i,j]
            elif j==0: S[i,j] = S[i-1,j] + D[i,j]
            else: S[i,j] = np.min([S[i-1,j],S[i,j-1],S[i-1,j-1]]) + D[i,j]
    return S[-1,-1] # return the bottom right hand corner distance

# features_test, features_train are lists of numpy arrays containing MFCCs. 
# label_test and label_train are just python lists of class labels (0='A', 1='B' etc.)    
correct = 0
total = 0
for i in range(len(X_test)):
    best_dist = 10**10
    best_label = -1
    for j in range(len(X_train)):
        dist_ij = dtw_dist(X_test[i],X_train[j])
        if dist_ij < best_dist:
            best_dist = dist_ij
            best_label = y_train[j]
    if best_label == y_test[i]: correct += 1
    total += 1
print('accuracy:',correct/total)