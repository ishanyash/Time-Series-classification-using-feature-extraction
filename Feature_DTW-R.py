# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:27:47 2019

@author: Ishan Yash
"""

import numpy as np
from pyts.classification import KNeighborsClassifier

print('Feature-DTW-R')
ww = input("Enter the Warping window: ")
warping_window = float(ww) #warping window

clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',metric_params={'window_size': warping_window})

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
ds = input('Enter the Time series Dataset: ')
dataset = str(ds) 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
X_test, y_test = fulltest[:, 1:], fulltest[:, 0]



#print('Accuracy DTW_W: ', 1 - error_dtw_w )
#print("Error rate with Dynamic Time Warping with a learned warping ""window: {0:.4f}".format(error_dtw_w))
#error_dtw_w_list.append(error_dtw_w)