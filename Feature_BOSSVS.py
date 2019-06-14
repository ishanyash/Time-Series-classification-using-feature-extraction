# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:59:16 2019

@author: Ishan Yash
"""

import numpy as np
from sklearn import svm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
from pyts.metrics import dtw
from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                              _return_path, _multiscale_region)
from pyts.classification import BOSSVS

print('Feature-BOSSVS')
'''
nps = input("Enter the number of segments for PAA: ")
n_paa_segments = int(nps) #number of segments for PAA
nss = input('Enter the number if segments for SAX: ')
n_sax_symbols = int(nss)
w = input('Enter the window size:')
window_size = int(w)
'''

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
ds = input('Enter the Time series Dataset: ')
dataset = str(ds) 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
X_test, y_test = fulltest[:, 1:], fulltest[:, 0]

# BOSSVS transformation
bossvs = BOSSVS(word_size=2, n_bins=3, window_size=10)
bossvs.fit(X_train, y_train)
tfidf = bossvs.tfidf_
vocabulary_length = len(bossvs.vocabulary_)
X_new = bossvs.decision_function(X_train) #tfidf vectors
