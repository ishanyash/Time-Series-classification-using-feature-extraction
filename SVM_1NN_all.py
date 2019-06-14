# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:22:33 2019

@author: Ishan Yash
"""

import numpy as np
from pyts.classification import KNeighborsClassifier
from sklearn import svm
from tqdm import tqdm

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset_list = ['ECG200','CBF','SyntheticControl'] 
warping_window_list = [0,11,6]
n_paa_segments_list = [40,16,40]
n_sax_symbols_list =[8,4,8]

clf = svm.SVC(gamma='scale')

ALL = [['Dataset','1NN DTWR','ED SVM','DTWSVM','DTWR_SVM','SVM_SAX','DTW_DTWR_SAX','DTW_DTWR_SVM']]


for i,(dataset,warping_window,n_paa_segments,n_sax_symbols) in tqdm(enumerate(zip(dataset_list,warping_window_list,n_paa_segments_list,n_sax_symbols_list))):
    app = []
    file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
    file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"
    
    fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
    fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)
    
    X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
    X_test, y_test = fulltest[:, 1:], fulltest[:, 0]
    
    app.append(dataset)
    print('1NN-DTWR')
    ww = warping_window
    clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',metric_params={'window_size': ww})

    error_dtw_w = 1- clf_dtw_w.fit(X_train, y_train).score(X_test, y_test)
    app.append(error_dtw_w)
    
    print('ED started')
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
    app.append(1-clf.fit(EDist_train, y_train).score(EDist_test,y_test))

    from pyts.metrics import dtw

    print('DTW started')
    DTW_Classic_test = []
    DTW_Classic_train = []
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            dtw_classic, path_classic = dtw(X_test[i], X_train[j], dist='square',
                                    method='classic', return_path=True)
            DTW_Classic_test.append(dtw_classic)

    for i in range(len(X_train)):
        for j in range(len(X_train)):
            dtw_classic, path_classic = dtw(X_train[i], X_train[j], dist='square',method='classic', return_path=True)
            DTW_Classic_train.append(dtw_classic)
    DTW_Classic_train = np.array(DTW_Classic_train)
    DTW_Classic_train.resize(y_train.shape[0],int(len(DTW_Classic_train)/y_train.shape[0]))
    DTW_Classic_test = np.array(DTW_Classic_test)
    DTW_Classic_test.resize(y_test.shape[0],int(len(DTW_Classic_test)/y_test.shape[0]))
    app.append(1-clf.fit(DTW_Classic_train, y_train).score(DTW_Classic_test,y_test))

    #DTWR
    DTW_sakoe_test = []
    DTW_sakoe_train = []
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            dtw_sakoechiba, path_sakoechiba = dtw(X_test[i], X_train[j], dist='square', method='sakoechiba',options={'window_size': warping_window}, return_path=True)
            DTW_sakoe_test.append(dtw_sakoechiba)

    for i in range(len(X_train)):
        for j in range(len(X_train)):
            dtw_sakoechiba, path_sakoechiba = dtw(X_train[i], X_train[j], dist='square', method='sakoechiba',options={'window_size': warping_window}, return_path=True)
            DTW_sakoe_train.append(dtw_sakoechiba)

    DTW_sakoe_train = np.array(DTW_sakoe_train)
    DTW_sakoe_train.resize(y_train.shape[0],int(len(DTW_sakoe_train)/y_train.shape[0]))
    DTW_sakoe_test = np.array(DTW_sakoe_test)
    DTW_sakoe_test.resize(y_test.shape[0],int(len(DTW_sakoe_test)/y_test.shape[0]))
    app.append(1-clf.fit(DTW_sakoe_train, y_train).score(DTW_sakoe_test,y_test))

    print('SAX-SVM started')
    from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
    from tslearn.piecewise import PiecewiseAggregateApproximation


    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    Xtrain_sax = sax.inverse_transform(sax.fit_transform(X_train))
    Xtest_sax = sax.inverse_transform(sax.fit_transform(X_test))

    Xtest_sax =Xtest_sax[:,:,0]
    Xtrain_sax = Xtrain_sax[:,:,0]

    print('Feature DTW-DTWr-SAX started')
    #SAX Transform + SAX feature extraction


    SAXDist_test = []
    for i in range(len(y_test)):
        for j in range(len(y_train)):
            #dtw_sakoechiba, path_sakoechiba = dtw(SAX_test[i], SAX_train[j], dist='square', method='sakoechiba',options={'window_size': window_size}, return_path=True)
            dtw_classic, path_classic = dtw(Xtest_sax[i], Xtrain_sax[j], dist='square',method='classic', return_path=True)
            #dist = np.sqrt(np.sum((np.array(SAX_test[i,:])-np.array(SAX_train[j,:]))**2))
            #SAXDist_test.append(dtw_sakoechiba)
            SAXDist_test.append(dtw_classic)
            #SAXDist_test.append(dist)

    SAXDist_train = []
    for i in range(len(y_train)):
        for j in range(len(y_train)):
            #dtw_sakoechiba, path_sakoechiba = dtw(SAX_train[i], SAX_train[j], dist='square', method='sakoechiba',options={'window_size': window_size}, return_path=True)
            dtw_classic, path_classic = dtw(Xtrain_sax[i], Xtrain_sax[j], dist='square',method='classic', return_path=True)
            #dist1 = np.sqrt(np.sum((np.array(SAX_train[i,:])-np.array(SAX_train[j,:]))**2))
            #SAXDist_train.append(dtw_sakoechiba)
            SAXDist_train.append(dtw_classic)
            #SAXDist_train.append(dist1)

    SAXDist_train = np.array(SAXDist_train)
    SAXDist_train.resize(y_train.shape[0],int(len(SAXDist_train)/y_train.shape[0]))
    SAXDist_test = np.array(SAXDist_test)
    SAXDist_test.resize(y_test.shape[0],int(len(SAXDist_test)/y_test.shape[0]))

    app.append(1-clf.fit(Xtrain_sax, y_train).score(Xtest_sax,y_test))

    test_DTW_DTWR_SAX = np.hstack((DTW_sakoe_test,DTW_Classic_test,SAXDist_test))
    train_DTW_DTWR_SAX = np.hstack((DTW_sakoe_train,DTW_Classic_train,SAXDist_train))

    test_DTW_DTWR = np.hstack((DTW_sakoe_test,DTW_Classic_test))
    train_DTW_DTWR = np.hstack((DTW_sakoe_train,DTW_Classic_train))

    app.append(1-clf.fit(train_DTW_DTWR_SAX, y_train).score(test_DTW_DTWR_SAX,y_test))
    app.append(1-clf.fit(train_DTW_DTWR, y_train).score(test_DTW_DTWR,y_test)) 
    ALL.append(app)
import pandas as pd
df = pd.DataFrame(ALL)
