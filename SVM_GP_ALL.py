# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:25:16 2019

@author: Ishan Yash
"""
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from pyts.metrics import dtw
from pyts.classification import KNeighborsClassifier





#print("pyts: {0}".format(pyts.__version__))

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset_list = ['CBF','ECG200','SyntheticControl','TwoPatterns','Wafer']
warping_window_list = [11,0,6,4,1]
n_paa_segments_list = [16,40,40,96,80]
n_sax_symbols_list = [4,8,8,8,8]
ALL = [['Dataset Name','Manhattan SVM','Manhattan GP','ED SVM','ED GP','DTW SVM','DTW GP','DTWR SVM','DTWR GP','1NN DTWR','Feature DTW DTWR','SVM SAX','Feature DTW DTWR SAX ']]


for i, (dataset, warping_window,n_paa_segments,n_sax_symbols) in tqdm(enumerate(zip(dataset_list, warping_window_list,n_paa_segments_list,n_sax_symbols_list))):
    app = []
    
    file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
    file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"
    app.append(dataset)
    
    print("Dataset: {}".format(dataset))
    print('Manhattan started')
    
    fulltrain = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
    fulltest = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

    X_train, y_train = fulltrain[:, 1:], fulltrain[:, 0]
    X_test, y_test = fulltest[:, 1:], fulltest[:, 0]
    
    mandist_train = manhattan_distances(X_train)
    mandist_test = manhattan_distances(X_test,X_train)
    
    clf = svm.SVC(gamma='scale')
    app.append(1-clf.fit(mandist_train, y_train).score(mandist_test,y_test))
    
    gpc = GaussianProcessClassifier()
    app.append(1-gpc.fit(X_train, y_train).score(X_test,y_test))
    
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
    app.append(1-gpc.fit(EDist_train, y_train).score(EDist_test,y_test))
    
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
    
    #DTW sakoechiba
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

    app.append(1-clf.fit(DTW_Classic_train, y_train).score(DTW_Classic_test,y_test))
    app.append(1-gpc.fit(DTW_Classic_train, y_train).score(DTW_Classic_test,y_test))
    app.append(1-clf.fit(DTW_sakoe_train, y_train).score(DTW_sakoe_test,y_test))
    app.append(1-gpc.fit(DTW_sakoe_train, y_train).score(DTW_sakoe_test,y_test))    

    print('1NN started')

    clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',metric_params={'window_size': warping_window})
    error_dtw_w = 1- clf_dtw_w.fit(X_train, y_train).score(X_test, y_test)
    app.append(error_dtw_w)
    
    print('DTW-DTWR started')
    from sklearn import svm
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from pyts.metrics import dtw
    from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                                  _return_path, _multiscale_region)


    test_DTW_DTWR = np.concatenate((DTW_sakoe_test,DTW_Classic_test),axis=1)

    train_DTW_DTWR = np.concatenate((DTW_Classic_train,DTW_sakoe_train),axis=1)


    train_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(train_DTW_DTWR)
    test_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(test_DTW_DTWR)

    train_concat_mv.resize(train_DTW_DTWR.shape[0],train_DTW_DTWR.shape[1])
    test_concat_mv.resize(test_DTW_DTWR.shape[0],test_DTW_DTWR.shape[1])

    #SVM

    clf = svm.SVC(gamma='scale')
    clf.fit(train_concat_mv, y_train)

    Error_rateDTW_DTWR =1-clf.score(test_concat_mv,y_test)
    app.append(Error_rateDTW_DTWR)
    
    print('SAX-SVM started')
    from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
    from tslearn.piecewise import PiecewiseAggregateApproximation


    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    Xtrain_sax = sax.inverse_transform(sax.fit_transform(X_train))
    Xtest_sax = sax.inverse_transform(sax.fit_transform(X_test))

    Xtest_sax =Xtest_sax[:,:,0]
    Xtrain_sax = Xtrain_sax[:,:,0]

    clf = svm.SVC(gamma='scale')
    clf.fit(Xtrain_sax, y_train)

    Error_rate_SVMSAX = 1-clf.score(Xtest_sax,y_test)
    app.append(Error_rate_SVMSAX)
    
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



    #Feature concatenation

    test = np.concatenate((EDist_test,DTW_Classic_test,DTW_sakoe_test,SAXDist_test,Xtest_sax),axis=1)
    train = np.concatenate((EDist_train,DTW_Classic_train,DTW_sakoe_train,SAXDist_train,Xtrain_sax),axis=1)

    #test = np.concatenate((DTW_Classic_test,DTW_sakoe_test,SAXDist_test),axis=1)
    #train = np.concatenate((DTW_Classic_train,DTW_sakoe_train,SAXDist_train),axis=1)

    train_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(train)
    test_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(test)

    train_concat_mv.resize(train.shape[0],train.shape[1])
    test_concat_mv.resize(test.shape[0],test.shape[1])

    #SVM

    clf = svm.SVC(gamma='scale')
    clf.fit(train_concat_mv, y_train)

    
    Error_ratefeatureall = 1-clf.score(test_concat_mv,y_test)
    app.append(Error_ratefeatureall)
    ALL.append(app)


import pandas as pd
df1 = pd.DataFrame(ALL)
#export_csv = df.to_csv('G:/Coding/ML/Feature_resultsALL/resultssvmgp_'+str(len(ALL)-1)+'_csv.csv', index=None, header=True)