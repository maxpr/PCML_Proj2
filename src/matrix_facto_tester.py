#!/usr/bin/env python3
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import pickle
import random
import re
from glove_routines import *

import os
import re

def extrac_param(name):
    params = []
    splited = str(name).split("_")
    for i in range(2,np.shape(splited)[0]):
        p = splited[i]
        value = re.findall(r"[+-]?\d+(?:\.\d+)?",p)
        params.append(float(value[0]))
    return params

def choose_parameters(threshold):
    '''
    Test different parameters to see which one gives the better results.
    '''
    embedding_dim = [5, 30, 50, 100, 200]
    for subdir, dirs, files in os.walk('metadata'):
        size = np.shape(files)[0]
        losses = np.zeros(size)
        names = []
        param = []
        i=0
        for file in files:
            f = np.load('metadata/'+str(file))
            losses[i] = f
            names.append(file)
            param.append(extrac_param(file))
            i+=1
    print(np.shape(names)[0], "results found.")
    smallest_loss=np.ones(201)*pow(10,20)
    smallest_idx = np.zeros(201)
    small_lost = []
    small_idx = []
    small_lost_param = []
    for i in range(0,np.shape(losses)[0]):
        p = param[i]
        emb = p[0]
        loss = losses[i]
        if(loss<smallest_loss[emb]):
            smallest_loss[emb]=loss
            smallest_idx[emb] = int(i)
            
    for i in range(0,np.shape(losses)[0]):
        p = param[i]
        emb = p[0]
        loss = losses[i]
        if(loss<threshold*smallest_loss[emb]):
            small_lost.append(loss)
            small_lost_param.append(p)
            small_idx.append(i)

    print(np.shape(small_lost)[0], "small results found (",threshold,"threshold )")

    for i in embedding_dim:
        cu_params = param[int(smallest_idx[i])]
        print("Smallest lost for embeddings",cu_params[0],"is :",smallest_loss[i],"with params:",cu_params)
        print("   and name:",names[int(smallest_idx[i])])
    i=0
    for a in small_idx:
        p = param[a]
        print("   ",param[a], " loss is ", small_lost[i]/smallest_loss[p[0]])
        i+=1

def fix_param(data,idx,value):
    assert ((data[:,idx]==value).any())
    newdata = []
    for i in range(0,np.shape(data)[0]):
        if(data[i,idx]==value):

            newdata.append(data[i])
    #print(newdata)
    return recreate_param(newdata)

def recreate_param(data):
    x = np.shape(data)[0]
    y = np.shape(data)[1]
    param = np.zeros((x,y))
    for i in range(0,x):
        for j in range(0,y):
            param[i,j] = data[i][j]
    return param
from mpl_toolkits.mplot3d import Axes3D
def show_graph():
    embedding_dim = [5, 30, 50, 100, 200]
    for subdir, dirs, files in os.walk('metadata'):
        size = np.shape(files)[0]
        losses = np.zeros(size)
        names = []
        param = []
        i=0
        for file in files:
            f = np.load('metadata/'+str(file))
            losses[i] = f
            names.append(file)
            p = extrac_param(file)
            p.append(f)
            param.append(p)
            i+=1
    print(np.shape(names)[0], "results found.")
    param = recreate_param(param)
    fixed_emb = fix_param(param,0,5)
    fixed_eta = fix_param(fixed_emb,1,0.01)
    xs = fixed_eta[:,2]
    ys = fixed_eta[:,3]
    zs = fixed_eta[:,5]
    plt.figure()
    plt.subplot(121)
    plt.axis([min(xs)*0.9,max(xs)*1.1,min(zs)*0.9,max(zs)*1.1])
    plt.plot(xs,zs,'ro')
    plt.subplot(122)
    plt.xscale('log')
    plt.axis([min(ys)*0.9,max(ys)*1.1,min(zs)*0.9,max(zs)*1.1])

    plt.plot(ys,zs,'yo')
    
def test_embeddings(flag=""):
    #Load the training set
    path_neg = str("data/trainingset_neg"+flag+".npy")
    path_pos = str("data/trainingset_pos"+flag+".npy")
    ts_neg = np.load(path_neg)
    ts_pos = np.load(path_pos)    
    #Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed 
    #features, using the scikit learn library, or your own code from the earlier labs. Recall that the labels 
    #indicate if a tweet used to contain a :) or :( smiley.
    training_set = np.concatenate((ts_neg,ts_pos))
    y = training_set[:,0]
    X = training_set[:,1:np.shape(training_set)[1]]
    X = build_poly(X,2)
    
    #Construct the logistic regressor
    LR = sk.LogisticRegressionCV()
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
    #warm_start=False, n_jobs=1)[source]¶
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #train the logistic regressor
    nsplits = 5
    average_error=0
    kf = ms.KFold(n_splits=nsplits,shuffle=True)
    for train_idx, test_idx in kf.split(X):
        train_set = X[train_idx]
        test_set = X[test_idx]
        train_target = y[train_idx]
        test_target = y[test_idx]    
        LR.fit(train_set,train_target)
        predictions_temp = LR.predict(test_set)
        error = np.sum(np.power(predictions_temp-test_target,2))/np.shape(predictions_temp)[0]
        average_error+=error
    average_error = average_error/nsplits
    return average_error


def test_all_embeddings():
    for subdir, dirs, files in os.walk('embeddings'):
        i=0
        for file in files:
            i+=1
            flag = str("_test_"+str(i))
            print("Constructing features for",file)
            construct_features('embeddings/'+str(file),flag)
            print("   Calculation error")
            error = test_embeddings(flag)
            print("   For file : ",file)
            print("   Error is:",error)