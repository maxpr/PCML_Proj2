#!/usr/bin/env python3
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import sklearn.model_selection as ms
from sklearn import svm
import pickle
import random
import re
import string

def construct_vectors(data,set_to_fill,vocab,embeddings):
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    
    for j in range(0,np.shape(data)[0]):
        num_punctu = counter(data[j],string.punctuation)
        list_word = data[j].split()
        average = 0
        num3point = 0
        num_aux_pos =0
        num_aux_neg =0
        for i in list_word:
            average+=len(i)
            if(i=="..."):
                num3point+=1
            if(i in list_auxiliarry_pos):
                num_aux_pos+=1
            if(i in list_auxiliarry_neg):
                num_aux_neg+=1
            idx = vocab.get(i,-1)
            if(idx>=0):
                set_to_fill[j,1:np.shape(embeddings)[1]+1] += embeddings[idx]
        set_to_fill[j,1:np.shape(embeddings)[1]+1] = set_to_fill[j,1:np.shape(embeddings)[1]+1]/len(list_word)
        set_to_fill[j,np.shape(embeddings)[1]+1] = len(list_word) #add the # word
        set_to_fill[j,np.shape(embeddings)[1]+2] = num_punctu #add the # punctuation
        set_to_fill[j,np.shape(embeddings)[1]+3] = average/len(list_word) #add length of word in average
        set_to_fill[j,np.shape(embeddings)[1]+4] = num_aux_pos #word in a list of auxilarry
        set_to_fill[j,np.shape(embeddings)[1]+5] = num_aux_neg #word in a list of negative aux
        set_to_fill[j,np.shape(embeddings)[1]+6] = num3point #number of ...
    return set_to_fill
def construct_features():
    '''
    construct a feature representation of each training tweet 
    (by averaging the word vectors over all words of the tweet).
    '''
    #Load the training tweets and the built GloVe word embeddings
    additional_features = 6

    pos_train = open('data/pos_train.txt').readlines()
    neg_train = open('data/neg_train.txt').readlines()
    embeddings = np.load('data/embeddings.npy')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    pos_mask = np.zeros(np.shape(embeddings)[1]+1+additional_features)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    #will add 3 features , number of word , average length of words, and #punctuation
    training_set_pos = np.zeros(((np.shape(pos_train)[0],np.shape(embeddings)[1]+1+additional_features))) + pos_mask
    training_set_neg = np.zeros(((np.shape(neg_train)[0],np.shape(embeddings)[1]+1+additional_features)))
    #for each word, search if it is in pos_train or neg_train
    
    training_set_pos = construct_vectors(pos_train,training_set_pos,vocab,embeddings) #look at method above
    training_set_neg = construct_vectors(neg_train,training_set_neg,vocab,embeddings)
    np.save('data/trainingset_pos', training_set_pos)
    np.save('data/trainingset_neg', training_set_neg)
    
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def predict_labels(flag=".npy"):
    #Load the training set
    path_neg = str("data/trainingset_neg"+flag)
    path_pos = str("data/trainingset_pos"+flag)
    ts_neg = np.load(path_neg)
    ts_pos = np.load(path_pos)    
    #Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed 
    #features, using the scikit learn library, or your own code from the earlier labs. Recall that the labels 
    #indicate if a tweet used to contain a :) or :( smiley.
    training_set = np.concatenate((ts_neg,ts_pos))
    y = training_set[:,0]
    X = training_set[:,1:np.shape(training_set)[1]]
    X = build_poly(X,2)
    #Now we load and predict the data
    data = np.genfromtxt('data/test_data.txt', delimiter="\n",dtype=str)    
    idx = np.zeros(np.shape(data)[0])
    tweets = ["" for a in range(0,np.shape(data)[0])]
    for i in range(0,np.shape(data)[0]):
        spliter = data[i].split(",")
        idx[i] = spliter[0]
        tweet = spliter[1]
        for j in range(2,np.shape(spliter)[0]):
            tweet = tweet+","+spliter[j]
        tweets[i] = tweet
    
    #Construct the logistic regressor
    LR = sk.LogisticRegressionCV()
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
    #warm_start=False, n_jobs=1)[source]¶
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #train the logistic regressor
    kf = ms.KFold(n_splits=5,shuffle=True)
    for train_idx, test_idx in kf.split(X):
        train_set = X[train_idx]
        test_set = X[test_idx]
        train_target = y[train_idx]
        test_target = y[test_idx]    
        LR.fit(train_set,train_target)
        predictions_temp = LR.predict(test_set)
        print(predictions_temp.shape)
        print(test_target.shape)        
        error = np.sum(np.power(predictions_temp-test_target,2))/np.shape(predictions_temp)[0]
        print("Yet, error is",error)
    LR.fit(X,y)
    
    #And now, predict the results
    topredict = construct_features_for_test_set(tweets)
    topredict = build_poly(topredict,2)
    predictions = LR.predict(topredict)
    #Construct the submission
    predictions = predictions*2-1
    create_csv_submission(idx,predictions,"submission.csv")

def construct_features_for_test_set(test_set_tweet):
    embeddings = np.load('data/embeddings.npy')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    additional_features = 6
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    test_set = np.zeros((np.shape(test_set_tweet)[0],np.shape(embeddings)[1]+additional_features))
    #for each word, search if it is in a tweet
    for j in range(0,np.shape(test_set)[0]):
        list_word = test_set_tweet[j].split()
        divider = 0
        average = 0
        num3point = 0
        num_aux_pos = 0
        num_aux_neg = 0
        counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        num_punctu = counter(test_set_tweet[j],string.punctuation)
        for i in list_word:
            idx = vocab.get(i,-1)
            average+=len(i)
            if(i=="..."):
                num3point+=1
            if(i in list_auxiliarry_pos):
                num_aux_pos+=1
            if(i in list_auxiliarry_neg):
                num_aux_neg+=1
            if(idx>=0):
                divider+=1
                test_set[j,:np.shape(embeddings)[1]] += embeddings[idx]
        if(divider >0):
            test_set[j,:np.shape(embeddings)[1]] = test_set[j,:np.shape(embeddings)[1]]/divider
        test_set[j,np.shape(embeddings)[1]] = len(list_word) #add the # word
        test_set[j,np.shape(embeddings)[1]+1] = num_punctu #add the # punctuation
        if(len(list_word) >0):
            test_set[j,np.shape(embeddings)[1]+2] = average/len(list_word)#add length of word in average
        else : 
            test_set[j,np.shape(embeddings)[1]+2] = 0
        test_set[j,np.shape(embeddings)[1]+3] = num_aux_pos
        test_set[j,np.shape(embeddings)[1]+4] = num_aux_neg
        test_set[j,np.shape(embeddings)[1]+5] = num3point
    #then divide by number of words (averaging word vector over all words of the tweet)
    return test_set
construct_features()
predict_labels()
