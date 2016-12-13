#!/usr/bin/env python3
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import pickle
import random
import re

def construct_features():
    '''
    construct a feature representation of each training tweet 
    (by averaging the word vectors over all words of the tweet).
    '''
    #Load the training tweets and the built GloVe word embeddings.
    pos_train = open('data/pos_train.txt').readlines()
    neg_train = open('data/neg_train.txt').readlines()
    embeddings = np.load('data/embeddings.npy')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    pos_mask = np.zeros(np.shape(embeddings)[1]+1)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    training_set_pos = np.zeros(((np.shape(pos_train)[0],np.shape(embeddings)[1]+1))) + pos_mask
    training_set_neg = np.zeros(((np.shape(neg_train)[0],np.shape(embeddings)[1]+1)))
    #for each word, search if it is in pos_train or neg_train
    for j in range(0,np.shape(pos_train)[0]):
        list_word = pos_train[j].split()
        for i in list_word:
            idx = vocab.get(i,-1)
            if(idx>=0):
                training_set_pos[j,1:np.shape(embeddings)[1]+1] += embeddings[idx]
    for j in range(0,np.shape(neg_train)[0]):
        list_word = neg_train[j].split()
        for i in list_word:
            idx = vocab.get(i,-1)
            if(idx>=0):
                training_set_neg[j,1:np.shape(embeddings)[1]+1] += embeddings[idx]
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
    #Construct the logistic regressor
    LR = sk.LogisticRegression()
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
    #warm_start=False, n_jobs=1)[source]¶
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #train the logistic regressor
    LR.fit(X,y)

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
    #And now, predict the results
    topredict = construct_features_for_test_set(tweets)
    predictions = LR.predict(topredict)
    #Construct the submission
    predictions = predictions*2-1
    create_csv_submission(idx,predictions,"submission.csv")

def construct_features_for_test_set(test_set_tweet):
    embeddings = np.load('data/embeddings.npy')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    test_set = np.zeros((np.shape(test_set_tweet)[0],np.shape(embeddings)[1]))
    #for each word, search if it is in a tweet
    for j in range(0,np.shape(test_set)[0]):
        list_word = test_set_tweet[j].split()
        for i in list_word:
            idx = vocab.get(i,-1)
            if(idx>=0):
                test_set[j,:] += embeddings[idx]
    #then divide by number of words (averaging word vector over all words of the tweet)
    return test_set
predict_labels()


