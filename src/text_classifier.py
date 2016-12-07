#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pandas as pd
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

    
    #count number of word/tweet and store it
    word_nbr_per_tweet_pos = np.zeros(np.shape(pos_train)[0])
    for j in range(0,np.shape(pos_train)[0]):
        tweet = pos_train[j]
        size = len(re.findall(r'\w+', tweet))
        word_nbr_per_tweet_pos[j] = size
        
    word_nbr_per_tweet_neg = np.zeros(np.shape(neg_train)[0])
    for j in range(0,np.shape(neg_train)[0]):
        tweet = neg_train[j]
        size = len(re.findall(r'\w+', tweet))
        word_nbr_per_tweet_neg[j] = size
    
    i=0
    pos_mask = np.zeros(np.shape(embeddings)[1]+1)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    training_set_pos = np.zeros(((np.shape(pos_train)[0],np.shape(embeddings)[1]+1))) + pos_mask
    training_set_neg = np.zeros(((np.shape(neg_train)[0],np.shape(embeddings)[1]+1)))
    vocab = open('data/vocab_cut.txt')
    #for each word, search if it is in pos_train or neg_train
    for word in vocab:
        current_emb = embeddings[i]
        for j in range(0,np.shape(pos_train)[0]):
            #if yes, add its embeddings.
            if word in pos_train[j]:
                training_set_pos[j,1:np.shape(embeddings)[1]+1] += current_emb
        for j in range(0,np.shape(neg_train)[0]):
            if word in neg_train[j]:
                training_set_neg[j,1:np.shape(embeddings)[1]+1] += current_emb
        i+=1
    #then divide by number of words (averaging word vector over all words of the tweet)
    for i in range(0,20):
        training_set_pos[:,i+1] = training_set_pos[:,i+1]/word_nbr_per_tweet_pos
        training_set_neg[:,i+1] = training_set_neg[:,i+1]/word_nbr_per_tweet_neg
    np.save('data/trainingset_pos', training_set_pos)
    np.save('data/trainingset_neg', training_set_neg)
    