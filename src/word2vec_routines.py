#from gensim.models import word2vec
import re
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
from sklearn import svm
import sklearn.model_selection as ms
from sklearn import svm
import pickle
import random
import re
import string
from nltk.stem import PorterStemmer
from gensim.models import word2vec
import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
    
global_stemmer = PorterStemmer() 
class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """ 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word
def process_tweet():            
    input_file_location = 'data/full10_tweets.txt'
    output_file_location = 'data/full10_tweets_stemmed.txt'

    with open(input_file_location, 'r') as fin:
        with open(output_file_location, 'w') as fout:
            for l in fin:
                m = l.split()  # l.strip() to remove leading/trailing whitespace
                for i in m :
                    if(StemmingHelper.stem(i) != i):
                        fout.write('%s ' % StemmingHelper.stem(i))
                    else:
                        fout.write('%s ' %i)
                fout.write('\n')
    input_file_location = 'data/neg_train.txt'
    output_file_location = 'data/neg_train_stemmed.txt'

    with open(input_file_location, 'r') as fin:
        with open(output_file_location, 'w') as fout:
            for l in fin:
                m = l.split()  # l.strip() to remove leading/trailing whitespace
                for i in m :
                    if(StemmingHelper.stem(i) != i):
                        fout.write('%s ' % StemmingHelper.stem(i))
                    else:
                        fout.write('%s ' %i)
                fout.write('\n')
    input_file_location = 'data/pos_train.txt'
    output_file_location = 'data/pos_train_stemmed.txt'

    with open(input_file_location, 'r') as fin:
        with open(output_file_location, 'w') as fout:
            for l in fin:
                m = l.split()  # l.strip() to remove leading/trailing whitespace
                for i in m :
                    if(StemmingHelper.stem(i) != i):
                        fout.write('%s ' % StemmingHelper.stem(i))
                    else:
                        fout.write('%s ' %i)
                fout.write('\n')
			
    input_file_location = 'data/test_data.txt'
    output_file_location = 'data/test_data_stemmed.txt'

    with open(input_file_location, 'r') as fin:
        with open(output_file_location, 'w') as fout:
            for l in fin:
                m = l.split()  # l.strip() to remove leading/trailing whitespace
                for i in m :
                    if(StemmingHelper.stem(i) != i):
                        fout.write('%s ' % StemmingHelper.stem(i))
                    else:
                        fout.write('%s ' %i)
                fout.write('\n')
    print("finish preprocessing")
			
def construct_features():			
    size = 200
    window = 8
    sentences = word2vec.LineSentence('data/train_clean_v1.txt')
    model = word2vec.Word2Vec(sentences, size=size,window =window)
    print("finish construct model")
	
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) #Used later to count number fo punctuation
    additional_features = 8
    
    pos_train = open('data/pos_clean_v1.txt').readlines()
    lengt = size
    pos_mask = np.zeros(lengt+1+additional_features)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    #will add 3 features , number of word , average length of words, and #punctuation
    training_set_pos = np.zeros(((np.shape(pos_train)[0],lengt+1+additional_features))) + pos_mask
    #for each word, search if it is in pos_train or neg_train
    for j in range(0,np.shape(pos_train)[0]):
        list_word = pos_train[j].split()
        num_punctu = counter(pos_train[j],string.punctuation)
        divider = 0
        average = 0
        num_user =0
        num_url= 0
        num3point = 0
        num_aux_pos =0
        num_aux_neg =0
        for i in list_word:
            average+=len(i)
            if(i=="<user>"):
                num_user+=1
            if(i=="<url>"):
                num_url+=1
            if(i=="..."): 
                num3point+=1
            if(i in list_auxiliarry_pos):
                num_aux_pos+=1
            if(i in list_auxiliarry_neg):
                num_aux_neg+=1
            if(i in model):
                divider+=1
                training_set_pos[j,1:lengt+1] += model[i]
        if(divider>0):
            training_set_pos[j,1:lengt+1] = (training_set_pos[j,1:lengt+1]/divider)
        training_set_pos[j,lengt+1] = len(list_word) #add the # word
        training_set_pos[j,lengt+2] = num_punctu #add the # punctuation
        if(len(list_word)>0):
            training_set_pos[j,lengt+3] = average/len(list_word) #add length of word in average
        else :
            training_set_pos[j,lengt+3] = 0
        training_set_pos[j,lengt+4] = num_aux_pos #word in a list of auxilarry
        training_set_pos[j,lengt+5] = num_aux_neg #word in a list of negative aux
        training_set_pos[j,lengt+6] = num3point #number of ...
        training_set_pos[j,lengt+7] = num_user #number of <user>
        training_set_pos[j,lengt+8] = num_url #number of <url>
    neg_train = open('data/neg_clean_v1.txt',encoding='utf-8').readlines()
    training_set_neg = np.zeros(((np.shape(neg_train)[0],lengt+1+additional_features)))
    #for each word, search if it is in pos_train or neg_train
    for j in range(0,np.shape(neg_train)[0]):
        num_punctu = counter(neg_train[j],string.punctuation)
        list_word = neg_train[j].split()
        divider = 0
        average = 0
        num3point = 0
        num_user= 0
        num_url = 0
        num_aux_pos =0
        num_aux_neg =0
        for i in list_word:
            average+=len(i)
            if(i=="<user>"):
                num_user+=1
            if(i=="<url>"):
                num_url+=1
            if(i=="..."): 
                num3point+=1
            if(i in list_auxiliarry_pos):
                num_aux_pos+=1
            if(i in list_auxiliarry_neg):
                num_aux_neg+=1
            if(i in model):
                divider+=1
                training_set_neg[j,1:lengt+1] += model[i]
        if(divider>0):
            training_set_neg[j,1:lengt+1] = (training_set_neg[j,1:lengt+1]/divider)
        training_set_neg[j,lengt+1] = len(list_word) #add the # word
        training_set_neg[j,lengt+2] = num_punctu #add the # punctuation
        if(len(list_word)>0):
            training_set_neg[j,lengt+3] = average/len(list_word) #add length of word in average
        else:
            training_set_neg[j,lengt+3] = 0
        training_set_neg[j,lengt+4] = num_aux_pos #word in a list of auxilarry
        training_set_neg[j,lengt+5] = num_aux_neg #word in a list of negative aux
        training_set_neg[j,lengt+6] = num3point #number of ...
        training_set_neg[j,lengt+7] = num_user #number of <user>
        training_set_neg[j,lengt+8] = num_url #number of <url
    np.save('data/trainingsetword2vec_pos', training_set_pos)
    np.save('data/trainingsetword2vec_neg', training_set_neg)
    return model
	
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

			
			
def predict_labels(model,flag=".npy"):
    #Load the training set
    path_neg = str("data/trainingsetword2vec_neg"+flag)
    path_pos = str("data/trainingsetword2vec_pos"+flag)
    ts_neg = np.load(path_neg)
    ts_pos = np.load(path_pos)
    #Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed 
    #features, using the scikit learn library, or your own code from the earlier labs. Recall that the labels 
    #indicate if a tweet used to contain a :) or :( smiley.
    training_set = np.concatenate((ts_neg,ts_pos))
    y = training_set[:,0]
    X = training_set[:,1:np.shape(training_set)[1]]
    #Now we load and predict the data
    data = open('data/test_clean_v1.txt',encoding='utf-8').readlines()
    idx = np.zeros(np.shape(data)[0])
    tweets = ["" for a in range(0,np.shape(data)[0])]
    for i in range(0,np.shape(data)[0]):
        idx[i] =(i+1)
        tweets[i] = data[i]
    
    #Construct the logistic regressor
    LR = sk.LogisticRegressionCV()
    #clf = svm.SVC()
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
    #warm_start=False, n_jobs=1)[source]¶
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #train the logistic regressor
    X_poly = X
    LR.fit(X_poly,y)
    print("fitting done")
    #clf.fit(X_poly, y)
    #And now, predict the results
    topredict = construct_features_for_test_set(model,tweets)
    topredict_poly = topredict
    print("test set constructed")
    predictions = LR.predict(topredict_poly)
    #Construct the submission
    predictions = predictions*2-1
    create_csv_submission(idx,predictions,"submission.csv")
    
def construct_features_for_test_set(model,test_set_tweet):
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) #Used later to count number fo punctuation
    
    additional_features = 8
    lengt = 200
    test_set = np.zeros((np.shape(test_set_tweet)[0],lengt+additional_features))
    for j in range(0,np.shape(test_set)[0]):
        num_punctu = counter(test_set_tweet[j],string.punctuation)
        list_word = test_set_tweet[j].split()
        divider = 0
        average = 0
        num3point = 0
        num_aux_pos =0
        num_aux_neg =0
        num_user = 0
        num_url= 0
        for i in list_word:
            average+=len(i)
            if(i=="<user>"):
                num_user+=1
            if(i=="<url>"):
                num_url+=1
            if(i=="..."): 
                num3point+=1
            if(i in list_auxiliarry_pos):
                num_aux_pos+=1
            if(i in list_auxiliarry_neg):
                num_aux_neg+=1
            if(i in model):
                divider+=1
                test_set[j,:lengt] += model[i]
        if(divider>0):
            test_set[j,:lengt] = (test_set[j,:lengt]/divider)
        test_set[j,lengt] = len(list_word) #add the # word
        test_set[j,lengt+1] = num_punctu #add the # punctuation
        if(len(list_word) >0):
            test_set[j,lengt+2] = average/len(list_word)#add length of word in average
        else : 
            test_set[j,lengt+2] = 0
        test_set[j,lengt+3] = num_aux_pos #word in a list of auxilarry
        test_set[j,lengt+4] = num_aux_neg #word in a list of negative aux
        test_set[j,lengt+5] = num3point #number of ...
        test_set[j,lengt+6] = num_user #number of <user>
        test_set[j,lengt+7] = num_url #number of <url>
    return test_set

model = construct_features()
predict_labels(model)
