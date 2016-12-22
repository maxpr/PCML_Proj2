
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
from sklearn import svm
import sklearn.model_selection as ms
import string
from gensim.models import word2vec

def construct_vector(data,set_to_fill,model,lengt):
    """
    Creates an array that contains tweet features representation
    Arguments: data The set of tweet in string
               set_to_fill The array you need to fill with the corresponding features representation of the tweet at index i
               vocab a vocabulary that return the index of the word given in argument when doing vocab.get(word,-1) , and - 1 if not found
               embedding a Matrix that represent the feature representation of words (in accord of indexes with vocab)
    """  
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) #Used later to count number fo punctuation
    
    for j in range(0,np.shape(data)[0]): # For each tweet
        list_word = data[j].split() # Split into an array of words
        num_punctu = counter(data[j],string.punctuation) # count the punctuation
        divider = 0 #Initialize some parameters for additional features
        average = 0
        num_user =0
        num_url= 0
        num3point = 0
        num_aux_pos =0
        num_aux_neg =0
        for i in list_word: # For each word, fill the variable used for additional features
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
                set_to_fill[j,1:lengt+1] += model[i]
        if(divider>0):
            set_to_fill[j,1:lengt+1] = (set_to_fill[j,1:lengt+1]/divider)
        set_to_fill[j,lengt+1] = len(list_word) #add the # word
        set_to_fill[j,lengt+2] = num_punctu #add the # punctuation
        if(len(list_word)>0):
            set_to_fill[j,lengt+3] = average/len(list_word) #add length of word in average
        else :
            set_to_fill[j,lengt+3] = 0
        set_to_fill[j,lengt+4] = num_aux_pos #word in a list of auxilarry
        set_to_fill[j,lengt+5] = num_aux_neg #word in a list of negative aux
        set_to_fill[j,lengt+6] = num3point #number of ...
        set_to_fill[j,lengt+7] = num_user #number of <user>
        set_to_fill[j,lengt+8] = num_url #number of <url>
    return set_to_fill

def construct_features(path_pos,path_neg,train_path):
    '''
    construct a feature representation of each training tweet 
    (by averaging the word vectors over all words of the tweet).
    Using the model created by word2vec
    '''
    size = 200
    window = 8
    sentences = word2vec.LineSentence(train_path) #Load the tweet (the whole training set)
    model = word2vec.Word2Vec(sentences,min_count = 2, size=size,window =window) # create the embeddding for each words, that appear more than 2 time
                                                                                # and with an embedding size of 200 and dependency windows of 8 characters
    print("finish construct model")
	
    additional_features = 8 #Number of added features by hand (easier to scale the vectors)
    
    #Create tweet embedding for positive
    pos_train = open(path_pos,encoding='utf-8').readlines()
    lengt = size
    pos_mask = np.zeros(lengt+1+additional_features)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    #will add 3 features , number of word , average length of words, and #punctuation
    training_set_pos = np.zeros(((np.shape(pos_train)[0],lengt+1+additional_features))) + pos_mask
    #for each word, search if it is in pos_train or neg_train
    training_set_pos = construct_vector(pos_train,training_set_pos,model,lengt)
    
    #Create tweet embeddings for negative
    neg_train = open(path_neg,encoding='utf-8').readlines()
    training_set_neg = np.zeros(((np.shape(neg_train)[0],lengt+1+additional_features)))
    #for each word, search if it is in pos_train or neg_train
    training_set_neg = construct_vector(neg_train,training_set_neg,model,lengt)
    
    #Save the embeddings
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

			
			
def predict_labels(model,path_testing,flag=".npy"):
    """
    Used to predict the label on a given training Set
    With a constructed model from word2vec
    """
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
    data = open(path_testing,encoding='utf-8').readlines()
    idx = np.zeros(np.shape(data)[0])
    tweets = ["" for a in range(0,np.shape(data)[0])]
    for i in range(0,np.shape(data)[0]):
        idx[i] =(i+1)
        tweets[i] = data[i]
    
    #Construct the logistic regressor
    LR = sk.LogisticRegressionCV()
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
    #warm_start=False, n_jobs=1)[source]¶
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #train the logistic regressor
    
    #Do a K fold on the training set to have an idea of the error.
    kf = ms.KFold(n_splits=3,shuffle=True)
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
    #Fit the prediction model
    LR.fit(X,y)

    #And now, predict the results
    topredict = construct_features_for_test_set(model,tweets)
    topredict_poly = topredict
    print("test set constructed")
    predictions = LR.predict(topredict_poly)
    #Construct the submission
    predictions = predictions*2-1
    create_csv_submission(idx,predictions,"submission.csv")
    
def construct_features_for_test_set(model,test_set_tweet):
    """
    Creates Features representation for the test set, we do not use the same method
    as the structure is a little different ( no labels)
               test_set_tweet: the text representation of the given tweets
               model ; the model used for word feature representation
    return : the representation in features of the set of tweet
    """
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


