from gensim.models import word2vec
import csv
import numpy as np
import sklearn.linear_model as sk
import sklearn.model_selection as ms
import string

			
def construct_features():
    """ Create the model that maps word to vectors, of size 200 , and use a windows of 8 character to see
    dependences between word
    return : model , if you give him a word it'll return the numeric vector that represent it
            size , to know what dimension you're giving you vectors.
    """
    size = 200
    window = 8
    #Read the tweets in a format that Word2Vec can proceed
    sentences = word2vec.LineSentence('data/train_clean_v1_1.txt')
    #Create the embedding using two-layer neural network.
    #It uses the skip gram model to create embeddings of word, so it uses dependences of word in a window of 8 char long
    model = word2vec.Word2Vec(sentences, size=size,window =window)
    print("finish construct model")
    return model,size
    
def construct_vectors(data,set_to_fill,model,length):

    """
    Creates an array that contains tweet features representation
    Arguments: data The set of tweet in string
               set_to_fill The array you need to fill with the corresponding features representation of the tweet at index i
               model the obect that contains the feature representation of each word
               length , the size of the vector features representation
    """
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    #used for additional features
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"] #used for additional features
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) #Used later to count number fo punctuation
    
    for j in range(0,np.shape(data)[0]): # For each tweet
        list_word = data[j].split() # Split in word
        num_punctu = counter(data[j],string.punctuation) #Count the punctuation
        divider = 0 # initialize the different additional features
        average = 0
        num_user =0
        num_url= 0
        num3point = 0
        num_aux_pos =0
        num_aux_neg =0
        for i in list_word: # For each word fill the differents features
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
                set_to_fill[j,1:length+1] += model[i]
        if(divider >0): # Put all additional features in the array
            set_to_fill[j,1:length+1] = set_to_fill[j,1:length+1]/divider #put the average of the word vector in dim [1,21]
        set_to_fill[j,length+1] = len(list_word) #add the # word
        set_to_fill[j,length+2] = num_punctu #add the # punctuation
        if(len(list_word) > 0):
            set_to_fill[j,length+3] = average/len(list_word) #add length of word in average
        else :
            set_to_fill[j,length+3] = 0
        set_to_fill[j,length+4] = num_aux_pos #word in a list of auxilarry
        set_to_fill[j,length+5] = num_aux_neg #word in a list of negative aux
        set_to_fill[j,length+6] = num3point #number of "..."
        set_to_fill[j,length+7] = num_user #number of "<user>"
    return set_to_fill

def create_tweet_embedding():
    """
        Create the features representation of each tweet and save them in a file
        return , the model used
    """
    additional_features = 7 #number of features we add ourselves
    
    model,size = construct_features() #construct the model
    
    #Load both training set and initialize both tweet features representation.
    pos_train = open('data/pos_clean_v1_1.txt').readlines()
    length = size
    pos_mask = np.zeros(length+1+additional_features)
    pos_mask[0] +=1
    #adding 1 at start : this is target (1 is for happy emoji, 0 or -1 for sad face)
    training_set_pos = np.zeros(((np.shape(pos_train)[0],length+1+additional_features))) + pos_mask
    neg_train = open('data/neg_clean_v1_1.txt',encoding='utf-8').readlines()
    training_set_neg = np.zeros(((np.shape(neg_train)[0],length+1+additional_features)))
    
    #Fill both tweet representation thank to the previous method
    training_set_pos = construct_vectors(pos_train,training_set_pos,model,length)
    training_set_neg = construct_vectors(pos_train,training_set_neg,model,length)
    
    #Save the vector representation of tweets to be trained later
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
    """
    Used to predict the label on a given training Set
    """
    #Load the training sets
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
    data = open('data/test_clean_v1_1.txt',encoding='utf-8').readlines()
    #Used to put label and index together
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
    
    #Do a Kfold to have an idea of the error
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
    #Fit our training model
    LR.fit(X,y)
    print("fitting done")
    #clf.fit(X_poly, y)
    #And now, predict the results
    topredict = construct_features_for_test_set(model,tweets)
    print("test set constructed")
    predictions = LR.predict(topredict)
    #Construct the submission
    predictions = predictions*2-1
    create_csv_submission(idx,predictions,"submission.csv")
    
def construct_features_for_test_set(model,test_set_tweet):
    """
    Creates Features representation for the test set, we do not use the same method
    as the structure is a little different ( no labels)
               test_set_tweet: the text representation of the given tweets
    return : the representation in features of the set of tweet
    """
    
    list_auxiliarry_pos = ["must","need","should","may","might","can","could","shall","would","will"]
    list_auxiliarry_neg = ["won't","shouldn't","not","can't","couldn't","wouldn't"]
    counter = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) #Used later to count number fo punctuation
    
    additional_features = 7
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
        for i in list_word:
            average+=len(i)
            if(i=="<user>"):
                num_user+=1
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
    return test_set


model = create_tweet_embedding()
predict_labels(model)
