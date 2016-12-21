from word2vec_routines import *

path_train = "data//full_train_clean_v1.txt"
path_pos = "data/pos_train_clean_v1.txt"
path_neg = "data/neg_train_clean_v1.txt"
path_testing = "data/test_clean_v1.txt"
model =  construct_features(path_pos,path_neg,path_train)
predict_labels(model,path_testing)