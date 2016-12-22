import sys

from boosting.adaboost import adaboost
from dataCleaning import dataCleaning
from word2vec_routines import construct_features, predict_labels



def cleanTrainingSet(pathToOrigFile, pathToOutFile):
    dataC = dataCleaning(pathToOrigFile)
    dataC.setDuplicateLinesRemoved()
    dataC.setTweetTransform()
    dataC.save(pathToOutFile)

def cleanTest(pathToOrigFile, pathToOutFile):
    dataC = dataCleaning(pathToOrigFile)
    dataC.setRemoveTestTweetId()
    dataC.setTweetTransform()
    dataC.save(pathToOutFile)


def runBoosting(negTrain, posTrain, negTest, posTest, kaggleTest):

    # 1. Open the training data :

    pos = dataCleaning(posTrain).getData()
    neg = dataCleaning(negTrain).getData()

    kaggleTest = dataCleaning(kaggleTest).getData()

    posTest = dataCleaning(posTest).getData()
    negTest = dataCleaning(negTest).getData()

    # 2. Instantiate adaboost class :

    boosting = adaboost(pos, neg)

    # 3. Start the algorithm :

    boosting.run('result.txt', posTest, negTest, kaggleTest)

def word2vec():
    path_pos = "data/train_pos_full.txt"
    path_neg = "data/train_neg_full.txt"
    path_testing = "data/test_data.txt"
    
    path_testing_clean= "data_test_clean.txt"
    path_pos_clean = "train_pos_full_clean.txt"
    path_neg_clean = "train_neg_full_clean.txt"
    path_train_clean = "train_clean.txt"
    
    cleanTest(path_testing,path_testing_clean)
    cleanTrainingSet(path_neg,path_neg_clean)
    cleanTrainingSet(path_pos,path_pos_clean)

    filenames = [path_pos_clean,path_neg_clean]
    with open(path_train_clean, 'w',encoding="utf-8") as outfile:
        for fname in filenames:
            with open(fname,encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    model =  construct_features(path_pos_clean,path_neg_clean,path_train_clean)
    predict_labels(model,path_testing)


if len(sys.argv) != 2:
    print("you have to specify one of the following entry : ")
elif sys.argv[1] == "boosting":
    runBoosting("data/neg_train.txt",
                "data/pos_train.txt",
                "data/neg_train.txt",
                "data/pos_train.txt",
                "data/test_data.txt",)
elif sys.argv[1] == "word2vec":
    word2vec()






