import sys

from src.boosting.adaboost import adaboost
from src.dataCleaning import dataCleaning
"""from src.word2vec_routines import construct_features, predict_labels"""



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


if len(sys.argv) != 2:
    print("you have to specify one of the following entry : ")
elif sys.argv[1] == "boosting":
    runBoosting("data/train_neg.txt",
                "data/train_pos.txt",
                "data/neg_train.txt",
                "data/pos_train.txt",
                "data/test_data.txt",)


if False:
    path_train = "data//full_train_clean_v1.txt"
    path_pos = "data/pos_train_clean_v1.txt"
    path_neg = "data/neg_train_clean_v1.txt"
    path_testing = "data/test_clean_v1.txt"
    model =  construct_features(path_pos,path_neg,path_train)
    predict_labels(model,path_testing)


