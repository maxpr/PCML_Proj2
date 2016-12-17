from src.dataCleaning import dataCleaning
from src.boosting.adaboost import adaboost
from functools import reduce
from src.boosting.vocabulary import vocabulary



if False:


    one = {}

    one[1] = 1
    one[2] = 2

    two = one.copy()

    two[2] = 4

    print(one)
    print(two)

if False:

    dataC = dataCleaning('../data/train_pos_full.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.cutAndSave(0.9,'../data/pos_train_full.txt','../data/pos_test_full.txt',)
    print('done')


    dataC = dataCleaning('../data/train_neg_full.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.cutAndSave(0.9,'../data/neg_train_full.txt','../data/neg_test_full.txt',)
    print('done')



if False:

    dataC = dataCleaning('../data/test_data.txt')
    print('remove duplicate...')
    dataC.setRemoveTestTweetId()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.save('../data/test_data_clean.txt')
    print('done')


if False:
    dataC = dataCleaning('../data/pos_train_full.txt')
    dataC.cutAndSave(0.5, '../data/pos_train_full1.txt','../data/pos_train_full2.txt')

    dataC = dataCleaning('../data/neg_train_full.txt')
    dataC.cutAndSave(0.5, '../data/neg_train_full1.txt','../data/neg_train_full2.txt')

    print('done')


if True:

    boosting = adaboost('../data/pos_train_full1.txt','../data/neg_train_full1.txt')

    posTest = dataCleaning('../data/pos_test_full.txt').getData()
    negTest = dataCleaning('../data/neg_test_full.txt').getData()
    tweetsToPred = dataCleaning('../data/test_data_clean.txt').getData()

    boosting.learnAndTest('../data/pred_output_index.txt',posTest, negTest, tweetsToPred)





if False:

    dataC = dataCleaning('../data/neg_train_clean.txt')
    dataC.cutAndSave(0.8,'../data/neg_train_clean_cutTrain.txt','../data/neg_train_clean_cutTest.txt')
    dataC = dataCleaning('../data/pos_train_clean.txt')
    dataC.cutAndSave(0.8, '../data/pos_train_clean_cutTrain.txt', '../data/pos_train_clean_cutTest.txt')


if False:

    one = set([1,2,3])
    two = set([4,5,6])

    three = [one,two]

    four = reduce( ( lambda x,y: x.union(y) ),three)

    print(one)
    print(two)

    print(four)


if False:

    negTest = dataCleaning('../data/neg_train.txt')
    posTest = dataCleaning('../data/pos_train.txt')

    wordFuncTestPos = posTest.wordFuncTest()
    wordFuncTestNeg = negTest.wordFuncTest()

    for wF in wordFuncTestPos.keys():

        print("CURRENTLY ON : " + wF)

        occToTotalPos = wordFuncTestPos[wF]
        occToTotalNeg = {}
        if wF in wordFuncTestNeg:
            occToTotalNeg = wordFuncTestNeg[wF]


        for (occ, totalPos) in occToTotalPos.items():

            totalNeg = 0
            if occ in occToTotalNeg:
                totalNeg = occToTotalNeg[occ]

            print("\t occ = " + str(occ) + " posTotal = " + str(totalPos) + "\t\t negTotal = " + str(totalNeg))



if False:

    negTest = dataCleaning('../data/neg_train.txt')
    posTest = dataCleaning('../data/pos_train.txt')


    negSmallW = negTest.avgSmallWords()
    posSmallW = posTest.avgSmallWords()

    for (totalSmallWords, occNeg) in negSmallW.items():

        occPos = 0
        if totalSmallWords in posSmallW:
            occPos = posSmallW[totalSmallWords]

        print(str(totalSmallWords)+" negOcc : " + str(occNeg)+" posOcc : " + str(occPos) )




if False:

    voc = vocabulary.createVocabulary(['../data/pos_train_clean.txt', '../data/neg_train_clean.txt'])

    print(str(voc.size()))

    voc = vocabulary.createVocabulary(['../data/pos_train_clean_full.txt', '../data/neg_train_clean_full.txt'])

    print(str(voc.size()))


if False:

    print([1 / 5 for i in range(5)])

    array = [[1,2],[2,3]]

    array = sum(array,[])


    print(array)



    set1 = set([1,2,3,4])
    set2 = set([1,2,3,4])
    set3 = set([1,2,3,4,5])

    cont = [set1,set2,set3]
    one = reduce( (lambda x,y: x.union(y)), cont)

    print(one)







if False:

    dataC = dataCleaning('../data/test_data.txt')
    dataC.setRemoveTestTweetId()
    dataC.setTweetTransform()
    dataC.save('../data/test_data_clean.txt')


