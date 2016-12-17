import math

from src.boosting.weakLearner import weakLearner
from src.dataCleaning import dataCleaning
from src.boosting.adaboost import adaboost
from functools import reduce
from src.boosting.vocabulary import vocabulary


def occCounter(ls):

    occ = {}
    for t in ls:
        if t in occ:
            occ[t] = occ[t] +1
        else:
            occ[t] = 1

    return occ


def restoreAda(adaboost):

    wLearners = {}

    resFile = open('path', 'w')
    for line in resFile:
        value = line.strip().split('\t\t')
        word = value[0]
        label = int(float(value[1]))
        weight = float(value[2])

        if vocabulary.has(word):
            wLearners[adaboost.vocabulary.getId(word)] = weakLearner(adaboost.vocabulary.getId(word), label,weight)

    adaboost.wordToWeakLearner = wLearners

    for tweetId in range(adaboost.trainset.getSize()):

        expValue = 0

        for wordId in adaboost.trainset.tweetToWords[tweetId]:
            wLearner = wLearners[wordId]
            if wordId in wLearners:
                expValue = - wLearner.weight * adaboost.trainset.getExempleLabel(tweetId) * wLearner.getLabel()

        adaboost.trainset.tweetToWords[tweetId] = adaboost.trainset.tweetToWords[tweetId]*math.exp(expValue)


    totalWeight = 0

    for tweetId in range(adaboost.trainset.getSize()):
        totalWeight += adaboost.trainset.getWeight(tweetId)

    for tweetId in range(adaboost.trainset.getSize()):
        adaboost.trainset.tweetWeight[tweetId] = adaboost.trainset.getWeight(tweetId)/totalWeight


    resFile.close()




if False:


    one = {}

    one[1] = 1
    one[2] = 2

    two = one.copy()

    two[2] = 4

    print(one)
    print(two)

if False:


    neg = dataCleaning('../data/train_neg.txt').getData()
    pos = dataCleaning('../data/train_pos.txt').getData()


    negRes1 = []
    negRes2 = []

    for tweet in neg:

        wordOcc = {}
        for word in tweet.strip().split(' '):
            if word in wordOcc:
                wordOcc[word] = wordOcc[word] +1
            else:
                wordOcc[word] = 1

        mostOcc = 0
        mostOccWord = None
        for (word, occ) in wordOcc.items():
            if occ > mostOcc:
                mostOcc = occ
                mostOccWord = word

        negRes1.append(mostOcc)
        mostOcc = 0
        mostOccWord = None
        for (word, occ) in wordOcc.items():
            if occ > mostOcc and word != mostOccWord:
                mostOcc = occ
                mostOccWord = word

        negRes2.append(mostOcc)


    posRes1 = []
    posRes2 = []


    for tweet in pos:

        wordOcc = {}
        for word in tweet.strip().split(' '):
            if word in wordOcc:
                wordOcc[word] = wordOcc[word] + 1
            else:
                wordOcc[word] = 1

        mostOcc = 0
        mostOccWord = None
        for (word, occ) in wordOcc.items():
            if occ > mostOcc:
                mostOcc = occ
                mostOccWord = word

        posRes1.append(mostOcc)
        mostOcc = 0
        mostOccWord = None
        for (word, occ) in wordOcc.items():
            if occ > mostOcc and word != mostOccWord:
                mostOcc = occ
                mostOccWord = word

        posRes2.append(mostOcc)


    posRes1 = occCounter(posRes1)
    posRes2 = occCounter(posRes2)
    negRes1 = occCounter(negRes1)
    negRes2 = occCounter(negRes2)

    for i in range(10):
        pos1 = 0
        pos2 = 0
        neg1 = 0
        neg2 = 0
        if i in posRes1:
            pos1 = posRes1[i]
        if i in posRes2:
            pos2 = posRes2[i]
        if i in negRes1:
            neg1 = negRes1[i]
        if i in posRes2:
            neg2 = negRes2[i]
        print(str(pos1)+ "/" +str(neg1) + "\t\t\t" + str(pos2) + "/" +str(neg2))


if False:

    neg = dataCleaning('../data/neg_train_clean_v1.txt').getData()
    pos = dataCleaning('../data/pos_train_clean_v1.txt').getData()


    the = 0
    on = 0
    excl = 0


    for tweet in neg:
        words = set(tweet.strip().split(' '))
        if "the" in words:
            the += 1
        if "is" in words:
            on += 1
        if "!_1" in words:
            excl += 1

    for tweet in pos:
        words = set(tweet.strip().split(' '))
        if "the"in words:
            the += 1
        if "is" in words:
            on += 1
        if "!_1" in words:
            excl += 1

    the = the/(len(neg)+len(pos))
    on = on/(len(neg)+len(pos))
    excl = excl/(len(neg)+len(pos))

    print( str(the) +"   " + str(on) + "   " + str(excl) )

    maxThreeshold = 0.18*(len(neg)+len(pos))


    wordToOcc = {}

    for tweet in neg:
        """  TODO : could be considered out set"""
        for word in set(tweet.strip().split(' ')):
            if word not in wordToOcc:
                wordToOcc[word] = 1
            else:
                wordToOcc[word] = wordToOcc[word] + 1

    for tweet in pos:
        """  TODO : could be considered out set"""
        for word in set(tweet.strip().split(' ')):
            if word not in wordToOcc:
                wordToOcc[word] = 1
            else:
                wordToOcc[word] = wordToOcc[word] + 1

    for (word, occ) in wordToOcc.items():
        if occ > maxThreeshold:
            print(word)




if False:

    dataC = dataCleaning('../data/train_pos_full.txt')
    dataC.cutAndSave(0.5,'../data/train_pos_v1_1.txt','../data/train_pos_v1_2.txt')
    dataC = dataCleaning('../data/train_neg_full.txt')
    dataC.cutAndSave(0.5,'../data/train_neg_v1_1.txt','../data/train_neg_v1_2.txt')

if False:
    dataC = dataCleaning('../data/train_pos_v1_1.txt')
    dataC.cutAndSave(0.9, '../data/train_pos_v1_1.txt', '../data/test_pos_v1_1.txt')
    dataC = dataCleaning('../data/train_neg_v1_1.txt')
    dataC.cutAndSave(0.9, '../data/train_neg_v1_1.txt', '../data/test_neg_v1_1.txt')

if False:

    dataC = dataCleaning('../data/train_pos_v1_1.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.cutAndSave(0.9,'../data/pos_train_clean_v1.txt','../data/pos_test_clean_v1.txt')
    print('done')


    dataC = dataCleaning('../data/train_neg_v1_1.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.cutAndSave(0.9,'../data/neg_train_clean_v1.txt','../data/neg_test_clean_v1.txt')
    print('done')


    dataC = dataCleaning('../data/test_data.txt')

    dataC.setRemoveTestTweetId()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.save('../data/test_data_clean_v1.txt')
    print('done')



if False:
    dataC = dataCleaning('../data/pos_train_full.txt')
    dataC.cutAndSave(0.5, '../data/pos_train_full1.txt','../data/pos_train_full2.txt')

    dataC = dataCleaning('../data/neg_train_full.txt')
    dataC.cutAndSave(0.5, '../data/neg_train_full1.txt','../data/neg_train_full2.txt')

    print('done')

if False:

    dataC = dataCleaning('../data/pos_train_full1.txt')
    dataC.cutAndSave(0.25, '../data/pos_train_clean_v1.txt','../data/delete1.txt')

    dataC = dataCleaning('../data/neg_train_full1.txt')
    dataC.cutAndSave(0.25, '../data/neg_train_clean_v1.txt','../data/delete2.txt')


if False:
    dataC = dataCleaning('../data/test_data.txt')
    print('set tweet transformation, can take some times...')
    dataC.setRemoveTestTweetId()
    dataC.setTweetTransform()
    print('save the data')
    dataC.save('../data/test_data_clean_v1.txt')
    print('done')



if False:

    dataC = dataCleaning('../data/pos_train_clean_v1.txt')
    dataC.cutAndSave(0.5, '../data/pos_train_clean_v1_1.txt','../data/pos_train_clean_v1_2.txt')

    dataC = dataCleaning('../data/neg_train_clean_v1.txt')
    dataC.cutAndSave(0.5, '../data/neg_train_clean_v1_1.txt','../data/neg_train_clean_v1_2.txt')




if True:

    boosting = adaboost('../data/pos_train_clean_v1_1.txt','../data/neg_train_clean_v1_1.txt')

    posTest = dataCleaning('../data/pos_test_clean_v1.txt').getData()
    negTest = dataCleaning('../data/neg_test_clean_v1.txt').getData()
    tweetsToPred = dataCleaning('../data/test_data_clean_v1.txt').getData()

    boosting.learnAndTest('../data/pred_output_index.txt', posTest, negTest, tweetsToPred)



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


