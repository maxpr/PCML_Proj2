from src.boosting.vocabulary import vocabulary
from src.boosting.trainset import trainset
from src.boosting.weakLearner import weakLearner
from src.dataCleaning import dataCleaning
from src.boosting.adaboost import adaboost

"""
verify the trainset
"""
if False:



    vocabulary = vocabulary(['../data/pos_train_clean_cutTrain.txt', '../data/neg_train_clean_cutTrain.txt'])

    trainData = trainset(vocabulary,
                        '../data/pos_train_clean_cutTrain.txt',
                        '../data/neg_train_clean_cutTrain.txt')

    posTrain = dataCleaning('../data/pos_train_clean_cutTrain.txt').getData()
    negTrain = dataCleaning('../data/neg_train_clean_cutTrain.txt').getData()

    allData = posTrain[:]

    firstNegId = len(allData)

    for negTweet in negTrain:
        allData.append(negTweet)





    isSameData = (len(posTrain) == len(trainData.posTweets)) and (len(negTrain) == len(trainData.negTweets))

    for (data1, data2) in zip(posTrain, trainData.posTweets):
        isSameData = isSameData and (data1 == data2)


    for (data1, data2) in zip(negTrain, trainData.negTweets):
        isSameData = isSameData and (data1 == data2)

    print("work on same data : " + str(isSameData))

    isSameFirstNegAndSize = firstNegId == trainData.firstNegId and trainData.size == len(allData)

    print("same firsNeg and size : " + str(isSameFirstNegAndSize))



    """  COUNT OCC """

    wordToTweets = {}

    for word in vocabulary.getValues():
        wordToTweets[word] = set()

    wordToTotalPosTweet = {}
    wordToTotalNegTweet = {}

    currTweetId = 0

    for tweet in posTrain:
        for word in set(tweet.strip().split(' ')):

            if vocabulary.has(word):
                wordToTweets[vocabulary.getId(word)].add(currTweetId)

            if word in wordToTotalPosTweet and vocabulary.has(word):
                wordToTotalPosTweet[word] = wordToTotalPosTweet[word] + 1
            elif vocabulary.has(word):
                wordToTotalPosTweet[word] = 1

        currTweetId = currTweetId + 1

    firstNegId = currTweetId

    for tweet in negTrain:
        for word in set(tweet.strip().split(' ')):

            if vocabulary.has(word):
                wordToTweets[vocabulary.getId(word)].add(currTweetId)

            if word in wordToTotalNegTweet and vocabulary.has(word):
                wordToTotalNegTweet[word] = wordToTotalNegTweet[word] + 1
            elif vocabulary.has(word):
                wordToTotalNegTweet[word] = 1

        currTweetId = currTweetId + 1




    i = 0
    for word in vocabulary.getKey():

        trainData = trainset(vocabulary,
                        '../data/pos_train_clean_cutTrain.txt',
                        '../data/neg_train_clean_cutTrain.txt')

        tetaerr = trainData.compute_teta_err(vocabulary.getId(word))


        """
        for tweetId in trainData.getTweetIdContaining(word):
            print("======================>" + word)
            print(allData[tweetId])
        """
        print(word)
        totalPos = 0
        totalNeg = 0
        for tweet in posTrain:
            if word in set(tweet.strip().split()):
                totalPos = totalPos+1

        for tweet in negTrain:
            if word in set(tweet.strip().split()):
                totalNeg = totalNeg + 1

        print("real total : "+str(totalPos+totalNeg))

        rslt = trainData.wordToTweets[vocabulary.getId(word)]

        """
        rsltNeg = [t for t in rslt if trainData.getGivenLabel(t) == -1]
        rsltPos = [t for t in rslt if trainData.getGivenLabel(t) == 1]
        """

        """
        print("2nd real total : " +str(len(wordToTweet[vocabulary.getId(word)])))

        print("real pos : " + str(totalPos) + " / real neg : " + str(totalNeg))
        print("expected pos : " + str(totalInPos) + " / neg : " +str(totalInNeg))
        """
        """
        print("result pos : " + str(len(rsltPos)) + " / neg : " + str(len(rsltNeg)))
        """

        tetaerr_expected =  min([totalNeg,totalPos])*1/len(allData)/(totalPos+totalNeg)

        print("\n\nteta result : "+str(tetaerr) + " / teta expected : " + str(tetaerr_expected))


        wordId = vocabulary.getId(word)
        label = trainData.get_teta_label(wordId)

        wLearner = weakLearner([wordId], [label])

        wLErr = trainData.getError(wLearner)

        print("weak learnning err : " + str(wLErr))

        modifyTweetsId = wordToTweets[vocabulary.getId(word)]

        """
        no modifications if weights are not updated
        """
        allTetaErr1 = trainData.teta_cache.copy()
        trainData.setUpdatetetacache(modifyTweetsId)
        allTetaErr2 = trainData.teta_cache.copy()

        hasSameTetaErr = len(allTetaErr1) == len(allTetaErr2)
        for (err1,err2) in zip(allTetaErr1,allTetaErr2):
            hasSameTetaErr = hasSameTetaErr and (err1 == err2)

        print("useless update has no impact : " + str(hasSameTetaErr))

        trainData.setUpdateWeight(wLearner,wLErr)
        trainData.setUpdatetetacache(modifyTweetsId)
        allTetaErrWithUpdate = trainData.teta_cache.copy()


        modifyWordsId = set()
        for tweetId in modifyTweetsId:
            for word in allData[tweetId].strip().split(' '):
                if vocabulary.has(word):
                    modifyWordsId.add(vocabulary.getId(word))

        isUnmodifiableWordSameErr = True
        for wordId in vocabulary.getValues():
            if wordId not in modifyWordsId:
                isUnmodifiableWordSameErr = isUnmodifiableWordSameErr and (allTetaErr1[wordId] == allTetaErrWithUpdate[wordId])

        print("unmodified tweet has same err = " + str(isUnmodifiableWordSameErr))


        totalTweetClassified = 0
        totalTweetRightClassified = 0

        for tweetId in modifyTweetsId:

            wordsId = set()
            for word in allData[tweetId].strip().split(' '):
                if vocabulary.has(word):
                    wordsId.add(vocabulary.getId(word))

            predClass = wLearner.getClassification(wordsId)
            realClass = trainData.getGivenLabel(tweetId)

            totalTweetClassified = totalTweetClassified + 1

            if predClass == realClass:
                totalTweetRightClassified = totalTweetRightClassified + 1


        predSucessRatio = totalTweetRightClassified/totalTweetClassified

        print("weak learner pred err ratio = " + str(1-predSucessRatio))

        i = i+1
        if i>3:
            break



if True:

    vocabulary = vocabulary(['../data/pos_train_clean_cutTrain.txt', '../data/neg_train_clean_cutTrain.txt'])

    boosting = adaboost('../data/pos_train_clean_cutTrain.txt', '../data/neg_train_clean_cutTrain.txt')


    posTrain = dataCleaning('../data/pos_train_clean_cutTrain.txt').getData()
    negTrain = dataCleaning('../data/neg_train_clean_cutTrain.txt').getData()

    allData = posTrain[:]

    firstNegId = len(allData)

    for negTweet in negTrain:
        allData.append(negTweet)


    wL = boosting.getNextWeakLearner()


    rightClass = 0
    wrongClass = 0

    for tweet in posTrain:
        wIds = set([ vocabulary.getId(word) for word in tweet.strip().split(' ') if vocabulary.has(word)])
        pred = wL.getClassification(wIds)
        if pred == 1:
            rightClass = rightClass + 1
        else:
            wrongClass = wrongClass + 1


    for tweet in negTrain:
        wIds = set([ vocabulary.getId(word) for word in tweet.strip().split(' ') if vocabulary.has(word)])
        pred = wL.getClassification(wIds)
        if pred == -1:
            rightClass = rightClass + 1
        else:
            wrongClass = wrongClass + 1


    print("===> " + str(rightClass/(rightClass+wrongClass)))












