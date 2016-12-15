from src.boosting.trainset import trainset
from src.boosting.vocabulary import vocabulary
from src.dataCleaning import  dataCleaning
import math
import time

from src.boosting.weakLearner import weakLearner






class adaboost:




    def __init__(self, pathToPosTrainFile, pathToNegTrainFile, wordsByWeakLearner=100):


        self.vocabulary = vocabulary.createVocabulary([pathToPosTrainFile,pathToNegTrainFile])

        self.trainset = trainset(self.vocabulary, pathToPosTrainFile, pathToNegTrainFile)

        self.weakLearnerLs = []

        self.wordsByWeakLearner = wordsByWeakLearner

        self.dataClean = dataCleaning()

        """self.__learn()"""





    def getNextWeakLearner(self):

        print("find best word\n")

        """best $wordByWeakLearner words for the next weaklearner"""
        nextWordIds = set()
        tweetHavingWL = set(range(self.trainset.getSize()))


        while len(tweetHavingWL) > 0:

            tweetId = tweetHavingWL.pop()

            best_wordId = None
            best_teta = 1

            for curr_wordId in self.trainset.getWordIds(tweetId):

                curr_teta = self.trainset.get_teta_err(curr_wordId)

                if curr_teta < best_teta and curr_wordId not in nextWordIds:
                    best_teta = curr_teta
                    best_wordId = curr_wordId


            if best_wordId != None:
                nextWordIds.add(best_wordId)
                for tweetIdToRmv in self.trainset.getTweetIdContaining(best_wordId):
                    if tweetIdToRmv in tweetHavingWL:
                        tweetHavingWL.remove(tweetIdToRmv)


                label = self.trainset.get_teta_label(best_wordId)
                rightClass = 0
                wrongClass = 0
                for tweetId in self.trainset.getTweetIdContaining(best_wordId):
                    if self.trainset.getGivenLabel(tweetId) == label:
                        rightClass = rightClass + 1
                    else :
                        wrongClass = wrongClass + 1


        myWordIds = []
        myLabels = []

        for wordId in nextWordIds:
            myWordIds.append(wordId)
            res = self.trainset.get_teta_label(wordId)
            myLabels.append(res)


        return weakLearner(myWordIds, myLabels)






    def learnAndTest(self, posTweetsTest, negTweetsTest):


        wLearner = self.getNextWeakLearner()
        curr_err = self.trainset.getError(wLearner)
        local_voc = self.vocabulary



        while curr_err < 0.5:

            minThreshold = 0.05
            if curr_err < minThreshold:
                curr_err = minThreshold


            wLearner.setWeight(0.5 * math.log((1-curr_err)/curr_err))

            self.weakLearnerLs.append(wLearner)

            print("update...\n")
            modifiedWeight_tweetId = self.trainset.setUpdateWeight(wLearner,curr_err)

            self.trainset.setUpdatetetacache(modifiedWeight_tweetId)

            print("update : done\n")
            local_voc = local_voc.removeKeys(wLearner.getWordIds())

            if(local_voc.size() == 0):
                break

            print("========== TEST PREDICTION :: " + str(self.test(posTweetsTest,negTweetsTest)))



        print("========== TEST PREDICTION :: " + str(self.test(posTweetsTest, negTweetsTest)))
        print([wL.getWeight() for wL in self.weakLearnerLs])




    def test(self, posTweets, negTweets):

        goodPred = 0
        badPred = 0

        for tweet in posTweets:
            if self.predictLabel(tweet) == 1:
                goodPred = goodPred + 1
            else:
                badPred = badPred + 1

        for tweet in negTweets:
            if self.predictLabel(tweet) == -1:
                goodPred = goodPred + 1
            else:
                badPred = badPred + 1

        return goodPred/(goodPred+badPred)



    def predictLabel(self, tweet):

        """TODO : self.dataClean.setTweetTransform(tweet).split(' ')"""

        wordsId = set()

        for word in tweet.strip().split(' '):
            if self.vocabulary.has(word):
                wordsId.add(self.vocabulary.getId(word))

        result = 0

        for wLearner in self.weakLearnerLs:
            result = result + wLearner.getClassification(wordsId)

        """
        print(wordsId)
        print( [wL.getWeight() for wL in self.weakLearnerLs] )

        print(" res : "+str(result)+"\n")
        """

        if result < 0:
            return -1
        else:
            return 1


