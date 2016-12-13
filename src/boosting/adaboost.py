from src.boosting.trainset import trainset
from src.boosting.vocabulary import vocabulary
from src.dataCleaning import  dataCleaning
import math

from src.boosting.weakLearner import weakLearner






class adaboost:




    def __init__(self, pathToPosTrainFile, pathToNegTrainFile, wordsByWeakLearner=100):


        self.vocabulary = vocabulary([pathToPosTrainFile,pathToNegTrainFile])

        self.trainset = trainset(self.vocabulary, pathToPosTrainFile, pathToNegTrainFile)

        self.weakLearnerLs = []

        self.wordsByWeakLearner = wordsByWeakLearner

        self.dataClean = dataCleaning()

        """self.__learn()"""





    def __getNextWeakLearner(self):

        print("find best word\n")

        """best $wordByWeakLearner words for the next weaklearner"""
        nextWordIds = set()

        for i in range(self.wordsByWeakLearner):

            best_wordId = None
            best_Ω = 1

            for curr_wordId in self.vocabulary.getValues():

                curr_Ω = self.trainset.get_Ω(curr_wordId)

                if curr_Ω < best_Ω and curr_wordId not in nextWordIds:
                    best_Ω = curr_Ω
                    best_wordId = curr_wordId

            nextWordIds.add(best_wordId)


        myWordIds = []
        myLabels = []

        for wordId in nextWordIds:
            myWordIds.append(wordId)
            myLabels.append(self.trainset.getLabelIndicator(wordId))

        return weakLearner(myWordIds, myLabels)






    def __learn(self):


        wLearner = self.__getNextWeakLearner()
        curr_err = self.trainset.getError(wLearner)


        while curr_err < 0.5:

            wLearner.setWeight( 0.5 * math.log((1-curr_err)/max([curr_err,0.01])) )
            self.weakLearnerLs.append(wLearner)

            print("update...\n")
            modifiedWeight_tweetId = self.trainset.setUpdateWeight(wLearner,curr_err)

            self.trainset.setUpdateΩcache(modifiedWeight_tweetId)

            print("update : done\n")
            self.vocabulary.removeKeys(wLearner.getWordIds())

            if(self.vocabulary.size() == 0):
                break

            wLearner = self.__getNextWeakLearner()
            curr_err = self.trainset.getError(wLearner)





    def learnAndTest(self, posTweetsTest, negTweetsTest):


        wLearner = self.__getNextWeakLearner()
        curr_err = self.trainset.getError(wLearner)

        i = 0

        while curr_err < 0.5:

            i = i + 1

            wLearner.setWeight( 0.5 * math.log((1-curr_err)/max([curr_err,0.01])) )
            self.weakLearnerLs.append(wLearner)

            print("update...\n")
            modifiedWeight_tweetId = self.trainset.setUpdateWeight(wLearner,curr_err)

            self.trainset.setUpdateΩcache(modifiedWeight_tweetId)

            print("update : done\n")
            self.vocabulary.removeKeys(wLearner.getWordIds())

            if(self.vocabulary.size() == 0):
                break

            wLearner = self.__getNextWeakLearner()
            curr_err = self.trainset.getError(wLearner)


            if i%20 == 0:
                print("========== TEST PREDICTION :: " + str(self.test(posTweetsTest,negTweetsTest)) )







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

        words = tweet.split('\n')

        """TODO : self.dataClean.setTweetTransform(tweet).split(' ')"""

        wordsId = set()

        for word in words:
            if self.vocabulary.has(word):
                wordsId.add(self.vocabulary.getId(word))

        result = 0

        for wLearner in self.weakLearnerLs:
            print()
            result = result + wLearner.getClassification(wordsId)

        if result < 0:
            return -1
        else:
            return 1


