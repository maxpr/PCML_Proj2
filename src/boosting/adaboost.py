import random
import string

from src.boosting.trainset import trainset
from src.boosting.vocabulary import vocabulary
import math

from src.boosting.weakLearner import weakLearner
from src.dataCleaning import dataCleaning


class adaboost:




    def __init__(self, pathToPosTrainFile, pathToNegTrainFile, wordsByWeakLearner=1):

        self.vocabulary = vocabulary.createVocabulary([pathToPosTrainFile,pathToNegTrainFile])

        self.trainset = trainset(self.vocabulary, pathToPosTrainFile, pathToNegTrainFile)

        self.wordToWeakLearner = {}

        self.wordsByWeakLearner = wordsByWeakLearner


    """
    return the next weaklearner with the less error on the trainset
    """
    def getNextWeakLearner(self, unselected_wordIds):

        """best $wordByWeakLearner words for the next weaklearner"""
        best_wordId = None
        best_err = 1

        for curr_wordId in unselected_wordIds:

            curr_err = self.trainset.get_pred_err(curr_wordId)

            if curr_err < best_err:
                best_err = curr_err
                best_wordId = curr_wordId


        return weakLearner(best_wordId, self.trainset.get_pred_label(best_wordId))





    """
    learn from the train set and keep a log with the accuracy
    """
    def learnAndTest(self, pathToOuputRes, posTweetsTest, negTweetsTest, tweetsToPred):


        unselected_wordIds = [i for i in range(self.vocabulary.size())]

        wLearner = self.getNextWeakLearner(unselected_wordIds)
        curr_err = self.trainset.get_pred_err(wLearner.getWordId())
        minThreshold = 0.000000001


        i = 1

        while curr_err < 0.5:

            print(str(i)+"/"+str(self.vocabulary.size()) + "  ==> err = " +str(curr_err))

            if curr_err < minThreshold:
                curr_err = minThreshold

            wLearner.setWeight(0.5 * math.log((1-curr_err)/curr_err))

            self.wordToWeakLearner[wLearner.getWordId()] = wLearner

            modifiedWeight_tweetId = self.trainset.setUpdateWeight(wLearner,curr_err)


            self.trainset.setUpdateErrcache(modifiedWeight_tweetId)

            unselected_wordIds.remove(wLearner.getWordId())


            if(len(unselected_wordIds) == 0):
                break

            if(i%100 == 0):

                rsltTest = self.test(posTweetsTest,negTweetsTest)
                fileOut = open(pathToOuputRes,'a')

                resPath1 = '../output/'
                resPath1 += ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                resPath1 += '.txt'

                resPath2 = '../output/'
                resPath2 += ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                resPath2 += '.txt'


                detail_1 = str(i) +"\t\t"+ str(rsltTest) +"\t\t"+ resPath1 +"\t\t"+ resPath2 + "\n"


                fileOut.write(detail_1)
                fileOut.close()

                resFile = open(resPath1, 'w')
                for wL in self.wordToWeakLearner.values():
                    line = self.vocabulary.getWord(wL.getWordId())
                    line += "\t\t" + str(wL.getLabel())
                    line += "\t\t" + str(wL.getWeight()) + "\n"
                    resFile.write(line)

                resFile.close()

                results = []
                for tweet in tweetsToPred:
                    results.append(self.predictLabel(tweet))

                dataCleaning.saveTestData(resPath2,results)

                print(str(i)+"/" + str(self.vocabulary.size()) + " ========== TEST PREDICTION :: " + str(rsltTest))




            wLearner = self.getNextWeakLearner(unselected_wordIds)
            curr_err = self.trainset.get_pred_err(wLearner.getWordId())

            i = i+1





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

        result = 0
        for word in set(tweet.strip().split(' ')):
            if self.vocabulary.has(word) and self.vocabulary.getId(word) in self.wordToWeakLearner:
                id = self.vocabulary.getId(word)
                result = result + self.wordToWeakLearner[id].getLabel() * self.wordToWeakLearner[id].getWeight()

        if result < 0:
            return -1
        else:
            return 1


