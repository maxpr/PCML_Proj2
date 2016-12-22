import random
import string

from src.boosting.trainset import trainset
from src.boosting.vocabulary import vocabulary
import math

from src.boosting.weakLearner import weakLearner
from src.dataCleaning import dataCleaning


class adaboost:




    def __init__(self, posTweets, negTweets, wordsByWeakLearner=1):

        self.vocabulary = vocabulary.createVocabulary([posTweets,negTweets],10)

        self.trainset = trainset(self.vocabulary, posTweets, negTweets)

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


        return best_wordId, weakLearner(self.trainset.get_pred_label(best_wordId))





    """
    learn from the train set and keep a log with the accuracy
    """
    def learnAndTest(self, pathToOuputRes, posTweetsTest, negTweetsTest, tweetsToPred, weightThreshold=0.5, unselected_wordIds = None):

        if  unselected_wordIds == None:

            tmp = [i for i in range(self.vocabulary.size())]

            unselected_wordIds = []

            for wordId in tmp:
                unselected_wordIds.append(wordId)



        curr_wordId, wLearner = self.getNextWeakLearner(unselected_wordIds)
        curr_err = self.trainset.get_pred_err(curr_wordId)


        i = 1

        while curr_err < weightThreshold:

            if curr_err == 0:
                curr_err = max([curr_err, 0.0000000000000000001])

            print(str(i) + " " + str(len(unselected_wordIds)))
            wLearner.setWeight(0.5 * math.log((1-curr_err)/curr_err))

            self.wordToWeakLearner[self.vocabulary.getWord(curr_wordId)] = wLearner

            self.trainset.setUpdateWeight(curr_wordId,wLearner,curr_err)

            unselected_wordIds.remove(curr_wordId)

            if len(unselected_wordIds) == 0:
                break
            else:
                curr_wordId, wLearner = self.getNextWeakLearner(unselected_wordIds)
                curr_err = self.trainset.get_pred_err(curr_wordId)
                i = i+1


            if(i%1500 == 0 or len(unselected_wordIds) == 0 or curr_err>=0.5):

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
                for (word, wL) in self.wordToWeakLearner.items():
                    line = word
                    line += "\t\t" + str(wL.getLabel())
                    line += "\t\t" + str(wL.getWeight()) + "\n"
                    resFile.write(line)

                resFile.close()

                results = []
                for tweet in tweetsToPred:
                    results.append(self.predictLabel(tweet))

                dataCleaning.saveTestData(resPath2,results)

                print(str(i)+"/" + str(self.vocabulary.size()) + " ========== TEST PREDICTION :: " + str(rsltTest))



    """
    test the data given in parameters using the
    weak learner computed in test and learn
    """
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



    """
    predict the label of the tweet given as parameter
    using the weak learner computed in test and learn
    """
    def predictLabel(self, tweet):

        result = 0
        for word in set(tweet.strip().split(' ')):
            if word in self.wordToWeakLearner:
                result = result + self.wordToWeakLearner[word].getLabel() * self.wordToWeakLearner[word].getWeight()

        if result < 0:
            return -1
        else:
            return 1


