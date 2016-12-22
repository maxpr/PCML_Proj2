import math
from functools import reduce
from dataCleaning import dataCleaning

"""
The training set class helps adaboost to manage the weights example
in order to easily compute the error rate of any of the remaining
weak learner.
"""
class trainset:



    def __init__(self, vocabulary, posTweets, negTweets):


        self.vocabulary = vocabulary

        # 1. create 2 map : wordId => set{tweet} and tweetId => set{word}
        self.wordToTweets = []
        self.tweetToWords = []

        for wordId in range(vocabulary.size()):
            self.wordToTweets.append(set())

        currentTweetId = 0

        for tweet in posTweets:
            self.tweetToWords.append(set())
            for word in set(tweet.strip().split(' ')):
                if vocabulary.has(word):
                    self.wordToTweets[vocabulary.getId(word)].add(currentTweetId)
                    self.tweetToWords[currentTweetId].add(vocabulary.getId(word))

            currentTweetId = currentTweetId + 1

        self.firstNegId = currentTweetId

        for tweet in negTweets:
            self.tweetToWords.append(set())
            for word in set(tweet.strip().split(' ')):
                if vocabulary.has(word):
                    self.wordToTweets[vocabulary.getId(word)].add(currentTweetId)
                    self.tweetToWords[currentTweetId].add(vocabulary.getId(word))

            currentTweetId = currentTweetId + 1

        self.size = currentTweetId



        # 2. set all weights to 1/N

        self.tweetWeight = [1/self.size for i in range(self.size)]


        # 3. each word can be choosen as a weak learner,
        # compute pos weight and neg weight for each of them

        self.wordToPosW = []
        self.wordToNegW = []

        for i in range(self.vocabulary.size()):
                self.wordToPosW.append(0)
                self.wordToNegW.append(0)

        for i in range(self.size):
            if self.getExempleLabel(i) == 1:
                for wordId in self.tweetToWords[i]:
                    self.wordToPosW[wordId] += 1/self.size
            else:
                for wordId in self.tweetToWords[i]:
                    self.wordToNegW[wordId] += 1/self.size




    """
    return all tweets containing the given word id
    """
    def getTweetIdContaining(self,wordId):

        return self.wordToTweets[wordId]


    """
    param1: tweetId
    return the output label of the corresponding tweet id
    """
    def getExempleLabel(self, tweetId):

        if tweetId < self.firstNegId:
            return 1
        else:
            return -1



    """
    return the word id in the given tweet id
    """
    def getWordIds(self, tweetId):
        return self.tweetToWords[tweetId]


    """
    return the size of the training set
    """
    def getSize(self):
        return self.size



    """
    param1: tweetId
    return the weight of the requested training example
    """
    def getWeight(self, tweetId):
        return self.tweetWeight[tweetId]



    """
    Return a weak classifier error on the state of the current
    training set.
    """
    def get_pred_err(self, wordId):

        posW = self.wordToPosW[wordId]
        negW = self.wordToNegW[wordId]

        totalTweets = len(self.wordToTweets[wordId])

        err = min([posW, negW])*self.size/totalTweets

        if totalTweets < 15 and err > 0.2:
            err += 0.50
        elif totalTweets < 20 and err > 0.2:
            err += 0.10
        elif totalTweets < 40 and err > 0.3:
            err += 0.05

        return err



    """
    return the prediction made by a wordId
    """
    def get_pred_label(self, wordId):

        posW = self.wordToPosW[wordId]
        negW = self.wordToNegW[wordId]

        if posW > negW:
            return 1
        else:
            return -1




    """
    Update the weights of the training example after a new weak learner was chosen.
    """
    def setUpdateWeight(self, wordId, wLearner, err):

        relativeTweetIdSet = self.wordToTweets[wordId]

        Z = 2*math.sqrt(err*(1-err))

        for tweetId in relativeTweetIdSet:

            tweetLabel = self.getExempleLabel(tweetId)
            expValue = -wLearner.weight*tweetLabel*wLearner.getLabel()

            oldValue = self.tweetWeight[tweetId]
            newValue = oldValue*math.exp(expValue)/Z

            for wordId in self.tweetToWords[tweetId]:
                if tweetLabel == 1:
                    self.wordToPosW[wordId] += (newValue - oldValue)
                else:
                    self.wordToNegW[wordId] += (newValue - oldValue)


            self.tweetWeight[tweetId] = newValue





