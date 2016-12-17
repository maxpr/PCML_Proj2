import math
from functools import reduce
from src.dataCleaning import dataCleaning

"""
The training set of tweets with their relative labels and usefull functions
used in order to aid adaboost to parametrize their weak learners.
"""
class trainset:


    """
    Create the two directories :

    self.wordToTweets -> for a word identified in the vocabulary it maps all related tweet's id
    self.tweetToWords -> for a tweet identifier it maps all the related word's id that can be
                         identified in the vocabulary

    It also define the weight of all tweets exemple as 1/total_number_of_tweet_exemple.
    These weights will be modified by adaboost.
    """
    def __init__(self, vocabulary, pathToPosTrainFile, pathToNegTrainFile):


        self.vocabulary = vocabulary

        self.wordToTweets = []
        self.tweetToWords = []

        for wordId in range(vocabulary.size()):
            self.wordToTweets.append(set())


        posTweets = dataCleaning(pathToPosTrainFile).getData()
        negTweets = dataCleaning(pathToNegTrainFile).getData()

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


        self.tweetWeight = [1/self.size for i in range(self.size)]

        self.teta_cache = []
        for i in range(self.vocabulary.size()):
            self.teta_cache.append(self.compute_teta_err(i))






    def getTweetIdContaining(self,wordId):

        return self.wordToTweets[wordId]


    """
    param1: tweetId
    return the output label of the corresponding training exemple
    """
    def getGivenLabel(self, tweetId):

        if tweetId < self.firstNegId:
            return 1
        else:
            return -1



    def getWordIds(self, tweetId):
        return self.tweetToWords[tweetId]

    def getSize(self):
        return self.size


    def getTrainingSize(self):
        return self.size



    """
    param1: tweetId
    return the adaboost weight of corresponding training exemple
    """
    def getWeight(self, tweetId):

        """
        print("tweet id : "+ str(tweetId)+"\n")
        print("size : " + str(len(self.tweetWeight))+"\n")
        """

        return self.tweetWeight[tweetId]



    """
    If we try to use a word as a partial indicator to classification,
    we have to get the information, how the word will help in the process.
    the teta value returned is an error value over the subset of tweets concerned
    by the indicator. Then it computes the sum of the weights over the positive/negative
    tweets, it considers that word is an indicator for the most weighted class and returns
    the lower weight as teta. This weight is divided by the subset size of the
    concerned tweets in order to be comparable.
    """
    def compute_teta_err(self, wordId):

        posW = 0
        negW = 0
        for tweetId in self.wordToTweets[wordId]:
            if self.getGivenLabel(tweetId) == 1:
                posW += self.getWeight(tweetId)
            else:
                negW += self.getWeight(tweetId)


        """ TODO : also try to divide by the additional sum"""
        return min([posW, negW])/len(self.wordToTweets[wordId])*self.size



    def compute_teta_heuristic(self, wordId):

        """ TODO : also try to divide by the additional sum"""
        return self.compute_teta_err(wordId)/math.sqrt(len(self.wordToTweets[wordId]))


    """
    param1: a word identifier in the vocabulary used to build the class
    return the classification error (sum of exemple's weight on classification error)
    and the label for the word (ie: word can classify 1 or -1 a tweet)
    """
    def get_teta_err(self, wordId):

        return self.teta_cache[wordId]






    def get_teta_label(self, wordId):


        tweetIds = self.wordToTweets[wordId]

        posWeights = [self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == 1]
        negWeights = [self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == -1]

        posW = sum(posWeights)
        negW = sum(negWeights)

        """ teta indicator (~weaklearner) : indicates class with the bigger weights """
        if posW > negW:
            return 1
        else:
            return -1



    """
    param1: a word identifier in the vocabulary used to build the class
    return the classification error (sum of exemple's weight on classification error)
    and the label for the word (ie: word can classify 1 or -1 a tweet)
    """
    def getError(self, wLearner):

        relativeTweetIdSet = self.wordToTweets[wLearner.getWordId()]
        pred = wLearner.getLabel()

        err = 0

        for tweetId in relativeTweetIdSet:
            if (pred > 0 and self.getGivenLabel(tweetId)<0)  or (pred < 0 and self.getGivenLabel(tweetId)>0):
                err = err + self.getWeight(tweetId)


        return err/len(relativeTweetIdSet)*self.size




    """
    update the weight of the training exemple in order to find the best weak classifier
    over the highly weighted exemples (tweets that aren't in the right class are heavily weighted)
    """
    def setUpdateWeight(self, wLearner, err):

        relativeTweetIdSet = self.wordToTweets[wLearner.getWordId()]

        Z = 2*math.sqrt(err*(1-err))

        for tweetId in relativeTweetIdSet:

            expValue = -wLearner.weight*self.getGivenLabel(tweetId)*wLearner.getLabel()

            value = math.exp(expValue)/Z

            self.tweetWeight[tweetId] = self.tweetWeight[tweetId]*value

        return relativeTweetIdSet




    def setUpdatetetacache(self, tweetIdLs_weightModif):

        if len(tweetIdLs_weightModif) == 0:
            print("unexpected arg")
            return 0

        wordIds_inModif = [self.tweetToWords[tweetId] for tweetId in tweetIdLs_weightModif]
        wordIds_inModif = reduce( ( lambda x,y: x.union(y) ),wordIds_inModif)

        for wordId in wordIds_inModif:
                self.teta_cache[wordId] = self.compute_teta_err(wordId)





