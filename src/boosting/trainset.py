import math
from functools import reduce


"""
The training set of tweets with their relative labels and usefull functions
used in order to aid adaboost to parametrize their weak learners.
"""
class trainset:



    """
    TODO : ratio

    Create the two directories :

    self.wordToTweets -> for a word identified in the vocabulary it maps all related tweet's id
    self.tweetToWords -> for a tweet identifier it maps all the related word's id that can be
                         identified in the vocabulary

    It also define the weight of all tweets exemple as 1/total_number_of_tweet_exemple.
    These weights will be modified by adaboost.
    """
    def __init__(self, vocabulary, pathToPosTrainFile, pathToNegTrainFile):


        self.vocabulary = vocabulary

        self.wordToTweets = {}
        self.tweetToWords = {}

        for wordId in range(vocabulary.size()):
            self.wordToTweets[wordId] = []


        posfile = open(pathToPosTrainFile)
        negfile = open(pathToNegTrainFile)

        currentTweetId = 0

        for tweet in posfile:
            self.tweetToWords[currentTweetId] = set()
            for word in tweet.split(' '):
                if vocabulary.has(word):
                    self.wordToTweets[vocabulary.getId(word)].append(currentTweetId)
                    self.tweetToWords[currentTweetId].add(vocabulary.getId(word))

            currentTweetId = currentTweetId + 1

        self.firstNegId = currentTweetId

        for tweet in negfile:
            self.tweetToWords[currentTweetId] = set()
            for word in tweet.split(' '):
                if vocabulary.has(word):
                    self.wordToTweets[vocabulary.getId(word)].append(currentTweetId)
                    self.tweetToWords[currentTweetId].add(vocabulary.getId(word))

            currentTweetId = currentTweetId + 1

        self.size = currentTweetId
        negfile.close()
        posfile.close()

        self.tweetWeight = [1/self.size for i in range(self.size)]


        self.Ω_cache = {}

        for wordId in self.vocabulary.getValues():

            tweetIds = self.wordToTweets[wordId]
            negΩ = sum([self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == 1])
            posΩ = sum([self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == -1])

            if posΩ < negΩ:
                self.Ω_cache[wordId] = posΩ
            else:
                self.Ω_cache[wordId] = negΩ




    """
    param1: tweetId
    return the output label of the corresponding training exemple
    """
    def getGivenLabel(self, tweetId):

        if tweetId < self.firstNegId :
            return 1
        else:
            return -1


    """
    param1: tweetId
    return the adaboost weight of corresponding training exemple
    """
    def getWeight(self, tweetId):
        return self.tweetWeight[tweetId]



    """
    param1: a word identifier in the vocabulary used to build the class
    return the classification error (sum of exemple's weight on classification error)
    and the label for the word (ie: word can classify 1 or -1 a tweet)
    """
    def get_Ω(self, wordId):
        return self.Ω_cache[wordId]






    def getLabelIndicator(self, wordId):

        relativeTweetIdLs = self.wordToTweets[wordId]

        posLs = [tweetId for tweetId in relativeTweetIdLs if self.getGivenLabel(tweetId) == 1]
        negLs = [tweetId for tweetId in relativeTweetIdLs if self.getGivenLabel(tweetId) == -1]

        negClassificationError = sum([self.getWeight(tweetId) for tweetId in posLs])
        posClassificationError = sum([self.getWeight(tweetId) for tweetId in negLs])

        if negClassificationError < posClassificationError:
            return -1
        else:
            return 1



    """
    param1: a word identifier in the vocabulary used to build the class
    return the classification error (sum of exemple's weight on classification error)
    and the label for the word (ie: word can classify 1 or -1 a tweet)
    """
    def getError(self, wLearner):

        relativeTweetIdSet = [self.wordToTweets[wordId] for wordId in wLearner.getWordIds()]
        relativeTweetIdSet = set(sum(relativeTweetIdSet, []))

        err = 0

        for tweetId in relativeTweetIdSet:

            isCorrectlyClassify = (self.getGivenLabel(tweetId)  == wLearner.getClassification(self.tweetToWords[tweetId]))

            if not isCorrectlyClassify:
                err = err + self.getWeight(tweetId)

        return err




    """
    update the weight of the training exemple in order to find the best weak classifier
    over the highly weighted exemples (tweets that aren't in the right class are heavily weighted)
    """
    def setUpdateWeight(self, weakLearner, err):

        relativeTweetIdSet = [self.wordToTweets[wordId] for wordId in weakLearner.getWordIds()]
        relativeTweetIdSet = set(sum(relativeTweetIdSet, []))

        Z = max([2*math.sqrt(err*(1-err)),0.01])

        for tweetId in relativeTweetIdSet:

            expValue = -weakLearner.weight*self.getGivenLabel(tweetId)*weakLearner.getClassification(self.tweetToWords[tweetId])

            value = math.exp(expValue)/Z

            self.tweetWeight[tweetId] = self.tweetWeight[tweetId]*value

        return relativeTweetIdSet




    def setUpdateΩcache(self, tweetIdLs_weightModif):


        wordIds_inModif = reduce( ( lambda x,y: x.union(y) ),
                                 [self.tweetToWords[tweetId] for tweetId in tweetIdLs_weightModif]
                                 )

        for wordId in wordIds_inModif:

            tweetIds = self.wordToTweets[wordId]
            negΩ = sum([self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == 1])
            posΩ = sum([self.getWeight(tweetId) for tweetId in tweetIds if self.getGivenLabel(tweetId) == -1])

            if posΩ < negΩ:
                self.Ω_cache[wordId] = posΩ
            else:
                self.Ω_cache[wordId] = negΩ





