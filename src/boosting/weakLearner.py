
"""
weak learner used to partially clusterize tweets
"""
class weakLearner:



    def __init__(self, wordIds, labels, weight = 1):

        self.wordIds = wordIds
        self.weight = weight
        self.labels = labels



    def setWeight(self, weight):
        self.weight = weight


    def getWeight(self):
        return self.weight


    def getClassification(self, givenWordIds):

        value = 0

        for (word,label) in zip(self.wordIds, self.labels):
            if word in givenWordIds:
                value = value + label

        if value<0:
            return -self.weight
        else:
            return self.weight




    def getWordIds(self):
        return self.wordIds


    def getLabels(self):
        return self.labels


