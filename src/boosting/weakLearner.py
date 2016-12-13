
"""
weak learner used to partially clusterize tweets
"""
class weakLearner:



    def __init__(self, wordIds, labels, weight = 1):
        print(wordIds)
        self.wordIds = wordIds
        self.weight = weight
        self.labels = labels
        self.totalPos = len([l for l in labels if l == 1])
        self.totalNeg = len(labels)-self.totalPos



    def setWeight(self, weight):
        self.weight = weight


    def getWeight(self):
        return self.weight


    def getClassification(self, givenWordIds):

        value = 0

        for (word,label) in zip(self.wordIds, self.labels):
            if word in givenWordIds:
                if label == 1:
                    value = value + label/self.totalPos
                else:
                    value = value + label/self.totalNeg


        if value<0:
            return -self.weight
        else:
            return self.weight




    def getWordIds(self):
        return self.wordIds


    def getLabels(self):
        return self.labels


