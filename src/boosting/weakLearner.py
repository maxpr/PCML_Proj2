
"""
weak learner used to partially clusterize tweets
TODO : take each wordsId weight error and use it to determine the class
"""
class weakLearner:



    def __init__(self, wordIds, labels, weight = 1):
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
                    value = value + 1/self.totalPos
                else:
                    value = value - 1/self.totalNeg


        if value<0:
            return -self.weight
        elif value == 0:
            """
            print("almost total (~only empty tweet/no word in voc)")
            print(givenWordIds)
            """
            return 0
        else:
            return self.weight




    def getWordIds(self):
        return self.wordIds


    def getLabels(self):
        return self.labels


