
"""
weak learner used to partially clusterize tweets
TODO : take each wordsId weight error and use it to determine the class
"""
class weakLearner:



    def __init__(self, wordId, label, weight = 1):
        self.wordId = wordId
        self.weight = weight
        self.label = label



    def setWeight(self, weight):
        self.weight = weight


    def getWeight(self):
        return self.weight


    def getWordId(self):
        return self.wordId


    def getLabel(self):
        return self.label


