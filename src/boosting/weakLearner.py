
"""
Weak learner are classifier that performs better than random.
"""
class weakLearner:




    def __init__(self, label, weight = 1):
        self.weight = weight
        self.label = label



    """
    set the weight of the weak learner as the one given
    as parameter
    """
    def setWeight(self, weight):
        self.weight = weight

    """
    return the weight of the weak classifier
    """
    def getWeight(self):
        return self.weight


    """
    return the label of the weak classifier
    """
    def getLabel(self):
        return self.label


