
"""
extract all words in the files located at the path given in the init function
and filter only words that have enough occurrences to appear in the vocabulary.

(similar to cut_vocab.sh)
"""
class vocabulary:


    """
    param1 : all the paths to the files that have to be parsed
    param2 : occurrences threshold needed by word to appear in the vocabulary
    """
    def __init__(self, wordToId):

        self.wordToId = wordToId


    @staticmethod
    def createVocabulary(pathToFiles, minOccThreshold = 20):

        wordToOcc = {}

        for pathToFile in pathToFiles:
            file = open(pathToFile)
            for tweet in file:
                """  TODO : could be considered out set"""
                for word in set(tweet.strip().split(' ')):
                    if word not in wordToOcc:
                        wordToOcc[word] = 1
                    else:
                        wordToOcc[word] = wordToOcc[word] + 1

            file.close()


        freshId = 0

        wordToId = {}

        for (word, occ) in wordToOcc.items():
            if occ >= minOccThreshold:
                wordToId[word] = freshId
                freshId = freshId + 1

        return vocabulary(wordToId)




    """
    param1 : a word
    return the identifier for the word given as parameter
    """
    def getId(self, word):
        return self.wordToId[word]

    """
    param1 : a word
    return true if the vocabulary contains the word given as parameter
    """
    def has(self, word):
        return word in self.wordToId

    """
    return the size of the vocabulary
    """
    def size(self):
        return len(self.wordToId)


    """
    return the vocabulary as an iterable of words
    """
    def getKey(self):
        return self.wordToId.keys()

    """
    return the vocabulary identifier  as an iterable of integers
    """
    def getValues(self):
        return self.wordToId.values()

    """
    Remove the word from the vocabulary
    """
    def removeKeys(self, wordIds):

        """TODO"""
        wordToId = {}
        for (word, id) in self.wordToId.items():
            if id not in wordIds:
                wordToId[word] = id

        return vocabulary(wordToId)
