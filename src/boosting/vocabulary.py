
"""
extract all words in the files located at the path given in the init function
and filter only words that have enough occurrences to appear in the vocabulary.

(similar to cut_vocab.sh)
"""
class vocabulary:




    def __init__(self, wordToId, idToWord):

        self.wordToId = wordToId
        self.idToWord = idToWord




    """
    param1 : all the paths to the files that have to be parsed
    param2 : the occurrences threshold needed by words to appear in the vocabulary
    """
    @staticmethod
    def createVocabulary(datas, minOccThreshold = 20):


        wordToOcc = {}

        for data in datas:
            for tweet in data:
                """  TODO : could be considered out set"""
                for word in set(tweet.strip().split(' ')):
                    if word not in wordToOcc:
                        wordToOcc[word] = 1
                    else:
                        wordToOcc[word] = wordToOcc[word] + 1


        freshId = 0

        idToWord = {}
        wordToId = {}

        for (word, occ) in wordToOcc.items():
            if occ >= minOccThreshold:
                wordToId[word] = freshId
                idToWord[freshId] = word
                freshId = freshId + 1

        return vocabulary(wordToId, idToWord)




    """
    param1 : a word
    return the integer identifier for the word given as parameter
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
    return the word corresponding to the given id
    """
    def getWord(self, wordId):
        return self.idToWord[wordId]

