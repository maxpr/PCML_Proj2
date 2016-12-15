
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
    def __init__(self, paths, minOccurenceThreshold=50):

        self.wordToId = {}

        wordToOcc = {}

        for pathToFile in paths:
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
        for (word, occ) in wordToOcc.items():
            if occ >= minOccurenceThreshold:
                self.wordToId[word] = freshId
                freshId = freshId + 1




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
        words = set()
        for (word, wordId) in self.wordToId.items():
            if wordId in wordIds:
                words.add(word)

        for word in words:
            del self.wordToId[word]
