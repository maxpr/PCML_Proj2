
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
    def __init__(self, paths, minOccurenceThreshold=100):

        words = []

        for pathToFile in paths:
            file = open(pathToFile)
            for tweet in file:
                for word in tweet.split(' '):
                    words.append(word)
            file.close()


        words.sort()

        self.wordToId= {}
        currOcc = -1
        prevWord = None

        idWord = 0

        for currWord in words:

            if currWord != prevWord:

                if currOcc >= minOccurenceThreshold:
                    self.wordToId[prevWord] = idWord
                    idWord = idWord + 1

                currOcc = 1
                prevWord = currWord

            else:
                currOcc = currOcc + 1


        if currOcc>=minOccurenceThreshold:
            self.wordToId[prevWord] = idWord




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
