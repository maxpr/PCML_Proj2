import treetaggerwrapper
import re



"""
Clean the given data set.
"""
class dataCleaning:








    """
    input :
    param1 : a tweet set
    """
    def __init__(self, pathToFile = None):


        self.treeTagger = treetaggerwrapper.TreeTagger(
            TAGLANG='en',
            TAGPARFILE='../lib/english-utf8.par',
            TAGDIR='../tree-tagger')

        self.specCharSet = set("i<3>?@[.,\/#!$%\^&\*;:{}=\-_`~()]")
        self.trackedWordFunc = set(['POS','MD','NP','IN','VVD','VHZ',',','VV','NN','WP','VVN','SENT','SYM','(','MD',':','JJ','UH','CD','PP'])



        if pathToFile == None:

            self.data = None

        else:

            file = open(pathToFile)
            self.data = file.read().strip().split('\n')
            file.close()








    """
    param1 : positive tweets output file path
    param2 : negative tweets output file path

    save the mutable modifications performed in the disk
    in the location specified as parameter.
    """
    def save(self, pathToFile):

        file = open(pathToFile, 'w+')
        file.write('\n'.join(self.data))
        file.close()



    """
    param1 : percent of lines going to the pathToFile1
    param2 : pathToFile1 the path to first file where the first $percent lines will be stored
    param3 : pathToFile2 the path to second file where the last remaining lines will be stored
    """
    def cutAndSave(self, percent, pathToFile1, pathToFile2):

        file1Threshold = len(self.data)*percent

        file1 = open(pathToFile1, 'w+')
        file2 = open(pathToFile2, 'w+')


        for i in range(len(self.data)):

            if i<file1Threshold:
                file1.write(self.data[i]+"\n")
            else:
                file2.write(self.data[i]+"\n")


        file1.close()
        file2.close()



    """
    return the data extracted from the file as a string
    """
    def getData(self):
        return self.data


    """
    count the total number of small words
    """
    def avgSmallWords(self):

        w = {}
        for tweet in self.getData():
            totalSmall = 0
            for word in tweet.strip().split(' '):
                if len(word) < 4:
                    hasPunc = False
                    for char in word:
                        if char in self.specCharSet:
                            hasPunc = True
                            break
                    if not hasPunc:
                        totalSmall += 1

            if totalSmall in w:
                w[totalSmall] = w[totalSmall] +1
            else:
                w[totalSmall] = 1

        return w



    """
    Count the word function in the given tweet
    and concatenate these type of word with the occurrence
    in the tweet as new words
    """
    def setWordFunc(self, tweet):

        wordFuncToOcc = {}

        for res in self.treeTagger.tag_text(tweet):
            res = res.split('\t')
            """classic result length : word - word_function - stem"""
            if len(res) == 3:
                if res[1] in wordFuncToOcc:
                    wordFuncToOcc[res[1]] = wordFuncToOcc[res[1]] + 1
                else:
                    wordFuncToOcc[res[1]] = 1


        for (wF, occ) in wordFuncToOcc.items():

            if wF in self.trackedWordFunc:
                tweet =  tweet + " " + " WF_" + wF + str(occ)

        return tweet




    """
    Remove all duplicates tweets.
    """
    def setDuplicateLinesRemoved(self):

        assert(self.data != None)

        newData = []
        processedTweets = set()

        for tweet in self.data:
            if tweet.__hash__() not in processedTweets:
                processedTweets.add(tweet.__hash__())
                newData.append(tweet)

        self.data = newData




    """
    Remove the digit+',' before each tweet in the file test to provide in
    the kaggle interface
    """
    def setRemoveTestTweetId(self):

        assert(self.data != None)

        newData = []
        for tweet in self.data:
                newData.append(re.sub('^[0-9]+,', '', tweet))

        self.data = newData



    """
    save the data to provide to kaggle
    """
    @staticmethod
    def saveTestData(pathToFile, predictions):

        file = open(pathToFile, 'w+')

        file.write("Id,Prediction\n")

        i = 1
        for pred in predictions:
            file.write(str(i)+","+str(pred)+"\n")
            i += 1

        file.close()




    """
    All tweets are mapped in the same way applying some of the functions
    implemented in this class.
    """
    def setTweetTransform(self):

        assert (self.data != None)

        newData = []
        for tweet in self.data:
            newData.append(self.tweetTransform(tweet))

        self.data = newData



    """
    The mapping function used to transform a tweet.
    """
    def tweetTransform(self, tweet):

        """tmp = self.__tweetStemming(tweet)"""
        tmp = tweet
        tmp = self.setTweetSize(tmp)
        tmp = self.setWordFunc(tmp)
        tmp = self.setSpecCharTransform(tmp)

        return tmp



    """
    modify the punctiation in the content to a mapping char_occurrence
    """
    def setSpecCharTransform(self,tweet):

        puncCounter = {}

        for l in self.specCharSet:
            puncCounter[l] = 0


        for l in tweet.split(' '):
            if l in self.specCharSet:
                puncCounter[l] = puncCounter[l] + 1


        for l in self.specCharSet:
            puncCounter[l] = min([puncCounter[l],4])


        newTweet = ""
        for w in set(tweet.strip().split()):
            if w not in self.specCharSet:
                newTweet = newTweet + " " + w


        for (specChar, occ) in puncCounter.items():
            if occ > 0:
                newTweet = newTweet + " " + specChar + "_" + str(occ)


        return newTweet.strip()



    """
    Add the total number of word in the tweet as a new word
    """
    def setTweetSize(self, tweet):

        totalWords = 0
        for w in tweet.strip().split(' '):
            if len(w) > 3:
                totalWords = totalWords + 1


        return tweet + " twtSiz_%" +str(totalWords)




    """
    stem all the words in the string given as parameters
    """
    def tweetStemming(self, tweet):

        tokens = []
        for res in self.treeTagger.tag_text(tweet):
            res = res.split('\t')
            """classic result length : word - word_function - stem"""
            if len(res) == 3:
                tokens.append(res[2])
            elif len(res) == 1:
                tokens.append(res[0])
        return ' '.join(tokens)




    """
    Replace the letters that appear many consecutive times to only one occurrence
    with the aim to reduce the number of word variations (loool, lool, lol)
    """
    def consecutiveLetterRemoval(self, token):

        lastLetter = None
        mappedToken = ''

        for l in token:
            if l != lastLetter:
                lastLetter = l
                mappedToken = mappedToken + l

        return mappedToken




    """
    Replace all hashtags present in the given string by a constant string
    """
    def __hashTagRenaming(self, tweet):

        return re.sub("#(\w+)", 'hashtag_spec123', tweet)








