


from src.dataCleaning import dataCleaning
from src.boosting.adaboost import adaboost

######################################################
#####################################   WORKSHEET
######################################################


# 1. Open the training data :

pos = dataCleaning('../data/pos_clean_v1.txt').getData()
neg = dataCleaning('../data/neg_clean_v1.txt').getData()

tweetsToPred = dataCleaning('../data/test_clean_v1.txt').getData()

posTest = dataCleaning('../data/pos_test_clean_v1.txt').getData()
negTest = dataCleaning('../data/neg_test_clean_v1.txt').getData()



# 2. Instantiate adaboost class :

boosting = adaboost(pos,neg)



# 3. Start the algorithm :

boosting.learnAndTest('../data/pred_output_index2.txt',
                      posTest,
                      negTest,
                      tweetsToPred,
                      0.5)



