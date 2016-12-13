from src.dataCleaning import dataCleaning
from src.boosting.adaboost import adaboost
from functools import reduce
from src.boosting.vocabulary import vocabulary

if False:

    dataC = dataCleaning('../data/pos_train.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.save('../data/pos_train_clean.txt')
    print('done')


    dataC = dataCleaning('../data/neg_train.txt')

    print('remove duplicate...')
    dataC.setDuplicateLinesRemoved()
    print('set tweet transformation, can take some times...')
    dataC.setTweetTransform()
    print('save the data')
    dataC.save('../data/neg_train_clean.txt')
    print('done')

if False:

    dataC = dataCleaning('../data/neg_train_clean.txt')
    dataC.cutAndSave(0.8,'../data/neg_train_clean_cutTrain.txt','../data/neg_train_clean_cutTest.txt')
    dataC = dataCleaning('../data/pos_train_clean.txt')
    dataC.cutAndSave(0.8, '../data/pos_train_clean_cutTrain.txt', '../data/pos_train_clean_cutTest.txt')


if True:


    boosting = adaboost('../data/pos_train_clean_cutTrain.txt', '../data/neg_train_clean_cutTrain.txt')

    negTest = dataCleaning('../data/neg_train_clean_cutTest.txt').getData()
    posTest = dataCleaning('../data/pos_train_clean_cutTest.txt').getData()

    boosting.learnAndTest(posTest,negTest)




if False:

    voc = vocabulary(['../data/pos_train_clean.txt', '../data/neg_train_clean.txt'])

    print(str(voc.size()))

    voc = vocabulary(['../data/pos_train_clean_full.txt', '../data/neg_train_clean_full.txt'])

    print(str(voc.size()))


if False:

    print([1 / 5 for i in range(5)])

    array = [[1,2],[2,3]]

    array = sum(array,[])


    print(array)



    set1 = set([1,2,3,4])
    set2 = set([1,2,3,4])
    set3 = set([1,2,3,4,5])

    cont = [set1,set2,set3]
    one = reduce( (lambda x,y: x.union(y)), cont)

    print(one)







