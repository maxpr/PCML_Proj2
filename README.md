# Machine Learning - Project 2 - Hugo, Max & André

## Source files
- ```src/data/```: folder containing the source data (training and test sets) and where are stored the predictions uploaded on Kaggle
- ```src/plots/```: folder containing the plot used in the report
- ```src/```: folder containing the code. You can find the following in here
	- ```sandbox.ipnyb```:  notebook for tests

#### Input - IMPORTED FROM Project 1 TO CHANGE
- ```y```: the output variable
- ```tX```: the input variable
- ```kFold```: the number of folds to compute in the cross-validation
- ```seed```: fix the seed to make the outcome predictable but still random
- ```bin_output```: boolean to indicate wether or not the output y is binary, if True will calculate the accuracy alongside the RMSE
- ```model_fct_, param1=x, param2=y```: see section "Model Selection"

#### Output - IMPORTED FROM Project 1 TO CHANGE
- Displays in stdout the progression fold by fold, with their RMSE (and accuracy) for training and test
- Returns the mean values for the RMSE (and accuracy) for training and test


## How to generate final predictions
Run the script ```run.py```, which will output in file ```data/final-predictions.csv``` the final prediction

## TODO

- [ ] Code
  - [ ] xx baseline methods implemented and tested
	- [ ] SGD updates to train the matrix factorization
	- [ ] Construct feature representation of each training tweet
	- [ ] Linear Classifier Trainer (logistic regression or SVM, scikit lib)
	- [ ] Predicter : predict labels for all tweets in the test set
  - [ ] Cross-validation
  - [ ] Bias-Variance decomposition
  - [ ] Correlation check function

  
- [ ] Be 1st on the leaderboard - imported from project 1 : still useful?
  - [ ] make use of ensembling, but last step (combine several models) [reference](http://people.inf.ethz.ch/jaggim/meetup/slides/ML-meetup-9-vonRohr-kaggle.pdf)
  - [ ] Bias-Variance analysis, to improve variance (try smaller sets of features, increase regularization), to improve bias (try getting additional features, try adding polynomial features, decrease regularization)
  - [ ] try different cost functions 
  - [ ] try other models to improve accuracy and computation speed
	- [ ] Determine what we want to try
  - [ ] Feature Processing
    - [ ] take care of outliers 
    - [ ] scaling / standardizing / normalizing
    - [ ] check for duplicate rows
    - [ ] try applying transformations and check evolution on correlation and/or result of a simple model [reference](http://datascience.stackexchange.com/questions/10640/how-to-perform-feature-engineering-on-unknown-features)
    - [ ] feature transformation (sqrt, log, exp, tanh, abs)
    - [ ] polynomials
    - [ ] feature combination (a\*b, a-b, a+b, a/b)
    - [ ] try discretizing some features (if it makes sense)
    - [ ] try to add binary feature for describing missing values (e.g. : has_PRI_jet_subleading)
    - [ ] try one-hot encoding (mainly for the category variable) [reference](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
    - [ ] try feature deletions

- [ ] Report - imported from project 1 : still useful?
  - [ ] Abstract and introduction
  - [ ] Exploratory data analysis
    - [ ] Data description
    - [ ] Data visualization
    - [ ] Data cleaning
  - [ ] Feature Processing (description of final features used and why)
  - [ ] baseline methods description and comparison (parameters, test error)
    - [ ] over/under fit (bias-variance decomposition)
    - [ ] visualizations
    - [ ] cross validation (test error estimation)
    - [ ] cost function used
  - [ ] Describe the additional steps to enhance baseline results 
    - [ ] Cross-Validation and Bias-Variance decomposition for the final model(s) retained
  - [ ] Conclusion



## Grading
1. 1/3: final score on leaderboard translated to grade from 4 to 6
2. 1/3: executable and documented code, rules:
  - Reproducibility (file run.py producing exactly same .csv predictions of the best submission in Kaggle)
  - Documentation (in report and in code, README and includes data preparation, feature generation and cross-validation steps)
  - basic method implementations
3. 1/3: 4 pages report
  - 1/2: correctly use, implement and describe the baseline methods
  - 1/2: scientific contribution (scientific novelty, creativity, reproducibility, solid comparison baselines to support claims, writeup quality)

## Links
- [Submission on kaggle](https://inclass.kaggle.com/c/epfml-text)
- [Final submission](https://cmt3.research.microsoft.com/EPFML2016)
- [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [Feature Processing](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
- [General Tips](http://people.inf.ethz.ch/jaggim/meetup/slides/ML-meetup-9-vonRohr-kaggle.pdf)
- [General Tips 2](http://blog.kaggle.com/2014/08/01/learning-from-the-best/)
- [General Tips 3](http://blog.david-andrzejewski.com/machine-learning/practical-machine-learning-tricks-from-the-kdd-2011-best-industry-paper/)


# Project Text Sentiment Classification

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

As a baseline, we here provide sample code using word embeddings to build a text classifier system.

Submission system environment setup:

1. The text datasets are available from the course kaggle page here:

 [https://inclass.kaggle.com/c/epfml-text]

 Download the provided datasets `twitter-datasets.zip`.

2. To submit your solution to the online evaluation system, we require you to prepare a “.csv” file of the same structure as sampleSubmission.csv (the order of the predictions does not matter, but make sure the tweet ids and predictions match). Your submission is evaluated according to the classification error (number of misclassified tweets) of your predictions.

*Working with Twitter data:* We provide a large set of training tweets, one tweet per line. All tweets in the file train pos.txt (and the train pos full.txt counterpart) used to have positive smileys, those of train neg.txt used to have a negative smiley. Additionally, the file test data.txt contains 10’000 tweets without any labels, each line numbered by the tweet-id.

Your task is to predict the labels of these tweets, and upload the predictions to kaggle. Your submission file for the 10’000 tweets must be of the form `<tweet-id>`, `<prediction>`, see `sampleSubmission.csv`.

Note that all tweets have already been pre-processed so that all words (tokens) are separated by a single whitespace. Also, the smileys (labels) have been removed.

## Classification using Word-Vectors

For building a good text classifier, it is crucial to find a good feature representation of the input text. Here we will start by using the word vectors (word embeddings) of each word in the given tweet. For simplicity of a first baseline, we will construct the feature representation of the entire text by simply averaging the word vectors.

Below is a solution pipeline with an evaluation step:

### Generating Word Embeddings: 

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```

Now given the co-occurrence matrix and the vocabulary, it is not hard to train GloVe word embeddings, that is to compute an embedding vector for wach word in the vocabulary. We suggest to implement SGD updates to train the matrix factorization, as in

`glove_solution.py`

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets `pos_train_full.txt`, `neg_train_full.txt`

### Building a Text Classifier:
1. Construct Features for the Training Texts: Load the training tweets and the built GloVe word embeddings. Using the word embeddings, construct a feature representation of each training tweet (by averaging the word vectors over all words of the tweet).

2. Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed features, using the scikit learn library, or your own code from the earlier labs. Recall that the labels indicate if a tweet used to contain a :) or :( smiley.

3. Prediction: Predict labels for all tweets in the test set.

4. Submission / Evaluation: Submit your predictions to kaggle, and verify the obtained misclassification error score. (You can also use a local separate validation set to get faster feedback on the accuracy of your system). Try to tune your system for best evaluation score.

## Extensions:
Naturally, there are many ways to improve your solution, both in terms of accuracy and computation speed. More advanced techniques can be found in the recent literature.

