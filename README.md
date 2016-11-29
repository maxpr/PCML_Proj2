# Machine Learning - Project 2
# Hugo, Max & André

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
	- [ ] ADD HERE THE METHODS TO IMPLEMENT
  - [ ] Cross-validation
  - [ ] Bias-Variance decomposition
  - [ ] Correlation check function

  
- [ ] Be 1st on the leaderboard - imported from project 1 : still useful?
  - [ ] make use of ensembling, but last step (combine several models) [reference](http://people.inf.ethz.ch/jaggim/meetup/slides/ML-meetup-9-vonRohr-kaggle.pdf)
  - [ ] Bias-Variance analysis, to improve variance (try smaller sets of features, increase regularization), to improve bias (try getting additional features, try adding polynomial features, decrease regularization)
  - [ ] try different cost functions 
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
  - 6 basic method implementations
3. 1/3: 4 pages report
  - 1/2: correctly use, implement and describe the baseline methods
  - 1/2: scientific contribution (scientific novelty, creativity, reproducibility, solid comparison baselines to support claims, writeup quality)

## Links
- [Final submission](https://cmt3.research.microsoft.com/EPFML2016)
- [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [Feature Processing](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
- [General Tips](http://people.inf.ethz.ch/jaggim/meetup/slides/ML-meetup-9-vonRohr-kaggle.pdf)
- [General Tips 2](http://blog.kaggle.com/2014/08/01/learning-from-the-best/)
- [General Tips 3](http://blog.david-andrzejewski.com/machine-learning/practical-machine-learning-tricks-from-the-kdd-2011-best-industry-paper/)
