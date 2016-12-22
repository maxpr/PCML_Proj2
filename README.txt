# Machine Learning - Project 2 tweets prediction

## Source files
- ```dataCleaning.py```: all the method to clean up the tweets set
- ```glove_routines.py```: The SGD to factorize the co-occurence matrix
- ```boosting/adaboost.py```: The main function using boosting to train and predict
- ```boosting/trainset```: contains helper for adaboost to manage the weights example in order to easily compute the error rate of any of the remaining
weak learner.
- ```boosting/vocabulary.py```: Contains the method that extract the vocabulary of the given file path as input
- ```boosting/weakLearner.py```:contains the definition and method of weak learners

- ```text_classifier.py```: helper functions to read data and submit predictions
- ```word2vec_routines.py```: helper functions to read data and submit predictions
- ```run.py```: functions to generate a prediction with the final features and script to generate the best submission (with boosting or word2vec)
- ```data/```: folder containing the source data (training and test sets) and where are stored the predictions uploaded on Kaggle


- ```sandbox_2/max.ipnyb```:  notebook for tests


#### Input
-The paths to the files essetially

#### Output
- The csv file for prediction

-- AU DESSUS C FAIT
#### Input
- ```y```: the output variable
- ```tX```: the input variable
- ```kFold```: the number of folds to compute in the cross-validation
- ```seed```: fix the seed to make the outcome predictable but still random
- ```model_fct_, param1=x, param2=y```: see section "Model Selection"

#### Output
- Displays in stdout the progression fold by fold, with their RMSE and accuracy for training 
- Returns the mean values for the RMSE and accuracy for training
- Write as .csv the predictions for the test dataset TODO FALSE


### Model Selection
When calling the cross-validation functions, the model function and its parameters must be passed as argument.

#### Model
Replace ```model_fct``` by either the following (note the trailing underscore):

- ```least_squares_```
- ```ridge_regression_```
- ```least_squares_GD_```
- ```least_squares_SGD_```
- ```logistic_regression_```
- ```reg_logistic_regression_```

#### Model's parameters
Replace ```param1=x, param2=y``` by any number of parameters as adapted to each model, amongst:

- ```lambda_```
- ```gamma```
- ```max_iters```
- ```initial_w```


#### Example
```python
kFold = 5
seed = 1
cross_validation_final_features(y, tX, kFold, seed, ridge_regression_, lambda_=0.5)
```


## How to generate final predictions
Run the script ```run.py```, which will output in file ```data/predictions.csv``` the final prediction
The best prediction is obtained by doing ./run.py word2vec

