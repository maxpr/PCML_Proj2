# Machine Learning - Project 2 - Hugo, Max & André

## Source files
- ```src/data/```: folder containing the source data (training and test sets) and where are stored the predictions uploaded on Kaggle
- ```src/plots/```: folder containing the plot used in the report
- ```src/```: folder containing the code. You can find the following in here
	- ```glove_routines.py```:  contains the methods relative to the word embeddings
	- ```text_classifier.py```:  contains the methods relative to the classifier
	- ```run.py```: functions to generate a prediction with the final features and script to generate the best submission
	- ```word2vec_routines.py```:  contains the methods relative to the word2vec
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
