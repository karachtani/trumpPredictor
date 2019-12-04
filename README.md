# trumpPredictor
Written by Sierra Wulfson, Colby Tobin, and Karan Achtani in CS 6220 during Fall 2019.

## Installation

#### Create a new Virtual Environment: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
1. python3 -m venv ./venv
2. source venv/bin/activate
3. pip3 install -r requirements.txt

## Running

#### Initialize Virtual Environment
1. source venv/bin/activate
2. pip3 install -r requirements.txt

#### Run Code:
1. Run from within trumpPredictor directory
2. Make sure virtual environment "venv" is active
3. Run .py files with as "python *.py" for any file *

## File Explanations
### nn_model/

#### mlpcode.py
code that runs a tuned MLP and makes predictions csv for model_to_strategy2.py

#### model.py
tunes mlp

#### nn_voting.py
uses multiple lags to get predictions and has mode and mean voting to combine into predictions (not daily)

### nn_results/
Various results files from various experiments with the Neural Net model

### q_data/
stock and twitter data that has been preprocessed for q learning is cached here to avoid recomputation 
 
### q_learning/
#### q_experiment.py
q learner taught with optimal hyperparameters as found in Qtuner.py; this produced the graphs presented in workshop

#### qLearner.py
Q learning agent that uses reinforcement learning to learn based on rewards from the environemnt

#### Qlearningtrainer.py
Trains the Q learner by telling it what the rewards of certain actions are

#### qTuner.py
Gridsearch for Q learner to optimize hyper parameters

#### results_q_tuning.py
The results of the gridsearch in qTuner.py

### randomforests/
#### rf.py 
implements and tunes a Random Forest model

#### randomforestcode.py
Runs a tuned Random Forest and makes the bestrfresaggg.csv file used for model_to_strategy2.py

#### restrfresaggg.csv
Test predictions for the random forest strategy

#### rf_predictions.csv
Test predictions from the Random Forest strategy

### stats/
#### computetopics.ipynb
Performs topic modeling.
Basically reads in tweets_sentiments2.csv or tweets_sentiments4dla.csv and does topic modeling w mallet’s LDA and generates tweets_sents_lda.csv used in prepforlag.py

#### lagX_corr_heatmap.png
heat maps for correlation for original data files 

#### mallet-2.0.8.zip
must be unzipped and placed in same directory as computetopics.ipynb. It is the LDA model used by gensim’s wrapper to implement Mallet’s LDA. Path to mallet may need to be updated in computetopics.ipynb depending of directory structure / naming

#### stats.py
code for exploring correlations in data

#### topicmodeling_testvis.ipynb
code for playing around with LDA, topics, visualizations of topics 

#### tweets_sents_lda.csv
csv used in prepforlag.py

### stock_data/
Folder that contains csv of stock data grabbed from alphavantage to avoid using API on every time code is run

### Svms/
#### bestsvmmeanpt7cutoff.csv
Test predictions for the SVM strategy

#### lagXsvm.png files 
gridsearchcv tuning heat maps that can be generated from svmcode.py. These are from original data files

#### svmcode.py
runs a svm and makes predictions csv for model_to_strategy2.py

#### svmres.csv
Test predictions from the SVM strategy

### /

#### file_util.py
Util file to read and write from files to cache API data or computed data

#### Marketsimcode.py
code used by model_to_strategy.py and model_to_strategy2.py to implement trading portfolio

#### model_to_strategy.py
takes predictions and aggregates into daily predictions and implements a basic trading strategy

#### model_to_strategy2.py
takes aggregated daily predictions and implements a trading strategy

#### Prep2.py	
does data cleaning for lda and svm,rf,mlp

#### prepForLag.py 
implements lag and adds stock data after prep2.py & computetopics.ipynb are run

#### prepforq.py
preprocessing for the Q learner

#### requirements.txt 
Contains all 3rd party packages that we use. To install, open a virtual environment and run
   
    pip install -r requirements.txt

#### stock_util.py
Utility file to acquire and preprocess stock data. Also contains some other relevant functions to deal with stock data.

#### topicdefs.txt
contains the words and weights for topics assigned to tweets in lagxlda.csv files

#### tweets110916_111219.csv
contains tweet info downloaded from trumptwitterarchive.com. Collected from when he got elected 11/9/16 to 11/12/19

#### tweets_sentiments.csv
data for computetopics or prepforlag

#### tweets_sentiments4dla.csv
data needed for computetopics

#### tweets_stock_data.csv
contains tweet data joined with stock data


