# trumpPredictor
Written by Sierra Wulfson, Colby Tobin, and Karan Achtani in CS 6220 during Fall 2018.

## Installation

## Running

## File Explanations
### q_learning/
#### q_experiment.py
q learner taught with optimal hyperparameters as found in Qtuner.py; this produced the graphs presented in workshop

#### qLearner.py
Q learning agent that uses reinforcement learning to learn based on rewards from the environemnt

#### Qlearningtrainer.py
Trains the Q learner by telling it what the rewards of certain actions are

#### qTuner.py
Gridsearch for Q learner to optimize hyper parameters


### randomforests/
#### randomforestcode.py
Runs a Random Forest and makes the bestrfresaggg.csv file used for model_to_strategy2.py

#### restrfresaggg.csv
Test predictions for the random forest strategy

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

### /

#### Bestnnresmeanpt5cutoff.csv
Test predictions from the MLP strategy

#### file_util.py
Util file to read and write from files to cache API data or computed data

#### Marketsimcode.py
code used by model_to_strategy.py and model_to_strategy2.py to implement trading portfolio

#### Mlpcode.py
code that runs a mlp and makes predictions csv for model_to_strategy2.py

#### Model2.py
tunes mlp

#### model_to_strategy.py
takes predictions and aggregates into daily predictions and implements a basic trading strategy

#### model_to_strategy2.py
takes aggregated daily predictions and implements a trading strategy

#### nn_voting.py
uses multiple lags to get predictions and has mode and mean voting to combine into predictions (not daily)

#### Prep2.py	
does data cleaning for lda and svm,rf,mlp

#### prepForLag.py 
implements lag and adds stock data after prep2.py & computetopics.ipynb are run

#### prepforq.py
preprocessing for the Q learner

#### randomForest.py 
implements and tunes a Random Forest model

#### requirements.txt 
Contains all 3rd party packages that we use. To install, open a virtual environment and run
   
    pip install -r requirements.txt

#### rf_predictions.csv
Test predictions from the Random Forest strategy

#### stock_util.py
Utility file to acquire and preprocess stock data. Also contains some other relevant functions to deal with stock data.

#### svmres.csv
Test predictions from the SVM strategy

#### topicdefs.txt
contains the words and weights for topics assigned to tweets in lagxlda.csv files

#### tweets110916_111219.csv
contains tweet info downloaded from trumptwitterarchive.com. Collected from when he got elected 11/9/16 to 11/12/19

#### tweets_sentiments2.csv
data for computetopics or prepforlag

#### tweets_sentiments4dla.csv
data needed for computetopics

#### tweets_stock_data.csv
contains tweet data joined with stock data


