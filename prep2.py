import os
import json
import pandas as pd
import numpy as np
import csv
import datetime
import re #regular expression
import string
#from textblob import TextBlob
#import preprocessor as p
import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



#bc of how vader works i think less cleaning is better
#can try variations later and see how it changes accuracy
#might want different cleaning for LDA etc
#might want to remove stopwords or might want to wait till LDA
def clean(tweet):
    #remove RT @username:
    tweet = re.sub(r'RT\s@.+:\s', '', tweet)
    #add whitespace before urls incase he didnt
    tweet = re.sub(r'http', ' http', tweet)
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    #remove Trump's ..... at begeinning of tweets
    tweet = re.sub(r'^\.+', '', tweet)
    #remove @ to change usernames into names
    tweet = re.sub(r'@realDonaldTrump', 'Trump', tweet)
    tweet = re.sub(r'@', '', tweet)
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # remove hastags
    tweet = re.sub(r'#', '', tweet)
    return tweet

def sentiment_analyzer_scores_neg(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score['neg']

def sentiment_analyzer_scores_neu(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score['neu']

def sentiment_analyzer_scores_pos(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score['pos']

def sentiment_analyzer_scores_cmp(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score['compound']


df = pd.read_csv(filepath_or_buffer='tweets110916_111219.csv',index_col='id_str')

clean_df = df.copy()

clean_df['text'] = df['text']\
    .map(clean)
mask = (clean_df['text'].str.len() >= 20)
clean_df = clean_df[mask]
#print(df['text'].head(50))
#print(clean_df['text'].head(50))

analyser = SentimentIntensityAnalyzer()

clean_df['neg'] = clean_df['text']\
    .map(sentiment_analyzer_scores_neg)
clean_df['neu'] = clean_df['text']\
    .map(sentiment_analyzer_scores_neu)
clean_df['pos'] = clean_df['text']\
    .map(sentiment_analyzer_scores_pos)
clean_df['cmpd'] = clean_df['text']\
    .map(sentiment_analyzer_scores_cmp)
#print(clean_df)

def getdate(str):
    return datetime.datetime.strptime(str[:10], '%m-%d-%Y').strftime('%Y-%m-%d')
    
clean_df['date'] = clean_df['created_at']\
    .map(getdate).astype(str)
    
clean_df['avg_RTcount'] = clean_df.groupby('date')['retweet_count'].transform('mean')
clean_df['avg_neg'] = clean_df.groupby('date')['neg'].transform('mean')
clean_df['avg_neu'] = clean_df.groupby('date')['neu'].transform('mean')
clean_df['avg_pos'] = clean_df.groupby('date')['pos'].transform('mean')
clean_df['avg_cmpd'] = clean_df.groupby('date')['cmpd'].transform('mean')



clean_df.to_csv("tweets_sentiments.csv", index=True)

from stock_util import get_single_stock_data, clean_stock_data

stock_data = get_single_stock_data(start_date = "2016-01-05")
cleaned_stock_data = clean_stock_data(stock_data)

stock_plus_tweet = pd.merge(clean_df, cleaned_stock_data, on='date')

stock_plus_tweet.to_csv("tweets_stock_data.csv", index=True)


"""TODO"""
#combine with stock data
#might want to lemmatize, remove stop words, usernames, punctuation, etc for LDA
#might want to extend stop words for LDA
#granger causality to find lag to be used in classification NN



