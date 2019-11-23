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
    #tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r"http\S+", "", tweet)
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
def gettime(str):
    return str[11:]
    
clean_df['date'] = clean_df['created_at']\
    .map(getdate).astype(str)

clean_df['time'] = pd.to_timedelta(clean_df['created_at']\
    .map(gettime)) / np.timedelta64(1, 'h')

# clean_df['avg_RTcount'] = clean_df.groupby('date')['retweet_count'].transform('mean')
# clean_df['avg_neg'] = clean_df.groupby('date')['neg'].transform('mean')
# clean_df['avg_neu'] = clean_df.groupby('date')['neu'].transform('mean')
# clean_df['avg_pos'] = clean_df.groupby('date')['pos'].transform('mean')
# clean_df['avg_cmpd'] = clean_df.groupby('date')['cmpd'].transform('mean')


cdf= clean_df.copy()
cdf.loc[clean_df['is_retweet'] == True, 'is_retweet'] = 1
cdf.loc[clean_df['is_retweet'] == False, 'is_retweet'] = 0
#58 non retweets are marked as null
cdf.loc[ (clean_df['is_retweet'] != True) & (clean_df['is_retweet'] != False), 'is_retweet'] = 0

print(cdf[cdf['is_retweet'].isnull()])

print(cdf.dtypes)
print(cdf.info())
print(cdf.isnull().sum())
print(cdf)
print('----------------')

cdf.to_csv("tweets_sentiments2.csv", index=True)

from stock_util import get_single_stock_data, clean_stock_data

stock_data = get_single_stock_data(start_date = "2016-11-09", end_date="2019-11-12")
cleaned_stock_data = clean_stock_data(stock_data)

stock_plus_tweet = pd.merge(cdf, cleaned_stock_data, how='outer', on='date')


stock_plus_tweet['IsTradingDay'] = stock_plus_tweet['Output'].isnull().map({True: 0, False: 1})

stock_plus_tweet['Output'] = stock_plus_tweet['Output'].fillna(method='backfill')

stock_plus_tweet = stock_plus_tweet[pd.notna(stock_plus_tweet['Output'])]
stock_plus_tweet = stock_plus_tweet[pd.notna(stock_plus_tweet['text'])]

number_of_tweets = stock_plus_tweet.groupby('date').count()

number_of_tweets['numTweets'] = number_of_tweets['text']
number_of_tweets = number_of_tweets['numTweets']

stock_plus_tweet = pd.merge(stock_plus_tweet, number_of_tweets, how='left', on='date')

stock_plus_tweet = stock_plus_tweet[['date','time','retweet_count',
                                     'neg', 'neu', 'pos', 'cmpd',
                                     'IsTradingDay','is_retweet','numTweets','Output']]


# print(stock_plus_tweet)
stock_plus_tweet.to_csv("tweets_stock_data2.csv", index=True)


"""TODO"""
#combine with stock data
#might want to lemmatize, remove stop words, usernames, punctuation, etc for LDA
#might want to extend stop words for LDA
#granger causality to find lag to be used in classification NN



