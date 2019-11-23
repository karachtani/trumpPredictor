import pandas as pd
from stock_util import get_single_stock_data, clean_stock_data

for lag in range(0,8):
    data = pd.read_csv("tweets_sentiments2.csv")
    stock_data = get_single_stock_data(start_date = "2016-10-01", end_date="2019-11-19")
    cleaned_stock_data = clean_stock_data(stock_data, lag=lag)

    #print(cleaned_stock_data)
    stock_plus_tweet = pd.merge(data, cleaned_stock_data, how='outer', on='date')

    stock_plus_tweet['IsTradingDay'] = stock_plus_tweet['Output'].isnull().map({True: 0, False: 1})
    #print(stock_plus_tweet)

    stock_plus_tweet['Output'] = stock_plus_tweet['Output'].fillna(method='backfill')
    stock_plus_tweet['EMA5'] = stock_plus_tweet['EMA5'].fillna(method='backfill')
    stock_plus_tweet['EMA10'] = stock_plus_tweet['EMA10'].fillna(method='backfill')
    stock_plus_tweet['EMA20'] = stock_plus_tweet['EMA20'].fillna(method='backfill')


    stock_plus_tweet = stock_plus_tweet[pd.notna(stock_plus_tweet['Output'])]
    stock_plus_tweet = stock_plus_tweet[pd.notna(stock_plus_tweet['text'])]

    number_of_tweets = stock_plus_tweet.groupby('date').count()

    number_of_tweets['numTweets'] = number_of_tweets['text']
    number_of_tweets = number_of_tweets['numTweets']

    stock_plus_tweet = pd.merge(stock_plus_tweet, number_of_tweets, how='left', on='date')

    stock_plus_tweet = stock_plus_tweet[['date','time','retweet_count',
                                         'neg', 'neu', 'pos', 'cmpd',
                                         'IsTradingDay','is_retweet','numTweets','EMA5', 'EMA10', 'EMA20', 'Output']]

    stock_plus_tweet.to_csv("lag" + str(lag) + "_2.csv", index=True)

    print(stock_plus_tweet)
