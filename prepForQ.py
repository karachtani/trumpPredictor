import pandas as pd
import numpy as np
from stock_util import get_single_stock_data
from file_util import get_data_from_memory, save_to_memory

DIR_NAME = 'q_data'

def get_q_data(ticker='QQQ', start_date = "2016-11-01", end_date="2019-11-18"):
    stock_plus_tweet = get_data_from_memory(DIR_NAME, ticker, start_date, end_date)

    if stock_plus_tweet is None:
        stock_data = get_single_stock_data(ticker=ticker, start_date = start_date, end_date=end_date)
        stock_data['price'] = stock_data['5. adjusted close']
        stock_data = stock_data['price']
        stock_data /= stock_data[0]
        # print(stock_data)
        # pct_change from today to yesterday
        pct_change = stock_data.pct_change() * 100

        # pct_change from today to tomorrow
        pct_change[:-1] = pct_change[1:]
        pct_change = pct_change[:-1]

        stock_data = pd.DataFrame({'price': stock_data, 'pct_change': pct_change})
        stock_data.reset_index(level=0, inplace=True)
        stock_data = stock_data.rename(columns={'index':'date'})

        tweet_data = pd.read_csv("tweets_sentiments.csv")

        number_of_tweets = tweet_data.groupby('date').count()

        number_of_tweets['num_tweets'] = number_of_tweets['text']
        number_of_tweets = number_of_tweets['num_tweets']

        tweet_data = tweet_data.merge(number_of_tweets, how='left',on='date')

        stock_data['date'] = stock_data['date'].astype(str)
        stock_plus_tweet = tweet_data.merge(stock_data, how='outer', on='date')

        stock_plus_tweet = stock_plus_tweet[['retweet_count', 'cmpd', 'date', 'price', 'num_tweets']]
        stock_plus_tweet = stock_plus_tweet.sort_values(by='date')
        stock_plus_tweet = stock_plus_tweet.reset_index(drop=True)

        # set all non trading days to the next trading day
        stock_plus_tweet.loc[stock_plus_tweet['price'].isnull(), 'date'] = None
        stock_plus_tweet[['date', 'price']] = stock_plus_tweet[['date', 'price']].bfill()


        stock_plus_tweet = stock_plus_tweet.loc[stock_plus_tweet['cmpd'].notnull()]


        stock_plus_tweet = stock_plus_tweet.groupby('date').mean()
        stock_plus_tweet.reset_index(level=0, inplace=True)


        save_to_memory(DIR_NAME, ticker, start_date, end_date, stock_plus_tweet)

    return stock_plus_tweet

get_q_data()