import pandas as pd
from stock_util import get_single_stock_data, clean_stock_data
from sklearn.preprocessing import OneHotEncoder

#data = pd.read_csv("tweets_sentiments2.csv")
data = pd.read_csv("stats/tweets_sents_lda.csv", index_col=0)
print(data.columns)
print(data)
test = data[['Dominant_Topic']]
topicohe = OneHotEncoder()
X = topicohe.fit_transform(test.Dominant_Topic.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(X, columns=["topic_" + str(int(i)) for i in range(X.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
print(data)

for lag in range(0,8):
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
                                         'IsTradingDay','is_retweet','numTweets',
                                         'EMA5', 'EMA10', 'EMA20',
                                         'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
                                         'topic_0', 'topic_1', 'topic_2', 'topic_3',
                                         'topic_4', 'topic_5', 'topic_6',
                                         'Output']]

    stock_plus_tweet.to_csv("lag" + str(lag) + "lda.csv", index=True)

    print(stock_plus_tweet)
