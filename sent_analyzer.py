from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# CREDIT https://towardsdatascience.com/trump-tweets-and-trade-96ac157ef082

analyser = SentimentIntensityAnalyzer()


def sentence_to_sentiment(sentence):
    return analyser.polarity_scores(sentence)

def sentences_to_sentiments(sentence_df):
    sentiment_df = sentence_df.map(sentence_to_sentiment)
    sentiment_dict = sentiment_df.to_dict()
    return pd.DataFrame.from_dict(data=sentiment_dict, orient='index')

