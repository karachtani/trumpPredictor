from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# CREDIT https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f

analyser = SentimentIntensityAnalyzer()


def sentence_to_sentiment(sentence):
    return analyser.polarity_scores(sentence)

def sentences_to_sentiments(sentence_df):
    sentiment_df = sentence_df.map(sentence_to_sentiment)
    sentiment_dict = sentiment_df.to_dict()
    return pd.DataFrame.from_dict(data=sentiment_dict, orient='index')

