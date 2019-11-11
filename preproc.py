import os
import json
import pandas as pd
import numpy as np
import csv
import re #regular expression
import string
from textblob import TextBlob
import preprocessor as p
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf
#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
#https://gauravmodi.com/02-natural_-language_processing/cleaning_text_data/
#https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


# mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    #word_tokens = word_tokenize(tweet)

    # after tweepy preprocessing the colon left remain after removing mentions
    # or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    punctuation_table = str.maketrans('', '', string.punctuation)
    lemmatizer = WordNetLemmatizer()

    # filter using NLTK library append it to a string
    word_tokens = word_tokenize(tweet)

    #filtered_tweet = [w.translate(punctuation_table) for w in word_tokens if not w in stop_words]
    filtered_tweet = []


    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:

            w = w.translate(punctuation_table)
            if w.isalpha():
                w = w.lower()
                #w = lemmatizer.lemmatize(w)
            if w != np.NaN and w != '':
                filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


#with open('trumptweets.json') as json_file:
#    data = json.load(json_file)
#df = pd.read_json(data, orient='records')

df = pd.read_csv(filepath_or_buffer='./trumpTweets/2016.txt', index_col='id_str')
#print(df)

# https://pypi.org/project/tweet-preprocessor/
clean_df = df.copy()
clean_df['text'] = clean_df['text']\
    .map(p.clean)\
    .map(clean_tweets)
mask = (clean_df['text'].str.len() >= 20)
clean_df = clean_df[mask]
print(clean_df.head(25))

"""TODO"""
#maybe extend the default stop words as in
#https://towardsdatascience.com/trump-tweets-and-trade-96ac157ef082
#words should be lemmatized -- done lemmatizing but idk if its enough or we should stem instead
#ex is 'happened' ok?
# or should we do neither or use different for LDA vs vader?
#   https://opendatagroup.github.io/data%20science/2019/03/21/preprocessing-text.html
#bigrams
#BOW dictionary
#maybe save copy with only basic tokenization steps used in article
#should check these:
#https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/
#https://gauravmodi.com/02-natural_-language_processing/cleaning_text_data/
#save copies of the different versions of the data as csv's


