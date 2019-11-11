from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# CREDIT https://towardsdatascience.com/trump-tweets-and-trade-96ac157ef082

analyser = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

sentiment_analyzer_scores("russians playing fools funny watch nt clue tot")


