from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#import graphviz
#import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, SCORERS
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import warnings

#filter warnings so you can see output
#otherwise terminal gets filled with
#    UndefinedMetricWarning: F - score is ill - defined and being set to 0.0 in labels with no predicted samples.'precision', 'predicted', average, warn_for)
import warnings

warnings.filterwarnings('ignore',
                        '.*',
                        UserWarning,
                        'warnings_filtering',
                        )
filtertopics = 0
useaverage = 0
useema = 1
dropneu = 1
cutoffval = .2 #drops rows with cmpd or avgcmpd between -cuttoff and +cutoff

with open('results_random_forest.txt', "a") as log_file:
    for lag in range(0,6):
        data = pd.read_csv("data_lag/lag" + str(lag) + "lda.csv", index_col=0)


        def toint(output):
            return int(output)


        def to_day(date):
            dt = datetime.strptime(date, '%Y-%m-%d')
            return (dt - datetime(2016, 11, 9)).days


        # print(data.columns)

        data['output'] = data['Output'] \
            .map(toint)
        data['day'] = data['date'] \
            .map(to_day)
        data['is_retweet'] = data['is_retweet'] \
            .map(toint)
        # print(data.dtypes)
        # print(data.info())
        if filtertopics:
            indexNames = data[(data['Dominant_Topic'] != 2) & (data['Dominant_Topic'] != 3)].index
            data.drop(indexNames, inplace=True)
            data = data.reset_index(drop=True)
        # print(data)

        if useema:
            datacols = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet',
                        'numTweets',
                        'EMA5', 'EMA10', 'EMA20', 'Topic_Perc_Contrib', 'topic_0', 'topic_1', 'topic_2', 'topic_3',
                        'topic_4', 'topic_5', 'topic_6', 'output']
        else:
            datacols = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet',
                        'numTweets', 'Topic_Perc_Contrib', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4',
                        'topic_5', 'topic_6', 'output']
        data = data[datacols]
        # print(data.columns)

        if useaverage == 1:
            data['avg_RTcount'] = data.groupby('day')['retweet_count'].transform('mean')
            data['avg_neg'] = data.groupby('day')['neg'].transform('mean')
            data['avg_neu'] = data.groupby('day')['neu'].transform('mean')
            data['avg_pos'] = data.groupby('day')['pos'].transform('mean')
            data['avg_cmpd'] = data.groupby('day')['cmpd'].transform('mean')
            data['avg_time'] = data.groupby('day')['time'].transform('mean')

            if useema:
                avg_ema_traincols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                     'IsTradingDay', 'is_retweet', 'numTweets', 'EMA5', 'EMA10', 'EMA20', 'output']
                avg_ema_xtraincols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                      'IsTradingDay', 'is_retweet', 'numTweets', 'EMA5', 'EMA10', 'EMA20']
                avg_ema_scalecols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                     'numTweets', 'EMA5', 'EMA10', 'EMA20']
                daily_data = data[avg_ema_traincols]
            else:
                avg_traincols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                 'IsTradingDay', 'is_retweet', 'numTweets', 'output']
                avg_xtraincols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                  'IsTradingDay', 'is_retweet', 'numTweets']
                avg_scalecols = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd',
                                 'numTweets']
                daily_data = data[avg_traincols]
            daily_data.drop_duplicates()

        if dropneu:
            if useaverage:
                indexNames = data[(data['avg_cmpd'] <= cutoffval) & (data['avg_cmpd'] >= -cutoffval)].index
            else:
                indexNames = data[(data['cmpd'] <= cutoffval) & (data['cmpd'] >= -cutoffval)].index
            # print(indexNames)
            data.drop(indexNames, inplace=True)
            # print(data)
        data = data.reset_index(drop=True)
        # print(data)

        # print(data.columns)

        if useaverage:
            data_test = data[:315].sample(frac=1)
            data_train = data[315:].sample(frac=1)
        elif filtertopics:
            data_test = data[:585].sample(frac=1)
            data_train = data[585:].sample(frac=1)
        else:
            data_test = data[:4654].sample(frac=1)
            data_train = data[4654:].sample(frac=1)
        # print(data_test)
        # print(data_train)

        y_train = data_train['output']
        y_test = data_test['output']

        basic_xcols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet',
                       'numTweets', 'Topic_Perc_Contrib', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4',
                       'topic_5', 'topic_6']
        basic_xscalecols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets',
                            'Topic_Perc_Contrib']
        ema_xscalecols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets', 'EMA5', 'EMA10',
                          'EMA20', 'Topic_Perc_Contrib']
        ema_xcols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet',
                     'numTweets', 'EMA5', 'EMA10', 'EMA20', 'Topic_Perc_Contrib', 'topic_0', 'topic_1', 'topic_2',
                     'topic_3', 'topic_4', 'topic_5', 'topic_6']
        if useaverage:
            if useema:
                XCOLS = avg_ema_xtraincols
                sc_cols = avg_ema_scalecols
            else:
                XCOLS = avg_xtraincols
                sc_cols = avg_scalecols
        elif not useaverage:
            if useema:
                XCOLS = ema_xcols
                sc_cols = ema_xscalecols
            else:
                XCOLS = basic_xcols
                sc_cols = basic_xscalecols

        X_train = data_train[XCOLS]
        X_test = data_test[XCOLS]
        # print(X_train)
        # print(len(data.date.unique())) 1081 dates but 1098 days

        # columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
        # scale columns that are numerical values not labeled classes
        sc = StandardScaler()
        sc.fit(X_train[sc_cols])
        X_train[sc_cols] = \
            sc.transform(X_train[sc_cols])
        X_test[sc_cols] = \
            sc.transform(X_test[sc_cols])

        features = list(X_train.columns.values)

        estimators = [10, 25, 50, 75, 100, 150, 200, 250, 500]
        max_depths = [3, 6, 10, 15, 20]

        parameters = {"n_estimators": estimators,
                      "criterion": ["gini", "entropy"],
                      "max_depth": max_depths,
                      }
        print(sorted(SCORERS.keys()))
        clf = GridSearchCV(RandomForestClassifier(random_state=1),
                                           param_grid=parameters,
                                           scoring='accuracy', #'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
                                           cv=3, #5
                                           refit=True,
                                           verbose=1, #10 to see results
                                           return_train_score=True,
                                           n_jobs=-1
                           ) #higher verbose =more printed

        clf.fit(X_train, y_train)

        # print best parameter after tuning
        out = "==================================================="
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = "LAG " + str(lag)
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out ='best parameters found: ', clf.best_params_
        print(out)
        for elem in out:
            log_file.write('%s\n' % elem)  # save the message
        # print how our model looks after hyper-parameter tuning
        out = clf.best_estimator_
        print(out)
        log_file.write('%s\n' % out)  # save the message

        clf_pred = clf.predict(X_test)

        # print classification report
        out = 'Classification Report'
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = classification_report(y_test, clf_pred)
        print(out)
        log_file.write('%s\n' % out)  # save the message

        out = 'Confusion Matrix'
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = confusion_matrix(y_test, clf_pred)
        print(out)
        log_file.write('%s\n' % out)  # save the message

        out = "Accuracy:"
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = accuracy_score(y_test, clf_pred)
        print(out)
        log_file.write('%s\n' % out)  # save the message
log_file.close()
