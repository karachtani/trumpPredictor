from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, SCORERS
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import csv
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import sklearn.metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import warnings

#filter warnings so you can see output
#otherwise terminal gets filled with
#    UndefinedMetricWarning: F - score is ill - defined and being set to 0.0 in labels with no predicted samples.'precision', 'predicted', average, warn_for)
warnings.filterwarnings("always")

log_file_name = 'svm_results_lda2.txt'

def gs_plot(cv_results_, save_name):
  #https://www.kaggle.com/arindambanerjee/grid-search-simplified
  plt.clf()
  sns.set()
  C_list = list(cv_results_['param_C'].data)
  gamma_list = list(cv_results_['param_gamma'].data)
  plt.figure(figsize=(16,6))
  plt.subplot(1,2,1)
  data = pd.DataFrame(data={'gamma':gamma_list, 'C':C_list, 'Score':cv_results_['mean_train_score']})
  data = data.pivot(index='gamma', columns='C', values='Score')
  sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('Score for Training data')
  plt.subplot(1,2,2)
  data = pd.DataFrame(data={'gamma':gamma_list, 'C':C_list, 'Score':cv_results_['mean_test_score']})
  data = data.pivot(index='gamma', columns='C', values='Score')
  sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('Score for Test data')
  plt.savefig(save_name)
  plt.clf()

models = []

#most updated columns are
#[['date','time','retweet_count','neg', 'neu', 'pos', 'cmpd', 'IsTradingDay','is_retweet','numTweets', 'EMA5', 'EMA10', 'EMA20', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'Output']]

# filter warnings so you can see output
# otherwise terminal gets filled with
#    UndefinedMetricWarning: F - score is ill - defined and being set to 0.0 in labels with no predicted samples.'precision', 'predicted', average, warn_for)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# ignore all warnings
# def fxn():
#    warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    fxn()
filtertopics = 0
useaverage = 0 #average and filter topics or drop neutral not implemented  
useema = 1
dropneu = 0 #if using need to add split idx (length of df after dropping * .2 should be the #)
cutoffval = 0.4  # drops rows with cmpd or avgcmpd between -cuttoff and +cutoff

for lag in range(0, 8):
    with open(log_file_name, "a") as log_file:
        data = pd.read_csv("lag" + str(lag) + "lda.csv", index_col=0)


        def toint(output):
            return int(output)


        def to_day(date):
            dt = datetime.strptime(date, '%Y-%m-%d')
            return (dt - datetime(2016, 11, 9)).days


        #print(data.columns)

        data['output'] = data['Output'] \
            .map(toint)
        data['day'] = data['date'] \
            .map(to_day)
        data['is_retweet'] = data['is_retweet'] \
            .map(toint)
        #print(data.dtypes)
        #print(data.info())
        if filtertopics:
            indexNames = data[(data['Dominant_Topic'] != 2) & (data['Dominant_Topic'] != 3)].index
            data.drop(indexNames, inplace=True)
            data = data.reset_index(drop=True)
        #print(data)

        
        if useema:
            datacols = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet', 'numTweets',
                 'EMA5', 'EMA10', 'EMA20', 'Topic_Perc_Contrib','topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'output']
        else:
            datacols = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet', 'numTweets', 'Topic_Perc_Contrib','topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'output']
        data = data[datacols]
        #print(data.columns)
        

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
            #print(indexNames)
            data.drop(indexNames, inplace=True)
            #print(data)
        data = data.reset_index(drop=True)
        #print(data)

        
        print(data.columns)

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
                       'numTweets','Topic_Perc_Contrib','topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6']
        basic_xscalecols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets', 'Topic_Perc_Contrib']
        ema_xscalecols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets', 'EMA5', 'EMA10',
                          'EMA20', 'Topic_Perc_Contrib']
        ema_xcols = ['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'is_retweet',
                     'numTweets', 'EMA5', 'EMA10', 'EMA20','Topic_Perc_Contrib','topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6']
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
        print(X_train.shape)
        print(X_test.shape)
        ##############################################################
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], #[.1,.9], #[1e-5,1e-4, 1e-3, 1e-2, 0.1],# 1, 10, 100],
                          'gamma': [0.001, 0.01, 0.1, 1], #['scale',1,10,100], #'['auto','scale', 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                          'kernel': ['sigmoid']}
        print(sorted(SCORERS.keys()))

        clf = GridSearchCV(svm.SVC(probability=True),
                           param_grid=param_grid,
                           scoring='accuracy', #'roc_auc', #'f1_macro',
                           # 'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
                           cv=3,
                           refit=True,
                           verbose=10,  # 10 to see results
                           return_train_score=True,
                           # n_jobs=multiprocessing.cpu_count() - 5
                           # get error if using too much memory , njobs < cpu's availale -2
                           #n_jobs=30
                           )  # higher verbose =more printed

        clf.fit(X_train, y_train)
        models.append(clf)
        # print best parameter after tuning
        out = "==================================================="
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out =  "LAG " + str(lag)
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

        # print confusion matrix
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

        #plot gridsearch results
        #gs_plot(clf.cv_results_, 'lag'+str(lag)+'svm.png')
        #^^ this plotting function doesnt work if you're tuning > 2 things

    log_file.close()

# with open(log_file_name, "a") as lf:

#     #voting for each tweet now with each lag- not voting for daily yet
#     estimators=[('svm_0', models[0]), ('svm_1', models[1])], ('svm_2', models[2]), ('svm_3', models[3]),('svm_4', models[4])]

#     lag_ensemble_clf = VotingClassifier(estimators=estimators, voting='soft')
#     lag_ensemble_clf.fit(X_train, y_train)
#     svm_0 = models[0]
#     svm_1 = models[1]
#     svm_2 = models[2]
#     svm_3 = models[3]
#     svm_4 = models[4]
#     for clf, label in zip([svm_0, svm_1, svm_2, svm_3, svm_4,lag_ensemble_clf], ['svm_0','svm_1','svm_2','svm_3','svm_4','Ensemble']):
#         scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro') #'accuracy')

#     #get results from ensemble
#     ensemble_predictions = lag_ensemble_clf.predict(X_test)

#     # print classification report
#     out = 'Ensemble Classification Report'
#     print(out)
#     lf.write('%s\n' % out)  # save the message
#     out = classification_report(y_test, ensemble_predictions)
#     print(out)
#     lf.write('%s\n' % out)  # save the message
#     lf.close()

