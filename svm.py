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
import seaborn as sns
import matplotlib.pyplot as plt
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

with open('results_svm.txt', "a") as log_file:
    for lag in range(0, 6):
        data = pd.read_csv("lag"+str(lag)+".csv", index_col=0)
        def toint(output):
            return int(output)
        def to_day(date):
            dt = datetime.strptime(date, '%Y-%m-%d')
            return (dt - datetime(2016,11,9)).days


        data['output'] = data['Output']\
            .map(toint)
        data['day'] = data['date']\
            .map(to_day)
        data = data[['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd','IsTradingDay', 'numTweets', 'output']]

        print(data.columns)
        #'date', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd',
        #'IsTradingDay', 'numTweets', 'Output', 'output', 'day'

        data_test = data[:4654].sample(frac=1)
        data_train = data[4654:].sample(frac=1)
        #print(data_test)
        #print(data_train)

        y_train = data_train['output']
        y_test = data_test['output']

        X_train = data_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'numTweets']]
        X_test = data_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'numTweets']]
        #print(X_train)
        #print(len(data.date.unique())) 1081 dates but 1098 days

        #columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
        #scale columns that are numerical values not labeled classes
        sc = StandardScaler()
        sc.fit(X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets']])
        X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets']] = \
            sc.transform(X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets']])
        X_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets']] = \
            sc.transform(X_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets']])
        features = list(X_train.columns.values)

        param_grid = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 'scale', 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                          'kernel': ['rbf', 'sigmoid','linear']} #poly
        print(sorted(SCORERS.keys()))

        clf = GridSearchCV(svm.SVC(probability=True),
                           param_grid=param_grid,
                           scoring='f1_macro',
                           # 'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
                           cv=5,
                           refit=True,
                           verbose=0,  # 10 to see results
                           return_train_score=True,
                           n_jobs=-1
                           )  # higher verbose =more printed

        clf.fit(X_train, y_train)
        models[lag] = clf
        # print best parameter after tuning
        out = "==================================================="
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out =  "LAG " + str(lag)
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out ='best parameters found: ', clf.best_params_
        print(out)
        log_file.write('%s\n' % out)  # save the message
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
        gs_plot(clf.cv_results, 'lag'+str(lag)+'svm.png')

    #voting for each tweet now with each lag- not voting for daily yet
    estimators=[('svm_0', models[0]), ('svm_1', models[1]), ('svm_2', models[2]), ('svm_3', models[3]),
                ('svm_4', models[4])]

    lag_ensemble_clf = VotingClassifier(estimators=estimators, voting='soft')
    lag_ensemble_clf.fit(X_train, y_train)
    svm_0 = models[0]
    svm_1 = models[1]
    svm_2 = models[2]
    svm_3 = models[3]
    svm_4 = models[4]

    for clf, label in zip([svm_0, svm_1, svm_2, svm_3, svm_4, lag_ensemble_clf], ['svm_0','svm_1','svm_2','svm_3','svm_4','Ensemble']):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro') #'accuracy')

    #get results from ensemble
    ensemble_predictions = lag_ensemble_clf.predict(X_test)

    # print classification report
    out = 'Ensemble Classification Report'
    print(out)
    log_file.write('%s\n' % out)  # save the message
    out = classification_report(y_test, ensemble_predictions)
    print(out)
    log_file.write('%s\n' % out)  # save the message
    log_file.close()


