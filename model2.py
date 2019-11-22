from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#import graphviz
#import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
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
warnings.filterwarnings("always")

with open('results_nnmodel2.txt', "a") as log_file:
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

        parameters = {'activation': ['relu'],
                      'batch_size': ['auto'],
                      'beta_1': [0.9, 0.95],
                      'beta_2': [0.999, 0.99],
                      'epsilon': [1e-08],
                      'learning_rate': ['constant', 'adaptive'],
                      'learning_rate_init': [.00001,.0001,.001],
                      'max_iter': [5000], #[500,1000,1500],
                      'momentum': [.9, .95, .99],
                      'solver': ['sgd', 'adam', 'lbfgs'],
                      'solver': ['adam'],
                      'alpha': 10.0 ** -np.arange(1, 7),
                      'hidden_layer_sizes': [(5,2),(10,4),(2,5),(4,10),(4,4),(30,30,30),(5,5,2),(10,10,10),(100,),4,5,6,7,8,9]
                       #'hidden_layer_sizes': np.arange(5, 10)
                      }
        print(sorted(SCORERS.keys()))
        clf = GridSearchCV(MLPClassifier(random_state=1),
                                           param_grid=parameters,
                                           scoring='f1_macro', #'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
                                           cv=5,
                                           refit=True,
                                           verbose=0, #10 to see results
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
        log_file.write('%s\n' % out)  # save the message
        # print how our model looks after hyper-parameter tuning
        out = clf.best_estimator_
        print(out)
        log_file.write('%s\n' % out)  # save the message

        #best parameters found:  {'activation': 'relu', 'alpha': 0.1, 'batch_size': 'auto', 'beta_1': 0.95, 'beta_2': 0.99, 'epsilon': 1e-08, 'hidden_layer_sizes': 9, 'learning_rate': 'constant', 'max_iter': 500, 'momentum': 0.9, 'solver': 'adam'}
        #prints this:
        # MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.95,
         #             beta_2=0.99, early_stopping=False, epsilon=1e-08,
         #             hidden_layer_sizes=9, learning_rate='constant',
         #             learning_rate_init=0.001, max_iter=500, momentum=0.9,
         #             n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
         #             random_state=1, shuffle=True, solver='adam', tol=0.0001,
         #             validation_fraction=0.1, verbose=False, warm_start=False)
        #make sure hidden layer sizes options are correct and vary nlayers
        #vary lr
        #

        clf_pred = clf.predict(X_test)

        # print classification report
        out = 'Classification Report'
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = classification_report(y_test, clf_pred)
        print(out)
        log_file.write('%s\n' % out)  # save the message

        #Classification Report
        #              precision    recall  f1-score   support
        #
        #        -1.0       0.44      0.07      0.12       988
        #         1.0       0.58      0.94      0.72      1377

        #    accuracy                           0.57      2365
        #   macro avg       0.51      0.50      0.42      2365
        #weighted avg       0.52      0.57      0.47      2365

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
log_file.close()
