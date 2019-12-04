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
import scipy

#filter warnings so you can see output
#otherwise terminal gets filled with
#    UndefinedMetricWarning: F - score is ill - defined and being set to 0.0 in labels with no predicted samples.'precision', 'predicted', average, warn_for)
warnings.filterwarnings("always")

with open('nn_voter_log_wEMA_tuned.txt', "a") as log_file:
    for lag in [2,3,4]: #range(0, 8):
        data = pd.read_csv("data_lag/lag"+str(lag)+".csv", index_col=0)

        """ DATA PREP """
        def toint(output):
            return int(output)
        def to_day(date):
            dt = datetime.strptime(date, '%Y-%m-%d')
            return (dt - datetime(2016,11,9)).days


        data['output'] = data['Output']\
            .map(toint)
        data['day'] = data['date']\
            .map(to_day)
        data = data[['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd','IsTradingDay', 'numTweets','EMA5',
                     'EMA10','EMA20', 'output']]

        print(data.columns)

        data_test = data[:4654].sample(frac=1)
        data_train = data[4654:].sample(frac=1)
        #print(data_test)
        #print(data_train)

        y_train = data_train['output']
        y_test = data_test['output']

        X_train = data_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'numTweets',
                              'EMA5', 'EMA10','EMA20']]
        X_test = data_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay', 'numTweets',
                            'EMA5', 'EMA10','EMA20']]
        #print(X_train)
        #print(len(data.date.unique())) 1081 dates but 1098 days

        #columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
        #scale columns that are numerical values not labeled classes
        sc = StandardScaler()
        sc.fit(X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets','EMA5', 'EMA10','EMA20']])
        X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets','EMA5', 'EMA10','EMA20']] = \
            sc.transform(X_train[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets','EMA5', 'EMA10','EMA20']])
        X_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets','EMA5', 'EMA10','EMA20']] = \
            sc.transform(X_test[['time', 'day', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'numTweets','EMA5', 'EMA10','EMA20']])
        features = list(X_train.columns.values)
        expected_vals = X_test[['day']]
        expected_vals['y_test'] = y_test


        """ NN """
        #insert any sklearn model here

        if lag == 1:
            clf = MLPClassifier(activation='relu', alpha=.0001, batch_size='auto', beta_1=0.9,
                                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                  hidden_layer_sizes=(10,10,10), learning_rate='constant',
                                  learning_rate_init=0.001, max_iter=5000, momentum=0.9,
                                  n_iter_no_change=10,
                                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
                                  solver='adam', tol=0.0001, validation_fraction=0, verbose=False,
                                  warm_start=False)
        elif lag == 2: #one of the best acc results
            clf = MLPClassifier(activation='logistic', alpha=0.1, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10, 4), learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0, verbose=False, warm_start=False)

        elif lag == 3: #one of the best acc results
            clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10, 10, 10), learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0, verbose=False, warm_start=False)


        elif lag == 4: #one of the best acc results
            clf = MLPClassifier(activation='logistic', alpha=0.1, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0, verbose=False, warm_start=False)
        elif lag == 5:
            clf = MLPClassifier(activation='relu', alpha=.0001, batch_size='auto', beta_1=0.9,
                                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                  hidden_layer_sizes=(10,10,10), learning_rate='constant',
                                  learning_rate_init=0.001, max_iter=5000, momentum=0.9,
                                  n_iter_no_change=10,
                                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
                                  solver='adam', tol=0.0001, validation_fraction=0, verbose=False,
                                  warm_start=False)
        else:
            clf = MLPClassifier(activation='relu', alpha=.0001, batch_size='auto', beta_1=0.9,
                                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                  hidden_layer_sizes=(10,10,10), learning_rate='constant',
                                  learning_rate_init=0.001, max_iter=5000, momentum=0.9,
                                  n_iter_no_change=10,
                                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
                                  solver='adam', tol=0.0001, validation_fraction=0, verbose=False,
                                  warm_start=False)

        # parameters = {'activation': ['relu'],
        #               'batch_size': ['auto'],
        #               #'beta_1': [0.9, 0.95],
        #               #'beta_2': [0.999, 0.99],
        #               'epsilon': [1e-08],
        #               #'learning_rate': ['constant', 'adaptive'],
        #               #'learning_rate_init': [.00001,.0001,.001],
        #               'max_iter': [5000], #[500,1000,1500],
        #               #'momentum': [.9, .95, .99],
        #               'solver': ['adam'], #['sgd', 'adam', 'lbfgs'],
        #               #'alpha': 10.0 ** -np.arange(1, 7),
        #               #'hidden_layer_sizes': [(5,2),(10,4),(2,5),(4,10),(4,4),(30,30,30),(5,5,2),(10,10,10),(100,),4,5,6,7,8,9]
        #                #'hidden_layer_sizes': np.arange(5, 10)
        #               }
        # print(sorted(SCORERS.keys()))
        # clf = GridSearchCV(MLPClassifier(random_state=1),
        #                                    param_grid=parameters,
        #                                    scoring='f1_macro', #'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
        #                                    cv=5,
        #                                    refit=True,
        #                                    verbose=0, #10 to see results
        #                                    return_train_score=True,
        #                                    n_jobs=-1
        #                    ) #higher verbose =more printed

        clf.fit(X_train, y_train)

        # print best parameter after tuning
        out = "==================================================="
        print(out)
        log_file.write('%s\n' % out)  # save the message
        out = "LAG " + str(lag)
        print(out)
        log_file.write('%s\n' % out)  # save the message
        #out ='best parameters found: ', clf.best_params_
        modelparams=clf.get_params()
        out = ('Model Parameters: ')
        print(out)
        log_file.write('%s\n' % out)  # save the message
        print(modelparams)
        log_file.write('%s\n' % modelparams)  # save the message
        # print how our model looks after hyper-parameter tuning
        #out = clf.best_estimator_
        #print(out)
        #log_file.write('%s\n' % out)  # save the message

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


        """ VOTING """
        print('---------------------')

        clf_pred= clf_pred.reshape((4654,1))
        expected_vals['y_pred'] = clf_pred
        expected_vals.sort_index(axis=0, inplace=True)

        results1 = expected_vals.copy()
        results2 = expected_vals.copy()
        results3 = expected_vals.copy()
        expected_vals['mean_pred'] = results1.groupby('day', as_index=False)[['y_pred']].transform('mean')
        expected_vals['mode_pred'] = results2.groupby('day', as_index=False)[['y_pred']].transform(lambda x:x.value_counts().index[0]) #pd.Series.mode)
        #https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
        #https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

        res = pd.DataFrame(expected_vals)[['day', 'y_test', 'mean_pred', 'mode_pred']]
        res = res.sort_values('y_test',axis=0)
        res = res.drop_duplicates(['day'],keep='first')
        res = res.set_index('day').sort_index(axis='index')
        #print('res')
        #print(res)

        #combine entries into days by mode
        mode_acc = accuracy_score(res['y_test'], res['mode_pred'])
        print('mode acc: ',mode_acc)
        out = 'Combining predictions by day with mode:'
        log_file.write('%s\n' % out)  # save the message
        cr = classification_report(res['y_test'], res['mode_pred'])
        print('Mode Classification Report: \n',cr)
        log_file.write('%s\n' % cr)  # save the message


        #combine entries into days my mean (adjust cutoff values)
        for cutoff in [0,.1,.9,.2]:
            out = ('-- Cutoff ' +str(cutoff))
            print(out)
            log_file.write('%s\n' % out)  # save the message
            res2 = res.copy()
            res2['mean_adj'] = res2['mean_pred']
            res2.loc[res2['mean_pred'] <= 0, 'mean_adj'] = -1
            res2.loc[res2['mean_pred'] > 0, 'mean_adj'] = 1

            mode_acc = accuracy_score(res2['y_test'], res2['mean_adj'])
            print('mode acc: ',mode_acc)
            out = 'Combining predictions by day with mean:'
            log_file.write('%s\n' % out)  # save the message
            cr = classification_report(res2['y_test'], res2['mean_adj'])
            print('Mean Classification Report: \n',cr)
            log_file.write('%s\n' % cr)  # save the message


log_file.close()
