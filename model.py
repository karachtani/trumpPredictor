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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import itertools


for lag in range(1, 6):
    data = pd.read_csv("lag"+str(lag)+".csv", index_col=0)
    def toint(output):
        return int(output)
    print(data.columns)
    data['Ouput'] = data['Output']\
        .map(toint)

    #columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
    #scale columns that are numerical values not labeled classes
    #i dont think we have to scale sent vals because they should be -1 to 1
    sc = StandardScaler()
    data[['time', 'retweet_count', 'numTweets']] = sc.fit_transform(data[['time', 'retweet_count', 'numTweets']])

    y = data['Output']
    X = data[['time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    features = list(X_train.columns.values)

    parameters = {'activation': ['relu'],
                  'batch_size': ['auto'],
                  'beta_1': [0.9, 0.95],
                  'beta_2': [0.999, 0.99],
                  'epsilon': [1e-08],
                  'learning_rate': ['constant'],
                  #'learning_rate_init': [np.logspace(-5, 3, 5)],
                  'max_iter': [500,1000,1500],
                  'momentum': [.9, .95, .99],
                  'solver': ['sgd', 'adam', 'lbfgs'],
                  'alpha': 10.0 ** -np.arange(1, 7),
                  'hidden_layer_sizes': np.arange(5, 12)
                  }

    clf = GridSearchCV(MLPClassifier(random_state=1),
                                       param_grid=parameters,
                                       cv=5,
                                       refit=True,
                                       verbose=1,
                                       return_train_score=True,
                                       n_jobs=-1
                       ) #higher verbose =more printed

    clf.fit(X_train, y_train)

    # print best parameter after tuning
    print("===================================================")
    print("LAG " + str(lag))
    print('best parameters found: ', clf.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(clf.best_estimator_)
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
    print('Classification Report')
    print(classification_report(y_test, clf_pred))

    #Classification Report
    #              precision    recall  f1-score   support
    #
    #        -1.0       0.44      0.07      0.12       988
    #         1.0       0.58      0.94      0.72      1377

    #    accuracy                           0.57      2365
    #   macro avg       0.51      0.50      0.42      2365
    #weighted avg       0.52      0.57      0.47      2365

#repeat for lags 1-5
#further tune

