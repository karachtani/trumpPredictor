import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, SCORERS
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

with open('results_rf.txt', "a") as log_file:
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


        estimators = [10, 50, 100, 200, 500]
        max_depths = [3, 6, 10, 15, 20]

        grid_values = {'n_estimators': estimators, 'max_depth':max_depths}

        clf = GridSearchCV(RandomForestClassifier(), grid_values, scoring='roc_auc', n_jobs=-1, verbose=10, cv=3, return_train_score=True)
        clf.fit(X_train, y_train)
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
        best_n_estimators_value = clf.best_params_['n_estimators']
        best_max_depth_value = clf.best_params_['max_depth']
        best_score = clf.best_score_
        max_depth_list = list(clf.cv_results_['param_max_depth'].data)
        estimators_list = list(clf.cv_results_['param_n_estimators'].data)
        print(clf.cv_results_)

        plt.clf()
        sns.set_style("whitegrid")
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_train_score']})
        data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
        sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Training data')
        plt.subplot(1,2,2)
        data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_test_score']})
        data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
        sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Test data')
        plt.savefig('lag'+str(lag)+'gs_rf.png')
        plt.clf()
log_file.close()