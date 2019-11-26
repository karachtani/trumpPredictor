import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, SCORERS
from datetime import datetime
import warnings
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
    with open('rf_wlda.txt', "a") as log_file:
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

        #######################################################
        features = list(X_train.columns.values)

        estimators = [500, 600, 1000]  # [10, 50, 100, 200, 500]
        max_depths = [20, 50, 100, 500]  # [3, 6, 10, 15, 20]

        grid_values = {'n_estimators': estimators, 'max_depth':max_depths}

        clf = GridSearchCV(RandomForestClassifier(), grid_values, scoring='roc_auc', n_jobs=-1, verbose=0, cv=3, return_train_score=True)
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

#         plt.clf()
#         sns.set_style("whitegrid")
#         plt.figure(figsize=(16,6))
#         plt.subplot(1,2,1)
#         data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_train_score']})
#         data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
#         sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Training data')
#         plt.subplot(1,2,2)
#         data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_test_score']})
#         data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
#         sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Test data')
#         plt.savefig('lag'+str(lag)+'gs_rf.png')
#         plt.clf()
log_file.close()
