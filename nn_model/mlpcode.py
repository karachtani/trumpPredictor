from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# import graphviz
# import matplotlib.pyplot as plt
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

#most updated columns are
#[['date','time','retweet_count','neg', 'neu', 'pos', 'cmpd', 'IsTradingDay','is_retweet','numTweets', 'EMA5', 'EMA10', 'EMA20', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'Output']]

# filter warnings so you can see output
# otherwise terminal gets filled with
#    UndefinedMetricWarning: F - score is ill - defined and being set to 0.0 in labels with no predicted samples.'precision', 'predicted', average, warn_for)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

filtertopics = 0
useaverage = 0 #average and filter topics or drop neutral not implemented  
useema = 1
dropneu = 0 #if using need to add split idx (length of df after dropping * .2 should be the #)
cutoffval = 0.4  # drops rows with cmpd or avgcmpd between -cuttoff and +cutoff

np.random.seed(192)
for lag in [4]: #range(0, 8):
  with open('x.txt', "a") as log_file:
      data = pd.read_csv("../data_lag/lag" + str(lag) + "lda.csv", index_col=0)


      def toint(output):
          return int(output)


      def to_day(date):
          dt = datetime.strptime(date, '%Y-%m-%d')
          return (dt - datetime(2016, 11, 9)).days


      data['output'] = data['Output'] \
          .map(toint)
      data['day'] = data['date'] \
          .map(to_day)
      data['is_retweet'] = data['is_retweet'] \
          .map(toint)
      #print(data.dtypes)
      #print(data.info())
      if filtertopics: #related to trade
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
          data_test = data[:4654]#.sample(frac=1)
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

      expected_vals = X_test[['time', 'day', 'numTweets']]
      expected_vals['y_test'] = y_test
      # print(X_train)

      # columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
      # scale columns that are numerical values not labeled classes
      sc = StandardScaler()
      sc.fit(X_train[sc_cols])
      X_train[sc_cols] = \
          sc.transform(X_train[sc_cols])
      X_test[sc_cols] = \
          sc.transform(X_test[sc_cols])

      features = list(X_train.columns.values)
      clf = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
            beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(10,10,10), learning_rate='constant',
            learning_rate_init=0.001, max_iter=10000, momentum=0.9,
            n_iter_no_change=15, nesterovs_momentum=True, power_t=0.5,
            random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
            validation_fraction=0, verbose=False, warm_start=False)

      parameters1 = {'activation': ['logistic'], #['relu', 'logistic'],
                    'batch_size': ['auto'],
                    #'beta_1': [0.9, 0.95],
                    #'beta_2': [0.999, 0.99],
                    'epsilon': [1e-08],
                    #'learning_rate': ['constant', 'adaptive'],
                    #'learning_rate_init': [.00001, .000001, .0001, .001],
                    'max_iter': [20000],  # [500,1000,1500],
                    'momentum': [.9],  # .95, .99],
                    'solver': ['lbfgs'], #['adam', 'lbfgs', 'sgd'],
                    'n_iter_no_change': [15],
                    #'alpha': [.1, .01, .0001, .00001, 10],
                    # [1e-05,1e-04,1e-03,1e-02,.1,10,100],  #10.0 ** -np.arange(1, 7),
                    'hidden_layer_sizes': [(10,10,10), (100,), (10,4), (30,30,30)]
                    #[(30, 30, 30), (5, 5, 2), (100,), (10, 10, 10), (12, 36, 2), (10, 4), (5, 2), 5, 6, 9]  # 16000 -avg
                    # [(30,30,30),(5,5,2),(100,),(5,2),(10,4),(100,),(10,10,10)] #15943 -no avg
                    # [(10,4), (100,), (10,10,10)] #15942 - noavg
                    # [(100,),4,5,6] #colab
                    # [8,9,(4,10),(4,4)] #mine
                    # [(5,2),(10,4),(2,5)] #15273
                    # [8,9,(4,10),(4,4)] #mine
                    }
      #print(sorted(SCORERS.keys()))
      #clf = GridSearchCV(MLPClassifier(random_state=1, validation_fraction=0),
                         #param_grid=parameters,
                         #scoring= 'roc_auc', #'accuracy',
                         # 'f1_weighted', #'precision_weighted',#'average_precision', #'f1_macro',
                         #cv=8,  # 5
                         #refit=True,
                         #verbose=10,  # 10 to see results
                         #return_train_score=True,
                         #n_jobs=-1
                         #)  # higher verbose =more printed

      clf.fit(X_train, y_train)

      # print best parameter after tuning
      out = "==================================================="
      print(out)
      log_file.write('%s\n' % out)  # save the message
      out = "LAG " + str(lag)
      print(out)
      log_file.write('%s\n' % out)  # save the message
      out = 'best parameters found: ', clf.get_params() #best_params_
      print(out)
      for elem in out:
          log_file.write('%s\n' % elem)  # save the message
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

  log_file.close()

model_pred = clf_pred.reshape((4654, 1))
expected_vals['y_predicted'] = model_pred
expected_vals.sort_index(axis=0, inplace=True)
print(expected_vals)
bestres = accuracy_score(y_test, model_pred)
print(bestres)

results1 = expected_vals.copy()
expected_vals['mean_pred'] = results1.groupby('day', as_index=False)[['y_predicted']].transform('mean')

# https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

res = pd.DataFrame(expected_vals)[['day', 'numTweets', 'y_test', 'mean_pred', 'mode_pred', 'sum_pred']]
res = res.sort_values('y_test', axis=0)
res = res.drop_duplicates(['day'], keep='first')
res = res.set_index('day').sort_index(axis='index')

# combine entries into days my mean (adjust cutoff values)
for cutoff in [.5]: #[0, .1, .9, .2, .3, .4, .5, .6, .7, .8]:
  out = ('-- Cutoff ' + str(cutoff))
  aggmethod = ('meancutoff' + str(cutoff))
  print(out)
  #log_file.write('%s\n' % out)  # save the message
  res2 = res.copy()
  res2['mean_adj'] = res2['mean_pred']
  res2.loc[res['mean_pred'] <= -cutoff, 'mean_adj'] = -1
  res2.loc[res['mean_pred'] > cutoff, 'mean_adj'] = 1
  indexNames = res2[(res2['mean_adj'] != 1) & (res2['mean_adj'] != -1)].index
  # print(indexNames)
  res2.drop(indexNames, inplace=True)

  mean_acc = accuracy_score(res2['y_test'], res2['mean_adj'])
  print('mean acc: ', mean_acc)
  out = 'Combining predictions by day with mean:'
  #log_file.write('%s\n' % out)  # save the message
  cr = classification_report(res2['y_test'], res2['mean_adj'])
  print('Mean Classification Report: \n', cr)
  #log_file.write('%s\n' % cr)  # save the message
  print(confusion_matrix(res2['y_test'], res2['mean_adj']))

  finalaggdf = res2[['numTweets', 'y_test']]
  finalaggdf['y_predicted'] = res2['mean_adj']
  finalaggdf.to_csv('bestnnresmeanpt5cutoff.csv')


