import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
from scipy.stats import zscore


for lag in range(0,6):
    data = pd.read_csv("../lag" + str(lag) + ".csv", index_col=0)

    def toint(output):
        return int(output)


    def to_day(date):
        dt = datetime.strptime(date, '%Y-%m-%d')
        return (dt - datetime(2016, 11, 9)).days

    data['output'] = data['Output'] \
        .map(toint)
    data['day'] = data['date'] \
        .map(to_day)
    unaveraged_data = data[['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd','IsTradingDay', 'numTweets', 'output']]
    print(unaveraged_data.columns)

    if lag == 0:
        data['avg_RTcount'] = data.groupby('day')['retweet_count'].transform('mean')
        data['avg_neg'] = data.groupby('day')['neg'].transform('mean')
        data['avg_neu'] = data.groupby('day')['neu'].transform('mean')
        data['avg_pos'] = data.groupby('day')['pos'].transform('mean')
        data['avg_cmpd'] = data.groupby('day')['cmpd'].transform('mean')
        data['avg_time'] = data.groupby('day')['time'].transform('mean')

        daily_data = data[['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd', 'IsTradingDay', 'numTweets','output']]
        daily_data = daily_data.drop_duplicates()
        print(daily_data)

        data_thentonow = daily_data.copy()
        data_thentonow.reindex(index=data_thentonow.index[::-1])
        #colnames for full data
        # colnames = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd','IsTradingDay', 'numTweets', 'output']
        #colnames for daily data
        colnames = ['day', 'avg_time', 'avg_RTcount', 'avg_neg', 'avg_neu', 'avg_pos', 'avg_cmpd','IsTradingDay', 'numTweets', 'output']

        #'date', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd',
        #'IsTradingDay', 'numTweets', 'Output', 'output', 'day'
        #https://towardsdatascience.com/granger-causality-and-vector-auto-regressive-model-for-time-series-forecasting-3226a64889a6
        plt.clf()
        corr = unaveraged_data.corr()
        #sns.set()
        # sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, annot_kws={'size':12})
        sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, annot_kws={'size': 7}, cmap="YlGnBu").set_title('Lag '+str(lag)+' Correlation')
        heat_map = plt.gcf()
        # heat_map.set_size_inches(10,8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.savefig('lag'+str(lag)+'_corr_heatmap.png')
        plt.clf()

    for i in range(0,9):
        granger_test_result = grangercausalitytests(data_thentonow[[colnames[i],colnames[9]]], maxlag=30, verbose=False)
        optimal_lag = -1
        F_test = -1.0
        print('Granger Test Result '+str(lag)+' Lag' + colnames[i])
        for key in granger_test_result.keys():
            _F_test_ = granger_test_result[key][0]['params_ftest'][0]
            if _F_test_ > F_test:
                F_test = _F_test_
                optimal_lag = key
        print('optimal lag')
        print(optimal_lag)

# #stationarity check
# class StationarityTests:
#     def __init__(self, significance=.05):
#         self.SignificanceLevel = significance
#         self.pValue = None
#         self.isStationary = None
#
#     def ADF_Stationarity_Test(self, timeseries, printResults=True):
#         # Dickey-Fuller test:
#         adfTest = adfuller(timeseries, autolag='AIC')
#
#         self.pValue = adfTest[1]
#
#         if (self.pValue < self.SignificanceLevel):
#             self.isStationary = True
#         else:
#             self.isStationary = False
#
#         if printResults:
#             dfResults = pd.Series(adfTest[0:4],
#                                   index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])
#             # Add Critical Values
#             for key, value in adfTest[4].items():
#                 dfResults['Critical Value (%s)' % key] = value
#             print('Augmented Dickey-Fuller Test Results:')
#             print(dfResults)

    # def adfuller_test(series,signif=.05,name='',verbose=False):
    #     r = adfuller(series, autolag='AIC')
    #     output= {'test_statistic':round(r[0],4), 'pvalue':round(r[1],4), 'n_lags':round(r[2],4), 'n_obs':r[3]}
    #     p_value = output['pvalue']
    #     def adjust(val,length=6): return str(val).ljust(length)
    #     print(f'    Augmented Dickey-Fuller Test on "{name}"',"\n   ", '-'*47)
    #     print(f' Null Hypothesis: Data has unit root. Non-Stationary')
    #     print(f' Significance Level = {signif}')
    #     print(f' Test Statistic     = {output["test_statistic"]}')
    #     print(f' Num. Lags Chosen   = {output["n_lags"]}')
    #
    #     for key, val in r[4].items():
    #         print(f' Critical Value {adjust(key)} = {round(val,3)}')
    #
    #     if p_value <= signif:
    #         print(f" => P-Value = {p_value}. Rejecting Null Hypothesis")
    #         print(f" => Series is Stationary")
    #     else:
    #         print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis")
    #         print(f" => Series is Non-Stationary")
    #
    # colnames = ['day', 'time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd','IsTradingDay', 'numTweets', 'output']
    #
    # for i in range(0,10):
    #     colseries = data_thentonow.iloc[:,i]
    #     adfuller_test(colseries, name=colnames[i])
    #     print('\n')
    #
    #
