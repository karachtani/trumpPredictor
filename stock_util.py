import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import datetime

def get_single_stock_data(ticker = 'SPY', start_date = '2017-01-05', end_date = '2019-01-05'):
    ts = TimeSeries(key='L906TTW2PFZCXCVW', output_format='pandas')
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_daily_adjusted(ticker, outputsize='full')
    # print(meta_data)
    # print(data)
    data = data[data.index >= start_date]
    data = data[data.index <= end_date]
    data = data[['1. open','4. close','5. adjusted close']]
    data = data.sort_index()
    return data

def get_multi_stock_data(tickers = [], start_date = '2017-01-05', end_date = '2019-01-05'):
    data = pd.DataFrame(data=get_single_stock_data(ticker='SPY', start_date=start_date, end_date=end_date))
    data = data.dropna()

    for ticker in tickers:
        new_stock_data = get_single_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)
        data = data.join(new_stock_data)

    data = data.fillna(method='ffill').fillna(method='bfill')

    return data

def clean_stock_data(data):

    # data['Price Change'] = 1 if data['1. open'] < data['5. adjusted close'] else -1

    # data['Default Price Change Label'] = np.where((data['4. close'].diff() >= 0)
    #          , 1, -1)
    data['Output'] = np.where((data['5. adjusted close'].diff() >= 0)
                                            , 1, -1)

    # data['Adjusted Price Change %'] = data['5. adjusted close'].pct_change() * 100
    # data['Default Price Change %'] = data['4. close'].pct_change() * 100
    #
    # data['Adjusted Price Change %'].fillna(0, inplace=True)
    # data['Default Price Change %'].fillna(0, inplace=True)


    data = data.reset_index()

    data['date'] = [x.strftime("%Y-%m-%d") for x in data['date']]
    data['date'] = data['date'].astype(str)

    return data


#
# data = get_single_stock_data()
# cleaned_data = clean_stock_data(data)
# print(cleaned_data)

