import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from file_util import get_data_from_memory, save_to_memory

DIR_NAME = 'stock_data'

def get_single_stock_data(ticker='QQQ', start_date = "2016-11-01", end_date="2019-11-18"):
    data = get_data_from_memory(DIR_NAME, ticker, start_date, end_date)

    if data is None:
        ts = TimeSeries(key='L906TTW2PFZCXCVW', output_format='pandas')
        # Get json object with the intraday data and another with  the call's metadata
        data, meta_data = ts.get_daily_adjusted(ticker, outputsize='full')
        save_to_memory(DIR_NAME, ticker, start_date, end_date, data)
    else:
        data = data.set_index('date')

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

def add_indicators(data):
    price = data[['5. adjusted close']]

    ema = pd.DataFrame(index = price.index)
    sma = pd.DataFrame(index = price.index)
    multiplier = []
    ema_0_tmp = np.zeros((len(price)))
    windows = [5,10,20]
    for w in windows:
        sma['SMA' + str(w)] = price['5. adjusted close'].rolling(w).mean()
        mult = 2 / (w + 1)
        multiplier.append(mult)
        ema['EMA' + str(w)] = ema_0_tmp
        for i in range(0, len(price)):
            # pdb.set_trace()
            if i < w:
                ema['EMA' + str(w)][i] = float('nan')
            elif i == w:
                ema['EMA' + str(w)][i] = (price['5. adjusted close'][i] - sma['SMA' + str(w)][i-1])\
                                                  * mult + sma['SMA' + str(w)][i-1]
            elif i > w:
                ema['EMA' + str(w)][i] = (price['5. adjusted close'][i] - ema['EMA' + str(w)][i-1])\
                                                  * mult + ema['EMA' + str(w)][i-1]

    data_ema = pd.concat([data,ema], axis=1)
    #print(data_ema)
    return data_ema

def clean_stock_data(data, lag=0):

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

    data = add_indicators(data)

    data = data.reset_index()

    data['date'] = [(np.datetime64(x) - np.timedelta64(lag, 'D')) for x in data['date']]
    data['date'] = data['date'].astype(str)
    print(data['date'])

    return data


get_single_stock_data(ticker='VGT')

