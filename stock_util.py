import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from alpha_vantage.timeseries import TimeSeries
import numpy as np

def get_single_stock_data(ticker = 'SPY', start_date = '2017-01-05', end_date = '2019-01-05'):
    ts = TimeSeries(key='L906TTW2PFZCXCVW', output_format='pandas')
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_daily_adjusted(ticker, outputsize='full')
    # print(meta_data)
    # print(data)
    data = data[data.index >= start_date]
    data = data[data.index <= end_date]
    data = pd.concat([data['1. open'], data['4. close'], data['5. adjusted close']], axis=1)
    # print(data)
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

    data['Default Price Change'] = np.where((data['1. open'] <= data['4. close'])
             , 1, -1)
    data['Adjusted Price Change'] = np.where((data['1. open'] <= data['5. adjusted close'])
                                            , 1, -1)

    # print(data)
    return data



data = get_single_stock_data()
cleaned_data = clean_stock_data(data)
print(cleaned_data)

