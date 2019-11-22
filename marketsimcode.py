"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

Karan Achtani
kachtani3
"""

import pandas as pd
import numpy as np


def compute_portvals(ordersDF = None, prices = None, start_val=1000000):

    ordersDF = ordersDF.sort_values('Date')

    portfolioDates = prices.index.values
    portfolioColumns = ['Cash', 'Value', 'Shares']

    portfolio = pd.DataFrame(index=portfolioDates, columns=portfolioColumns)
    portfolio = portfolio.fillna(0.0)

    portfolio.loc[:,'Cash'] = start_val

    for index, order in ordersDF.iterrows():
        date = order['Date']
        shares = int(order['Shares'])
        price = prices.get_value(index=date, col='price')

        if order["Order"] == "BUY":
            portfolio.loc[date:,'Shares'] += shares
            expense = price * shares
            portfolio.loc[date:, 'Cash'] -= expense
        else:
            portfolio.loc[date:, 'Shares'] -= shares
            proceeds = price * shares
            portfolio.loc[date:, 'Cash'] += proceeds

    # multiplying the number of shares by the value of those shares then adding it to cash and putting it into value
    # the indices assume the following: share #'s are the first columns in portfolio, SPY is the first column in stockVals
    portfolio['Value'] = portfolio['Cash'] + portfolio['Shares'] * prices['price']
    portfolio['Value'].ffill()

    finalValues = portfolio['Value']

    print(portfolio)
    return finalValues


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    # print portvals
    start = portvals.index.tolist()[0]
    end = portvals.index.tolist()[-1]

    daily_returns = (portvals.values[1:] / portvals.values[:-1]) - 1.0
    cum_ret = (portvals.iloc[-1] / sv) - 1.0
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std(ddof=1)
    sharpe_ratio = (avg_daily_ret) / std_daily_ret  # sharpe ratio is returns/risk
    sharpe_ratio *= np.sqrt(252)  # annualized with sampling frequency

    # Compare portfolio against $SPX
    print( 'Date Range: {} to {}'.format(start, end))
    print()
    print( "Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print( "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print()
    print( "Cumulative Return of Fund: {}".format(cum_ret))
    print( "Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print()
    print( "Standard Deviation of Fund: {}".format(std_daily_ret))
    print( "Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print()
    print( "Average Daily Return of Fund: {}".format(avg_daily_ret))
    print( "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print()
    print( "Final Portfolio Value: {}".format(portvals[-1]))


if __name__ == "__main__":
    test_code()
