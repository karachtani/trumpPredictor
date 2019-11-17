import pandas as pd
import numpy as np

buy = "BUY"
sell = "SELL"
nothing = None

POS_IN = 1
POS_OUT = 0

def compute_performance(prices_series, orders_df):
    cash = 100
    position = POS_OUT
    shares = 0

    portfolio = pd.DataFrame(columns=['Shares', 'Cash', 'TotalValue', 'Position'], index=prices_series.index)


    for date, row in orders_df.iterrows():
        if row['Type'] == buy and position == POS_OUT:
            shares = cash / prices_series[date]
            cash = 0
            position = POS_IN

        if row['Type'] == sell and position == POS_IN:
            cash = shares * prices_series[date]
            shares = 0
            position = POS_OUT

        portfolio.loc[date, 'Cash'] = cash
        portfolio.loc[date, 'Position'] = position
        portfolio.loc[date, 'Shares'] = shares


    portfolio['Shares'] = portfolio['Shares'].ffill().fillna(value=0)
    portfolio['Cash'] = portfolio['Cash'].ffill().fillna(value=100)
    portfolio['Position'] = portfolio['Position'].ffill().fillna(value=0)

    portfolio['TotalValue'] = portfolio['Shares'] * prices_series + portfolio['Cash']
    portfolio['TotalValue'] /= portfolio.iloc[0]['TotalValue']

    return portfolio

# minimal baseline
def buy_and_hold_strategy(prices_series):
    orders_df = pd.DataFrame({'Date':[prices_series.index[0]], 'Type':[buy]})
    orders_df = orders_df.set_index('Date')

    return compute_performance(prices_series, orders_df)

# maximal baseline
def optimal_strategy(prices_series):
    orders_df = pd.DataFrame(columns=['Date','Type'])
    orders_df = orders_df.set_index('Date')
    orders_df['Type'] = prices_series.pct_change()
    orders_df['Type'][0:-1] = orders_df['Type'][1:]
    orders_df = orders_df.iloc[0:-1, :]
    orders_df['Type'] = orders_df['Type'].map(decision_based_on_difference)

    return compute_performance(prices_series, orders_df)

def labels_to_orders(prices_series, labels_series):
    orders_df = labels_series.groupby(['Date']).agg({'Output': np.sum})
    orders_df['Type'] = orders_df['Output'].map(majority_vote)

    return compute_performance(prices_series, orders_df)


def majority_vote(x):
    if x > 0:
        return buy
    elif x == 0:
        return nothing
    else:
        return sell


def decision_based_on_difference(diff):
    if diff > 0:
        return buy
    else:
        return sell

