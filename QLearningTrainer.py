"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Code written by Karan Achtani in CS 4646 (ML4T)
and modified for the purposes of this project.
"""

import datetime as dt
import pandas as pd
import numpy as np
import QLearner as ql
import marketsimcode as mktsim


class StrategyLearner(object):

    # constructor
    def __init__(self, data, verbose = False):
        self.verbose = verbose
        self.learner = None
        self.numBins = 4
        self.train_ratio = 0.8

        data['cmpd'] = self.discretize(data['cmpd'])
        data['num_tweets'] = self.discretize(data['num_tweets'])
        data['retweet_count'] = self.discretize(data['retweet_count'])

        data['state'] = self.statify(indicators=[data['cmpd'], data['num_tweets'], data['retweet_count']])

        num_data_points = data.shape[0]
        test_cutoff = int(num_data_points * self.train_ratio)
        self.train_data = data.iloc[:test_cutoff]
        self.test_data = data.iloc[test_cutoff + 1:]

    # this method should create a QLearner, and train it for trading
    def train(self,
              sv = 100000,
              alpha=0.2,
              gamma=0.9,
              rar=0.8,
              epochs=100):

        # get the data and discretize them into buckets
        data = self.train_data

        self.learner = ql.QLearner(num_states=int(pow(self.numBins, 3)),
                                   num_actions=3,
                                   alpha=alpha,
                                   gamma=gamma,
                                   rar=rar)

        portValues = []

        while len(portValues) < epochs:

            # set up orders array
            orders_np = np.empty([0, 3])

            curPortState = 0
            cash = sv
            portValue = cash

            action = self.learner.querysetstate(s=data['state'].iloc[0])

            for index, row in data.iterrows():
                date = str_to_date(row['date'])
                order = self.actionToOrder(action, curPortState)
                if order is not None:
                    order_type = order[0]
                    num_shares = order[1]
                    curPortState = order[2]

                    orders_np = np.append(orders_np, [[str(date), order_type, num_shares]], axis=0)

                    if order_type == "BUY":
                        cash -= num_shares * row['price']
                    else:
                        cash += num_shares * row['price']

                newPortValue = cash + curPortState * row['price'] * 1000

                action = self.learner.query(s_prime=int(row["state"]), r=(newPortValue - portValue))
                portValue = newPortValue
                # print portValue

            portValues.append(portValue)

        return pd.DataFrame(data=portValues)

            # this method should use the existing policy and test it against new data

    def test(self, symbol="IBM", \
             sd=dt.datetime(2010, 1, 1), \
             ed=dt.datetime(2011, 12, 31), \
             sv=100000):

        data = self.test_data

        # set up orders array
        ordersNP = np.empty([0, 3])

        curPortState = 0

        self.learner.rar = 0
        action = self.learner.querysetstate(s=data["state"].iloc[0])

        for index, row in data.iterrows():
            date = str_to_date(row['date'])
            order = self.actionToOrder(action, curPortState)
            if order is not None:
                order_type = order[0]
                num_shares = order[1]
                curPortState = order[2]

                ordersNP = np.append(ordersNP, [[str(date), order_type, num_shares]], axis=0)

            action = self.learner.querysetstate(s=row["state"])

        ordersDF = pd.DataFrame(columns=['Date', 'Order', 'Shares'], data=ordersNP)

        prices_df = data[['price','date']]
        prices_df = prices_df.set_index('date')
        portfolio =  mktsim.compute_portvals(ordersDF=ordersDF, prices=prices_df, start_val=sv)

        return portfolio, ordersDF

    # returns (order type, shares, new port state)
    def actionToOrder(self, action, curPortState):
        # action mapping
        GO_LONG = 0
        CLOSE_OUT = 1
        GO_SHORT = 2
        LONG = 1
        SHORT = -1


        if (action == GO_LONG and curPortState == LONG) or\
                (action == GO_SHORT and curPortState == SHORT) or\
                (action == CLOSE_OUT and curPortState == 0):
            return None
        else:
            if (action == GO_LONG and curPortState == SHORT):
                return ("BUY", 2000, LONG)
            elif (action == GO_LONG and curPortState == 0):
                return ("BUY", 1000, LONG)
            elif (action == GO_SHORT and curPortState == LONG):
                return ("SELL", 2000, SHORT)
            elif (action == GO_SHORT and curPortState == 0):
                return ("SELL", 1000, SHORT)
            elif (action == CLOSE_OUT and curPortState == LONG):
                return ("SELL", 1000, 0)
            elif (action == CLOSE_OUT and curPortState == SHORT):
                return ("BUY", 1000, 0)

    def minifyOrder(self, order):
        if order is None:
            return 0

        multiplier = 0
        if order[0] == "BUY":
            multiplier = 1
        elif order[0] == "SELL":
            multiplier = -1

        return multiplier * order[1]


    def discretize(self, indicator):
        return pd.cut(x=indicator, bins=self.numBins, labels=False)


    def statify(self, indicators=[]):
        if len(indicators) == 0:
            return None

        state = pd.Series(index=indicators[0].index)
        state[:] = 0

        for i in range(len(indicators)):
            state += indicators[i] * pow(self.numBins, i)


        state = state.map(lambda s: int(s))
        return state

def str_to_date(string):
    year, month, day = string.split("-")
    return dt.date(month=int(month), day=int(day), year=int(year))


