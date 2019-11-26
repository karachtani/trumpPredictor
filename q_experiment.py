import pandas as pd
import matplotlib.pyplot as plt

from QLearningTrainer import StrategyLearner
from prepForQ import get_q_data
import datetime as dt

def scale(series):
    # return series / series.iloc[0]
    min = series.min()
    max = series.max()
    return (series - min) / (max - min)

trainer = StrategyLearner(get_q_data(ticker='QQQ'))
trainer.train(sv = 10000,
              alpha=0.8,
              gamma=0.9,
              rar=0.5,
              epochs=1000)
portfolio, ordersDF = trainer.test()
print(portfolio)
print(ordersDF)

data = get_q_data(ticker='QQQ')
data['date'] = data['date'].apply(pd.Timestamp)
data = data.set_index('date')

data = data.merge(portfolio, how='right', left_index=True, right_index=True)
data /= data.iloc[0]

data['price'] /= data['price'].iloc[0]
data['price'] -= 1
data['Value'] /= data['Value'].iloc[0]
data['Value'] -= 1
data['cmpd'] = data['cmpd'] / 125

data[['price','cmpd']].plot(linewidth=3)
data[['Value','cmpd']].plot(linewidth=3)
data[['Value','price']].plot(linewidth=3)
data[['Value','price','cmpd']].plot(linewidth=3)


buys = ordersDF.loc[ordersDF['Order'] == 'BUY', 'Date']
buys = buys.apply(pd.Timestamp)
sells = ordersDF.loc[ordersDF['Order'] == 'SELL', 'Date']
sells = sells.apply(pd.Timestamp)

for i, date in buys.items():
    plt.axvline(x=date, color='green', linewidth=0.5)

for i, date in sells.items():
    plt.axvline(x=date, color='red', linewidth=0.5)


plt.show()



