from prepForQ import get_q_data
import matplotlib.pyplot as plt

data = get_q_data(ticker='SPY')
data = data.loc[data['cmpd'] > 0.6, :]

data.plot.scatter(x='price', y='cmpd')
plt.show()