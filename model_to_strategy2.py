from stock_util import map_predictions_to_orders, get_single_stock_data
from marketsimcode import compute_portvals
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


def from_day(num_days):
    return (datetime(2016, 11, 9) + timedelta(days=num_days)).date()


#preds = pd.read_csv('svms/bestsvmresmeanpt7cutoff.csv')
#preds = pd.read_csv('bestnnresmeanpt5cutoff.csv')
preds = pd.read_csv('randomforests/bestrfresaggg.csv')


#print(preds)


preds['date'] = preds['day'].map(from_day)  

#preds = preds.groupby('day').sum()

preds = preds.set_index('date')
preds = preds['y_predicted']

print(preds)

date_format = '%Y-%m-%d'
prices = get_single_stock_data('SPY',start_date= datetime(2019,4,9).strftime(date_format), end_date= datetime(2019,11,12).strftime(date_format))

prices['price'] = prices['5. adjusted close']
prices = prices['price']

dates = pd.date_range(datetime(2019,4,9), datetime(2019,11,12))
df = pd.DataFrame(index=dates)
df = df.join(prices)

df = df.fillna(method='ffill').fillna(method='bfill')

joined = df.join(preds)

orders = map_predictions_to_orders(joined['y_predicted'])
print(orders)


portfolio = compute_portvals(ordersDF=orders, prices=joined['price'], start_val=10000)

print(portfolio)

final_df = pd.DataFrame({'price':joined['price'] / joined['price'].iloc[0], 'value': portfolio / portfolio.iloc[0]})
print(final_df)

plottingdf = final_df.copy()
plottingdf = plottingdf.rename(columns={"price": "SPY", "value": "Portfolio"}, errors="raise")
ax = plottingdf.plot(title="SPY Strategy vs Benchmark", fontsize=12)
ax.set_xlabel("Date")
ax.set_ylabel("Normed Value")
plt.show()

plt.clf()



ax1 = final_df['price'].plot(label='SPY Price')
ax1.set_ylabel('SPY Price')
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1)
ax2 = final_df['value'].plot(secondary_y=True, label='% Returns')
ax2.set_ylabel('% Returns')
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2, labels2)

plt.show()


