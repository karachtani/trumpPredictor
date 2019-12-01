from stock_util import map_predictions_to_orders, get_single_stock_data
from marketsimcode import compute_portvals
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


def from_day(num_days):
    return (datetime(2016, 11, 9) + timedelta(days=num_days)).date()

def join_preds_prices(preds, prices):
    # map datetimes to strings
    preds.index = preds.index.map(lambda d: str(d))
    prices.index = prices.index.map(lambda d: str(d))

    # join both series into a DF
    joined = pd.DataFrame({'pred': preds, 'price': prices})

    # create date column with index
    joined['date'] = joined.index
    joined = joined.sort_index()

    # remove all dates without a price
    joined.loc[joined['price'].isnull(), 'date'] = None

    # backfill next trading day onto nontrading days
    joined['date'] = joined['date'].bfill()

    # sum predictions and take the majority vote
    joined = joined.groupby('date').sum()
    joined['pred'] = joined['pred'].map(lambda pred: 1 if pred > 0 else -1)

    return joined

preds = pd.read_csv('bestnnres.csv')

preds['day'] = preds['day'].map(from_day)

preds = preds.groupby('day').sum()

preds = preds['y_predicted']
orders = map_predictions_to_orders(preds)

date_format = '%Y-%m-%d'
prices = get_single_stock_data('SPY', start_date=preds.index.min().strftime(date_format), end_date=preds.index.max().strftime(date_format))


prices['price'] = prices['5. adjusted close']
prices = prices['price']

joined = join_preds_prices(preds, prices)

portfolio = compute_portvals(ordersDF=orders, prices=joined['price'], start_val=10000)

print(portfolio)

final_df = pd.DataFrame({'price':joined['price'] / joined['price'].iloc[0], 'value': portfolio / portfolio.iloc[0]})
final_df.plot()
plt.show()


