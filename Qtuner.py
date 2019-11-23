import pandas as pd
from QLearningTrainer import StrategyLearner
from prepForQ import get_q_data
from file_util import save_to_memory_with_fname


def file_name(args=[]):
    return '_'.join(args) + ".csv"

rars = list(range(1, 10, 1))
alphas = list(range(1, 10, 1))
gammas = list(range(1, 10, 1))
epochs = [5, 10, 20, 40, 80, 120, 160, 200, 240]
tickers = ['AAPL', 'AMZN', 'DJIA', 'QQQ', 'SPY', 'VGT', 'XLI']

train_perf_dir = "train_perf"
test_port_dir = "test_portfolios"
test_order_dir = "test_orders"

row_list = []
try:
    for rar in rars:
        for a in alphas:
            for g in gammas:
                for e in epochs:
                    for ticker in tickers:
                        trainer = StrategyLearner(get_q_data(ticker=ticker))
                        train_perf = trainer.train(rar=rar/10, gamma=g/10, alpha=a/10, epochs=e)
                        portfolio, ordersDF = trainer.test()

                        perf = portfolio[-1] / portfolio[0]

                        row_list.append({'rar':rar / 10,
                                        'alpha':a / 10,
                                        'gamma':g / 10,
                                        'epochs':e,
                                        'ticker':ticker,
                                        'perf': perf})

                        args = [rar, a, g, e, ticker]
                        args = list(map(lambda d: str(d), args))

                        f_name = file_name(args)
                        save_to_memory_with_fname(train_perf_dir, f_name, train_perf)
                        save_to_memory_with_fname(test_port_dir, f_name, portfolio)
                        save_to_memory_with_fname(test_order_dir, f_name, ordersDF)
finally:
    save_to_memory_with_fname('q_result', 'results.csv', pd.DataFrame(data=row_list))