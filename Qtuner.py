import pandas as pd
from QLearningTrainer import StrategyLearner
from prepForQ import get_q_data
from file_util import save_to_memory_with_fname


def file_name(args=[]):
    return '_'.join(args) + ".csv"

rars = list(range(7, 10, 2))
alphas = list(range(1, 10, 1))
gammas = list(range(1, 10, 2))
epochs = [40,60,80]
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

                        row = {'rar':rar / 10,
                                        'alpha':a / 10,
                                        'gamma':g / 10,
                                        'epochs':e,
                                        'ticker':ticker,
                                        'perf': perf}

                        print(row)

                        row_list.append(row)

                        if perf >= 1.0:
                            args = [rar, a, g, e, ticker]
                            args = list(map(lambda d: str(d), args))

                            f_name = file_name(args)
                            save_to_memory_with_fname(train_perf_dir, f_name, train_perf)
                            save_to_memory_with_fname(test_port_dir, f_name, portfolio)
                            save_to_memory_with_fname(test_order_dir, f_name, ordersDF)
finally:
    prev_results = pd.read_csv('q_result/results.csv')
    prev_results = prev_results[['rar', 'alpha', 'gamma', 'epochs', 'ticker', 'perf']]
    results = pd.concat([prev_results, pd.DataFrame(data=row_list)])
    save_to_memory_with_fname('q_result', 'results2.csv', results)