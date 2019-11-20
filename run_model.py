from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

data = pd.read_csv("lag1.csv", index_col=0)

y = data['Output'].to_list()
X = data[['time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay']].values.tolist()

parameters = {'hidden_layer_sizes': tuple((element,) for element in range(2, 200, 10)),
                  'max_iter': np.arange(500, 1100, 50)}
clf = model_selection.GridSearchCV(MLPClassifier(solver="adam", random_state=1),
                               parameters,
                               cv=5,
                               verbose=1,
                               n_jobs=4)
clf.fit(X=X, y=y)
print(clf.cv_results_)
