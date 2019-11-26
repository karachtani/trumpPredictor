from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

for lag in range(1):
    if lag == 0:
        data = pd.read_csv("tweets_stock_data.csv", index_col=0)
    else:
        data = pd.read_csv("lag"+str(lag)+".csv", index_col=0)
    def toint(output):
        return int(output)
    # print(data.columns)
    data['Ouput'] = data['Output']\
        .map(toint)

    #columns are date,time,retweet_count,neg,neu,pos,cmpd,IsTradingDay,numTweets,Output
    #scale columns that are numerical values not labeled classes
    #i dont think we have to scale sent vals because they should be -1 to 1
    sc = StandardScaler()
    data[['time', 'retweet_count', 'numTweets']] = sc.fit_transform(data[['time', 'retweet_count', 'numTweets']])

    y = data['Output']
    X = data[['time', 'retweet_count', 'neg', 'neu', 'pos', 'cmpd', 'IsTradingDay']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    features = list(X_train.columns.values)
    if lag == 1:
        model = MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.95,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=5, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)
    elif lag == 2:
        model = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=6, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)

    elif lag == 3:
        model = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=6, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.99,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)

    elif lag == 4:
        model = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=9, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)

    elif lag == 5:
        model = MLPClassifier(activation='relu', alpha=0.01, batch_size='auto', beta_1=0.95,
                    beta_2=0.99, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=7, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)
    else:
        model = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.95,
                    beta_2=0.99, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=9, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5,
                    random_state=1, shuffle=True, solver='adam', tol=0.0001,
                    validation_fraction=0.1, verbose=False, warm_start=False)


    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(predictions.head(50))

    print("==============================")
    print("LAG " + str(lag))
    print("Accuracy:")
    print(accuracy_score(y_test, predictions))
    print("CLassificaiton Score:")
    print(classification_report(y_test, predictions))

